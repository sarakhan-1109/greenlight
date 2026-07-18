"""
Prediction seam — now backed by the real trained XGBoost model.

Loads the model + metadata produced by model/train.py once at import, then turns
a user's pre-release FilmFeatures into:
  - a predicted tier + confidence + full probability distribution, and
  - the model's OWN per-feature contributions (XGBoost SHAP values) for the
    factor-breakdown chart.

The LLM never runs here. This file is pure ML. See interpreter.py for the note.
"""

import json
import os

import pandas as pd
import xgboost as xgb

from .schemas import FactorContribution, FilmFeatures, PredictionResponse, Tier

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
_MODEL_PATH = os.path.join(_HERE, "artifacts", "greenlight_model.json")
_META_PATH = os.path.join(_HERE, "artifacts", "metadata.json")

with open(_META_PATH) as f:
    META = json.load(f)

FEATURES: list[str] = META["features"]
CATEGORICAL: list[str] = META["categorical"]
GENRE_OPTIONS: list[str] = META["genre_options"]
TIERS: list[str] = META["tiers"]
INFERENCE_YEAR: int = META["inference_year"]

MODEL_VERSION = f"xgb-1.0.0 (acc {META['test_accuracy']*100:.0f}%)"

_booster = xgb.Booster()
_booster.load_model(_MODEL_PATH)

# Human-friendly labels for the factor chart. `year` is excluded because the
# user can't change it (we fix it to the modern era at inference).
_FACTOR_LABELS = {
    "budget": "Budget",
    "star_power_level": "Star power",
    "is_franchise": "Franchise",
    "genre": "Genre",
    "release_month": "Release timing",
    "runtime": "Runtime",
}


def _to_frame(features: FilmFeatures) -> pd.DataFrame:
    """Build the exact one-row feature matrix the model was trained on."""
    genre = features.genre if features.genre in GENRE_OPTIONS else "Other"
    row = {
        "budget": float(features.budget_usd),
        "runtime": int(features.runtime_min),
        "release_month": int(features.release_month),
        "is_franchise": int(features.is_franchise),
        "year": INFERENCE_YEAR,  # predict "as if released in the modern era"
        "genre": genre,
        "star_power_level": int(features.lead_star_power),
    }
    X = pd.DataFrame([row])[FEATURES]
    # Recreate the SAME category ordering used in training so codes line up.
    X["genre"] = pd.Categorical(X["genre"], categories=GENRE_OPTIONS)
    return X


def predict(features: FilmFeatures) -> PredictionResponse:
    """Predict a box-office tier and explain the drivers with SHAP values."""
    X = _to_frame(features)
    dmatrix = xgb.DMatrix(X, enable_categorical=True)

    # Probabilities for the 4 tiers (multi:softprob) -> shape (1, 4).
    probs = _booster.predict(dmatrix)[0]
    pred_idx = int(probs.argmax())
    tier = Tier(TIERS[pred_idx])
    probabilities = {TIERS[i]: round(float(probs[i]), 3) for i in range(len(TIERS))}
    confidence = round(float(probs[pred_idx]), 3)

    # SHAP contributions toward the PREDICTED class. For multiclass, shape is
    # (1, n_class, n_features + 1); the last column is the bias term.
    contribs = _booster.predict(dmatrix, pred_contribs=True)
    class_contribs = contribs[0][pred_idx]  # (n_features + 1,)

    factors = []
    for i, feat in enumerate(FEATURES):
        if feat not in _FACTOR_LABELS:
            continue  # skip `year` (not user-controllable)
        factors.append(
            FactorContribution(
                feature=_FACTOR_LABELS[feat],
                contribution=round(float(class_contribs[i]), 3),
            )
        )
    factors.sort(key=lambda f: abs(f.contribution), reverse=True)

    return PredictionResponse(
        tier=tier,
        confidence=confidence,
        probabilities=probabilities,
        factors=factors,
        interpretation="",  # filled by the interpreter seam
        model_version=MODEL_VERSION,
    )
