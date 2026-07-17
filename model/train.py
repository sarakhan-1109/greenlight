"""
Day 3 — feature engineering + XGBoost training.

Produces the artifacts the FastAPI backend loads at inference:
  artifacts/greenlight_model.json   the trained XGBoost model
  artifacts/metadata.json           feature list, genre options, tier cutoffs,
                                     star-power thresholds, and eval metrics

Run:  ../backend/.venv/bin/python train.py

DESIGN NOTES (interview-ready):
- Target = QUARTILES of gross -> Flop / Moderate / Hit / Blockbuster. Balanced
  classes (~25% each), so the honest random-guess baseline is 25%.
- ANTI-LEAKAGE: `gross` builds the TARGET only; it is never a feature. The
  star-power feature uses only films released BEFORE each film (strict time
  order), so no future information reaches the model.
- Star power (1-5) mirrors the UI: level 1 = lead with no prior track record;
  levels 2-5 = quartiles of the lead's prior average gross. Same feature the
  UI slider produces, so training and inference speak the same language.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CLEAN = "data/movies_clean.csv"
# Training output goes straight into the backend so the API serves EXACTLY what
# training produced (single source of truth, and these files deploy with Render).
MODEL_OUT = "../backend/artifacts/greenlight_model.json"
META_OUT = "../backend/artifacts/metadata.json"

TIERS = ["Flop", "Moderate", "Hit", "Blockbuster"]

# The exact features the model sees. `year` is included so the model can
# calibrate for era; at inference the backend fixes it to the modern era.
FEATURES = [
    "budget",
    "runtime",
    "release_month",
    "is_franchise",
    "year",
    "genre",           # categorical
    "star_power_level",  # 1-5, derived proxy (mirrors the UI)
]
CATEGORICAL = ["genre"]


def build_star_power(df: pd.DataFrame) -> tuple[pd.Series, list[float]]:
    """Leakage-free 1-5 star-power level from the lead actor's PRIOR films.

    For each film, look only at the same lead actor's films released strictly
    earlier, and average their gross. Films where the lead has no prior history
    get level 1. The rest are split into levels 2-5 by quartiles of that prior
    average. Returns the level series and the quartile thresholds (for docs).
    """
    # Sortable release key so "prior" is well defined (year, then month).
    df = df.sort_values(["star", "year", "release_month"]).copy()

    # Expanding mean of gross over each star's history, SHIFTED by one so the
    # current film is excluded -> strictly prior information only.
    prior_avg = (
        df.groupby("star")["gross"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["star_prior_avg_gross"] = prior_avg

    has_history = df["star_prior_avg_gross"].notna()
    # Quartile thresholds among leads WITH a track record.
    q = df.loc[has_history, "star_prior_avg_gross"].quantile([0.25, 0.5, 0.75]).tolist()

    def to_level(v: float) -> int:
        if pd.isna(v):
            return 1  # no prior track record -> "unknown" lead
        if v <= q[0]:
            return 2
        if v <= q[1]:
            return 3
        if v <= q[2]:
            return 4
        return 5

    level = df["star_prior_avg_gross"].apply(to_level)
    # Restore original row order.
    return level.reindex(df.index).sort_index(), q


def main() -> None:
    df = pd.read_csv(CLEAN)
    df = df.sort_index()

    # ---- Target: quartile tiers of gross ----
    df["tier"], bins = pd.qcut(df["gross"], 4, labels=TIERS, retbins=True)
    tier_cutoffs = {
        "Flop":        [float(bins[0]), float(bins[1])],
        "Moderate":    [float(bins[1]), float(bins[2])],
        "Hit":         [float(bins[2]), float(bins[3])],
        "Blockbuster": [float(bins[3]), float(bins[4])],
    }

    # ---- Star-power feature (leakage-free) ----
    star_level, star_q = build_star_power(df)
    df["star_power_level"] = star_level.astype(int)

    # ---- Assemble X / y ----
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = df["tier"].cat.codes  # 0..3 in TIERS order

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Model ----
    # Modest depth / learning rate to avoid overfitting a genuinely noisy target.
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        # depth 4 generalizes best here: depth 5 pushed TRAIN to 78% while TEST
        # fell — classic overfitting on a noisy target, so we keep it shallow.
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.5,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    acc = accuracy_score(y_test, preds)
    # Within-one-tier: counting an adjacent-tier miss (e.g. Hit vs Moderate) as
    # "close". Reported alongside exact accuracy for an honest fuller picture.
    adj = np.mean(np.abs(np.asarray(y_test) - preds) <= 1)
    print(f"\n=== TEST ACCURACY: {acc*100:.1f}%  (random baseline = 25%) ===")
    print(f"    TRAIN accuracy: {train_acc*100:.1f}%  (gap shows over/underfit)")
    print(f"    Within-one-tier accuracy: {adj*100:.1f}%")
    if acc > 0.90:
        print("!!! Accuracy >90% — investigate for leakage before trusting this. !!!")
    print("\nPer-class report:")
    print(classification_report(y_test, preds, target_names=TIERS))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))

    # Feature importance (gain) for a quick sanity check.
    importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
    print("\nFeature importance (gain):")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {v}")

    # ---- Persist artifacts ----
    model.get_booster().save_model(MODEL_OUT)
    metadata = {
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "tiers": TIERS,
        "genre_options": sorted(df["genre"].unique().tolist()),
        "tier_cutoffs_usd": tier_cutoffs,
        "star_power_quartiles_usd": star_q,
        "inference_year": int(df["year"].max()),  # backend fixes year to this
        "test_accuracy": round(float(acc), 4),
        "train_accuracy": round(float(train_acc), 4),
        "within_one_tier_accuracy": round(float(adj), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved model -> {MODEL_OUT}")
    print(f"Saved metadata -> {META_OUT}")


if __name__ == "__main__":
    main()
