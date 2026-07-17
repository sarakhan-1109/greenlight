"""
Prediction seam.

TODAY (Day 1): a deterministic *stub* that returns a plausible-looking result so
the whole frontend->backend pipe can be built and deployed before any ML exists.

DAY 3: this file becomes the only place that loads the trained XGBoost model
(from model/artifacts/) and turns FilmFeatures into a real prediction plus real
per-feature contributions. Because the rest of the app only talks to
`predict()`, swapping the stub for the real model touches nothing else.
"""

from .schemas import FactorContribution, FilmFeatures, PredictionResponse, Tier

MODEL_VERSION = "stub-0.1.0"


def predict(features: FilmFeatures) -> PredictionResponse:
    """Return a stubbed prediction. Replaced by the real XGBoost model on Day 3.

    The stub uses a few obvious heuristics purely so the UI has realistic-shaped
    data to render. It is NOT a real model and makes no accuracy claim.
    """
    # Toy scoring: bigger budget, more star power, and franchise status nudge up.
    score = 0.0
    score += min(features.budget_usd / 200_000_000, 1.0) * 2.0
    score += (features.lead_star_power - 3) * 0.5
    score += 1.0 if features.is_franchise else 0.0
    # Summer (May-Jul) and holiday (Nov-Dec) windows help.
    if features.release_month in (5, 6, 7, 11, 12):
        score += 0.5

    if score >= 2.5:
        tier = Tier.BLOCKBUSTER
    elif score >= 1.5:
        tier = Tier.HIT
    elif score >= 0.5:
        tier = Tier.MODERATE
    else:
        tier = Tier.FLOP

    # Fabricate a probability distribution peaked on the chosen tier.
    order = [Tier.FLOP, Tier.MODERATE, Tier.HIT, Tier.BLOCKBUSTER]
    idx = order.index(tier)
    raw = [1.0 / (1 + abs(i - idx)) for i in range(4)]
    total = sum(raw)
    probabilities = {t.value: round(p / total, 3) for t, p in zip(order, raw)}
    confidence = probabilities[tier.value]

    factors = [
        FactorContribution(feature="Budget", contribution=round(min(features.budget_usd / 200_000_000, 1.0) * 2.0, 2)),
        FactorContribution(feature="Star power", contribution=round((features.lead_star_power - 3) * 0.5, 2)),
        FactorContribution(feature="Franchise", contribution=1.0 if features.is_franchise else 0.0),
        FactorContribution(feature="Release timing", contribution=0.5 if features.release_month in (5, 6, 7, 11, 12) else -0.2),
    ]

    return PredictionResponse(
        tier=tier,
        confidence=confidence,
        probabilities=probabilities,
        factors=factors,
        interpretation="",  # Filled in by the interpreter seam.
        model_version=MODEL_VERSION,
    )
