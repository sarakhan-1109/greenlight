"""
Interpretation seam (the LLM layer).

CRITICAL SEPARATION: the LLM does NOT predict. The XGBoost model (predictor.py)
produces the tier, confidence, and factor contributions. This module only takes
that finished output and turns it into a plain-English "analyst's note" for a
non-technical reader. Keeping prediction and explanation in separate files makes
that separation obvious and defensible in an interview.

TODAY (Day 1): a template-based stub so the UI shows interpretation text with no
API key required. DAY 4: this becomes a real Anthropic Claude call that receives
the model's structured output and returns the note.
"""

from .schemas import FilmFeatures, PredictionResponse


def interpret(features: FilmFeatures, prediction: PredictionResponse) -> str:
    """Return a plain-English note explaining the model's output.

    Day 1 stub: deterministic template. Day 4: replaced by a Claude call that is
    given ONLY the model's already-computed prediction to explain.
    """
    top_factor = max(prediction.factors, key=lambda f: abs(f.contribution))
    direction = "raising" if top_factor.contribution >= 0 else "lowering"
    conf_pct = round(prediction.confidence * 100)
    return (
        f"The model projects a '{prediction.tier.value}' box-office outcome "
        f"({conf_pct}% confidence). The strongest driver is {top_factor.feature.lower()}, "
        f"{direction} the projection. This is a pre-release estimate based on "
        f"commercial signals only, not a guarantee of performance. "
        f"[Placeholder note — replaced by a Claude-generated analysis on Day 4.]"
    )
