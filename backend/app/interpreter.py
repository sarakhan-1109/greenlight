"""
Interpretation seam — the LLM layer (Anthropic Claude).

CRITICAL SEPARATION: the LLM does NOT predict. predictor.py (XGBoost) produces
the tier, confidence, and SHAP factor contributions. This module hands that
FINISHED output to Claude and asks only for a plain-English "analyst's note."

Robustness: if no ANTHROPIC_API_KEY is set, or the API call fails, we fall back
to a deterministic template so the endpoint never errors. The prediction is
always the model's; the note is presentation only.
"""

import os

from .schemas import FilmFeatures, PredictionResponse

# Small, fast, cheap — the note is only ~2-3 sentences. Haiku keeps latency low
# in the live demo; swap to a larger model here if richer prose is ever wanted.
_MODEL = "claude-haiku-4-5"

_SYSTEM = (
    "You are a film-industry box-office analyst. A machine-learning model has "
    "ALREADY produced a box-office tier prediction. Your ONLY job is to explain "
    "that existing prediction in 2-3 plain-English sentences for a non-technical "
    "reader. Do NOT change or second-guess the prediction. Do NOT invent numbers "
    "beyond those given. Be concrete, reference the strongest drivers, and keep a "
    "measured, honest tone (this is a pre-release estimate, not a guarantee)."
)


def _template_note(features: FilmFeatures, prediction: PredictionResponse) -> str:
    """Deterministic fallback used when the LLM is unavailable."""
    top = max(prediction.factors, key=lambda f: abs(f.contribution))
    direction = "raising" if top.contribution >= 0 else "lowering"
    conf = round(prediction.confidence * 100)
    return (
        f"The model projects a '{prediction.tier.value}' box-office outcome "
        f"({conf}% confidence), with {top.feature.lower()} {direction} the "
        f"projection most. This is a pre-release estimate based on commercial "
        f"signals only, not a guarantee of performance."
    )


def _build_user_prompt(features: FilmFeatures, prediction: PredictionResponse) -> str:
    """Assemble the prompt from ONLY the model's output + the film's inputs."""
    factors = "\n".join(
        f"  - {f.feature}: {'+' if f.contribution >= 0 else ''}{f.contribution}"
        for f in prediction.factors
    )
    return (
        f"FILM (pre-release inputs):\n"
        f"  - Budget: ${features.budget_usd:,.0f}\n"
        f"  - Genre: {features.genre}\n"
        f"  - Lead star power: {features.lead_star_power}/5\n"
        f"  - Release month: {features.release_month}\n"
        f"  - Franchise/sequel: {'yes' if features.is_franchise else 'no'}\n"
        f"  - Runtime: {features.runtime_min} min\n\n"
        f"MODEL OUTPUT (already decided — explain, don't change):\n"
        f"  - Predicted tier: {prediction.tier.value}\n"
        f"  - Confidence: {round(prediction.confidence * 100)}%\n"
        f"  - Factor contributions (+ pushed the tier up, - pushed it down):\n"
        f"{factors}\n\n"
        f"Write the analyst's note now."
    )


def interpret(features: FilmFeatures, prediction: PredictionResponse) -> str:
    """Return a plain-English note explaining the model's output."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_note(features, prediction)

    try:
        # Imported lazily so the app runs without the SDK/key during dev.
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=_MODEL,
            max_tokens=200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _build_user_prompt(features, prediction)}],
        )
        return message.content[0].text.strip()
    except Exception:
        # Never let the explanation layer break the prediction endpoint.
        return _template_note(features, prediction)
