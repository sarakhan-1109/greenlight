"""
Request/response shapes for the Greenlight API.

These Pydantic models are the *contract* between the React frontend and the
FastAPI backend. They are defined up front (Day 1) and stay stable even as the
prediction internals change (stub -> real XGBoost model on Day 3). Keeping the
contract fixed is what lets us build and deploy the whole pipe before the model
exists.

HARD RULE (anti-leakage): every field in FilmFeatures must be knowable BEFORE a
film is released. No actual revenue, ratings, vote counts, or post-release
popularity ever appears here. See README for why this is the headline design
decision.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Tier(str, Enum):
    """The four box-office outcome tiers the model predicts."""

    FLOP = "Flop"
    MODERATE = "Moderate"
    HIT = "Hit"
    BLOCKBUSTER = "Blockbuster"


class FilmFeatures(BaseModel):
    """Pre-release-only inputs a user supplies for one hypothetical film."""

    budget_usd: float = Field(
        ..., ge=0, description="Production budget in USD (pre-release, known)."
    )
    genre: str = Field(..., description="Primary genre, e.g. 'Action'.")
    lead_star_power: int = Field(
        ...,
        ge=1,
        le=5,
        description="Lead-actor star-power level 1-5 (pre-release popularity proxy).",
    )
    release_month: int = Field(
        ..., ge=1, le=12, description="Planned release month (1=Jan ... 12=Dec)."
    )
    is_franchise: bool = Field(
        ..., description="True if part of an existing franchise / a sequel."
    )
    runtime_min: int = Field(
        ..., ge=40, le=240, description="Planned runtime in minutes."
    )


class FactorContribution(BaseModel):
    """One input's push on the prediction, for the factor-breakdown chart."""

    feature: str = Field(..., description="Human-readable feature name.")
    contribution: float = Field(
        ...,
        description="Signed impact on the prediction (+ pushes up, - pushes down).",
    )


class PredictionResponse(BaseModel):
    """What the frontend renders: verdict + factor chart + analyst's note."""

    tier: Tier
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence in the predicted tier (0-1)."
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probability assigned to each of the four tiers."
    )
    factors: list[FactorContribution] = Field(
        ..., description="Per-feature contributions driving this prediction."
    )
    interpretation: str = Field(
        ..., description="LLM plain-English analyst's note explaining the output."
    )
    model_version: str = Field(
        ..., description="Which model produced this (for honesty/debugging)."
    )
