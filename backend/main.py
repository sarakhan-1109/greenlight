"""
Greenlight API — FastAPI entrypoint.

Run locally:  uvicorn main:app --reload --port 8000
Deployed on Render with the same command (see render.yaml).

Endpoints:
  GET  /health       -> liveness check (used by the frontend + Render).
  POST /api/predict  -> takes pre-release FilmFeatures, returns a tier prediction
                        + factor breakdown + analyst's note.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import interpreter, predictor
from app.schemas import FilmFeatures, PredictionResponse

app = FastAPI(
    title="Greenlight API",
    description="Pre-release box-office tier predictor (ML predicts, LLM explains).",
    version="0.1.0",
)

# CORS: the React frontend (Vercel) lives on a different origin than this API
# (Render), so browsers require us to explicitly allow it. ALLOWED_ORIGINS is a
# comma-separated env var set in production; "*" is fine while developing.
_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns ok + which model version is currently serving."""
    return {"status": "ok", "model_version": predictor.MODEL_VERSION}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: FilmFeatures) -> PredictionResponse:
    """Predict a box-office tier, then attach a plain-English interpretation.

    Order matters and encodes the core design: the model predicts FIRST, then
    the interpreter is handed that finished result to explain.
    """
    result = predictor.predict(features)
    result.interpretation = interpreter.interpret(features, result)
    return result
