"""TasteLayer HTTP API. For now only the trained taste tagger; recommender endpoints get added here later.

Run: uvicorn tastelayer.api:app --port 8000
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model.predict import predict_batch

app = FastAPI(title="TasteLayer API")


class TasteRequest(BaseModel):
    """Minimal dish description; any extra Dish fields are ignored."""
    name: str
    description: str = ""
    cuisine: Literal["indian", "italian"]
    ingredients: list[str] = []
    ingredients_canonical: Optional[list[str]] = None


class TasteResponse(BaseModel):
    taste: dict[str, float]
    taste_confidence: dict[str, float]
    taste_model_version: str
    calibrated: bool


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/taste", response_model=TasteResponse)
def taste(req: TasteRequest) -> TasteResponse:
    try:
        return TasteResponse(**predict_batch([req.model_dump()])[0])
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/taste/batch", response_model=list[TasteResponse])
def taste_batch(reqs: list[TasteRequest]) -> list[TasteResponse]:
    try:
        return [TasteResponse(**p) for p in predict_batch([r.model_dump() for r in reqs])]
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


# ---- Recommender (block E). Deterministic; hard filters run before any scoring. ----
from .reco.profile import RecoProfile  # noqa: E402
from .reco.recommend import load_personas, recommend as _recommend  # noqa: E402


class RecommendRequest(BaseModel):
    profile_id: Optional[str] = None
    profile: Optional[RecoProfile] = None
    occasion: Optional[str] = None
    party_size: Optional[int] = None
    budget: Optional[float] = None
    k: int = 10
    theta: float = 0.0
    direction: Optional[str] = None
    strict: Optional[bool] = None


def _resolve_profile(profile: Optional[RecoProfile], profile_id: Optional[str]) -> RecoProfile:
    if profile is not None:
        return profile
    personas = load_personas()
    if profile_id and profile_id in personas:
        return personas[profile_id]
    raise HTTPException(404, f"unknown profile {profile_id!r}; pass an inline profile or a persona id")


def _run(p: RecoProfile, **kw) -> list[dict]:
    try:
        return _recommend(p, **kw)["results"]
    except ValueError as e:  # e.g. unknown direction
        raise HTTPException(400, str(e)) from e


@app.post("/recommend")
def recommend_post(req: RecommendRequest) -> list[dict]:
    """Returns ranked [{restaurant, dishes[], score, explanation, warnings[]}] (BRIEF 4.5)."""
    p = _resolve_profile(req.profile, req.profile_id)
    return _run(p, occasion=req.occasion, k=req.k, theta=req.theta, direction=req.direction, strict=req.strict,
                budget=req.budget, party_size=req.party_size)


@app.get("/recommend")
def recommend_get(user_id: str, k: int = 10, theta: float = 0.0, direction: Optional[str] = None,
                  occasion: Optional[str] = None, strict: Optional[bool] = None) -> list[dict]:
    return _run(_resolve_profile(None, user_id), occasion=occasion, k=k, theta=theta, direction=direction,
                strict=strict)
