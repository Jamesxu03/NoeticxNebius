"""
FastAPI backend: the endpoints in BRIEF.md 4.5.

    uvicorn backend.main:app --reload            # from the repo root
    DATA_DIR=data uvicorn backend.main:app        # switch from mock to real data

Profiles live in memory (seeded with the personas) so the demo has no database to break.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import recommender as rec
from schema.models import (
    Dish, ProfileCreate, RatingsIn, Recommendation, RecommendRequest,
    TasteMap, TasteVector, UserProfile, VenueChoice, VenueRound,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_DIR", "data/mock")

cat = rec.Catalogue(str(ROOT / DATA_DIR))
PROFILES: dict[str, UserProfile] = {
    p["id"]: UserProfile(**p) for p in json.loads((ROOT / "backend/personas/personas.json").read_text())
}

app = FastAPI(title="Noetic taste API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_profile(profile_id: str) -> UserProfile:
    if profile_id not in PROFILES:
        raise HTTPException(404, f"profile {profile_id} not found")
    return PROFILES[profile_id]


@app.get("/taste-dimensions")
def taste_dimensions() -> dict:
    return json.loads((ROOT / "data/taste_dimensions.json").read_text())


@app.post("/profile")
def create_profile(body: ProfileCreate) -> UserProfile:
    pid = f"u_{len(PROFILES) + 1:03d}"
    profile = UserProfile(id=pid, created_at=datetime.now(timezone.utc).isoformat(), **body.model_dump())
    PROFILES[pid] = profile
    return profile


@app.get("/profile/{profile_id}")
def read_profile(profile_id: str) -> UserProfile:
    return get_profile(profile_id)


@app.put("/profile/{profile_id}/tolerances")
def set_tolerances(profile_id: str, body: TasteVector) -> UserProfile:
    profile = get_profile(profile_id)
    profile.taste_vector = body
    profile.taste_confidence = type(profile.taste_confidence)(**{k: 0.3 for k in rec.TASTE_KEYS})
    return profile


@app.get("/survey/dishes")
def survey_dishes(profile_id: str) -> list[Dish]:
    return rec.survey_dishes(get_profile(profile_id), cat)


@app.post("/profile/{profile_id}/ratings")
def post_ratings(profile_id: str, body: RatingsIn) -> UserProfile:
    profile = get_profile(profile_id)
    rec.apply_ratings(profile, body.ratings, cat)
    return profile


@app.get("/survey/venue-rounds")
def venue_rounds(profile_id: str) -> list[VenueRound]:
    return rec.venue_rounds(get_profile(profile_id), cat)


@app.post("/profile/{profile_id}/venue-choice")
def post_venue_choice(profile_id: str, body: VenueChoice) -> UserProfile:
    profile = get_profile(profile_id)
    rec.apply_venue_choice(profile, body, cat)
    return profile


@app.get("/profile/{profile_id}/taste-map")
def taste_map(profile_id: str) -> TasteMap:
    return rec.taste_map(get_profile(profile_id))


@app.post("/recommend")
def recommend(body: RecommendRequest) -> list[Recommendation]:
    return rec.recommend(get_profile(body.profile_id), cat, body.occasion, budget=body.budget)
