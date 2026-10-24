"""
FastAPI backend: the endpoints in BRIEF.md 4.5.

    uvicorn backend.main:app --reload            # from the repo root
    DATA_DIR=data uvicorn backend.main:app        # switch from mock to real data

Profiles live in memory (seeded with the personas) so the demo has no database to break.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend import menu_rank, uber_import
from backend import recommender as rec
from schema.models import (
    Dish, MenuItemResult, MenuRankRequest, ProfileCreate, RatingsIn, Recommendation, RecommendRequest,
    TasteMap, TasteVector, UberImportResult, UserProfile, VenueChoice, VenueRound,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_DIR", "data/mock")

cat = rec.Catalogue(str(ROOT / DATA_DIR))
PROFILES: dict[str, UserProfile] = {
    p["id"]: UserProfile(**p) for p in json.loads((ROOT / "backend/personas/personas.json").read_text())
}

app = FastAPI(title="Noetic taste API")
threading.Thread(target=menu_rank.warm_up, daemon=True).start()  # no-op unless TASTELAYER_URL is set
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
    pid = f"u_{uuid.uuid4().hex[:10]}"  # random: restored browser profiles must never collide
    profile = UserProfile(id=pid, created_at=datetime.now(timezone.utc).isoformat(), **body.model_dump())
    PROFILES[pid] = profile
    return profile


@app.get("/profile/{profile_id}")
def read_profile(profile_id: str) -> UserProfile:
    return get_profile(profile_id)


@app.put("/profile/{profile_id}")
def restore_profile(profile_id: str, body: UserProfile) -> UserProfile:
    """Create or replace a whole profile, e.g. from the browser's saved copy after a backend restart."""
    if body.id != profile_id:
        raise HTTPException(400, "profile id in the URL and body differ")
    PROFILES[profile_id] = body
    return body


@app.put("/profile/{profile_id}/tolerances")
def set_tolerances(profile_id: str, body: TasteVector) -> UserProfile:
    profile = get_profile(profile_id)
    profile.taste_vector = body
    # Sliders give at least 0.3 confidence; keep anything ratings or imports already earned (safe to resend).
    profile.taste_confidence = type(profile.taste_confidence)(
        **{k: max(0.3, getattr(profile.taste_confidence, k)) for k in rec.TASTE_KEYS})
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


@app.post("/profile/{profile_id}/import/uber")
async def import_uber(profile_id: str, file: UploadFile) -> UberImportResult:
    """Warm-start from an Uber Eats export (zip or its Eats order CSV). Read in memory, never stored."""
    profile = get_profile(profile_id)
    data = await file.read(uber_import.MAX_BYTES + 1)
    try:
        items = uber_import.read_orders(data, file.filename or "")
    except (ValueError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error) as exc:
        raise HTTPException(400, str(exc)) from exc
    if not items:
        raise HTTPException(400, "no completed Uber Eats orders found in this file")
    summary = uber_import.apply_orders(profile, items, cat)
    return UberImportResult(profile=profile, summary=summary)


@app.get("/profile/{profile_id}/taste-map")
def taste_map(profile_id: str) -> TasteMap:
    return rec.taste_map(get_profile(profile_id))


@app.post("/recommend")
def recommend(body: RecommendRequest) -> list[Recommendation]:
    return rec.recommend(get_profile(body.profile_id), cat, body.occasion, budget=body.budget)


@app.post("/menu/rank")
def rank_menu(body: MenuRankRequest) -> list[MenuItemResult]:
    """Rank the reviewed lines of a scanned menu: safety rules, taste estimate, Epicure flavour."""
    return menu_rank.rank_menu(get_profile(body.profile_id), body.items, cat)
