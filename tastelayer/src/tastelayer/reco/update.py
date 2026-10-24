"""Profile updates from the survey. All functions return a NEW profile (inputs untouched).

init_from_sliders: u = slider values, confidence 0.3 per dim.
rate (stars r on dish x, n = ratings so far):
    alpha = 0.3 / (1 + 0.1 n)
    u <- clip(u + alpha * ((r - 3) / 2) * (x - u), 0, 10)   for r != 3   (r < 3 moves away from x)
    conf_d <- min(1, conf_d + 0.05)                          for r != 3
    if r >= 4: ingredient_centroid <- normalize(0.8 * c + 0.2 * dish_centroid)
venue_choice: price_pref <- 0.7 price_pref + 0.3 tier; vibe_pref[tag] += 0.3 for the chosen venue's
    tags, then normalised to sum 1; plus an implicit 4-star rating on the dish shown in that round.
adventurousness (0-10) = 10 * (0.4 challenge + 0.2 spread + 0.4 novel_like), where
    challenge  = mean(u_spice, u_sour, u_bitter) / 10        (tolerance for challenging flavours)
    spread     = min(1, std(u) / 4)                           (how far sliders move from the middle)
    novel_like = share of rated high-novelty dishes (novelty >= NOVEL_HI = 0.48) given >= 4 stars; 0.5 if none rated
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ..schema import TASTE_DIMS, Dish, DishRating, Restaurant, VenueChoice
from .profile import DEFAULT_CONF, RecoProfile
from .score import NOVEL_HI, IngredientSpace, _unit, dish_taste, novelty, user_centroid

CENTROID_EMA = 0.8
HIGH_NOVELTY = NOVEL_HI


def init_from_sliders(sliders: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    u = {d: float(np.clip(sliders.get(d, 5.0), 0, 10)) for d in TASTE_DIMS}
    return u, {d: DEFAULT_CONF for d in TASTE_DIMS}


def new_profile(pid: str, sliders: dict[str, float], **kw) -> RecoProfile:
    u, c = init_from_sliders(sliders)
    p = RecoProfile(id=pid, taste_vector=u, taste_confidence=c,
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **kw)
    return p.model_copy(update={"adventurousness": adventurousness(p)})


def rate(profile: RecoProfile, dish: Dish, stars: int, space: IngredientSpace | None = None) -> RecoProfile:
    if not 1 <= stars <= 5:
        raise ValueError("stars must be 1-5")
    p = profile.model_copy(deep=True)
    n = len(p.dish_ratings)
    if stars != 3:
        alpha = 0.3 / (1 + n * 0.1)
        u, x = np.array(p.u()), np.array(dish_taste(dish))
        u = np.clip(u + alpha * ((stars - 3) / 2) * (x - u), 0, 10)
        p.taste_vector = {d: float(v) for d, v in zip(TASTE_DIMS, u)}
        p.taste_confidence = {d: min(1.0, c + 0.05) for d, c in zip(TASTE_DIMS, p.conf())}
    if stars >= 4 and space is not None:
        dc = space.dish_centroid(dish)
        if dc is not None:
            c = user_centroid(p, space)
            new_c = dc if c is None else _unit(CENTROID_EMA * c + (1 - CENTROID_EMA) * dc)
            p.ingredient_centroid = [float(v) for v in new_c]
    p.dish_ratings.append(DishRating(dish_id=dish.id, stars=stars))
    return p


def venue_choice(profile: RecoProfile, round_id: str, options: list[str], chosen: Restaurant,
                 dish: Dish | None = None, space: IngredientSpace | None = None) -> RecoProfile:
    p = profile.model_copy(deep=True)
    p.price_pref = float(0.7 * p.price_pref + 0.3 * chosen.price_tier)
    vibe = dict(p.vibe_pref)
    for t in chosen.vibe_tags:
        vibe[t] = vibe.get(t, 0.0) + 0.3
    total = sum(vibe.values())
    p.vibe_pref = {t: v / total for t, v in sorted(vibe.items())} if total > 0 else vibe
    p.venue_choices.append(VenueChoice(round_id=round_id, options=options, chosen=chosen.id))
    if dish is not None:
        p = rate(p, dish, 4, space)
    return p


def adventurousness(profile: RecoProfile, dishes: dict[str, Dish] | None = None,
                    space: IngredientSpace | None = None) -> float:
    u = dict(zip(TASTE_DIMS, profile.u()))
    challenge = (u["spice"] + u["sour"] + u["bitter"]) / 30.0
    spread = min(1.0, float(np.std(list(u.values()))) / 4.0)
    novel_like = 0.5
    if dishes and space is not None:
        c = user_centroid(profile, space)
        novel = [r for r in profile.dish_ratings
                 if r.dish_id in dishes and novelty(space.dish_centroid(dishes[r.dish_id]), c) >= HIGH_NOVELTY]
        if novel:
            novel_like = sum(r.stars >= 4 for r in novel) / len(novel)
    return round(10.0 * (0.4 * challenge + 0.2 * spread + 0.4 * novel_like), 2)
