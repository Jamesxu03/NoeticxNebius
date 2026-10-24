"""
Deterministic recommender: profile updates, hard filters, scoring.

No LLM call happens here. Allergen and dietary decisions are plain code over
verified data (BRIEF.md 3 and 8.4). Explanation text is a template for now;
swap in a cached LLM rewrite later without touching the scoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from schema.models import (
    TASTE_KEYS, Dish, DishRating, Recommendation, Restaurant, TasteMap,
    TasteVector, UserProfile, VenueChoice, VenueOption, VenueRound,
)

# TBD with the team: occasion list and weights (BRIEF.md 9).
OCCASIONS = {
    "date-night": {"vibe": {"date-night": 1.0, "trendy": 0.5}, "price_shift": 1.0},
    "quick-bite": {"vibe": {"quick": 1.0, "casual": 0.7}, "price_shift": -1.0},
    "family": {"vibe": {"family": 1.0, "traditional": 0.6}, "price_shift": 0.0},
    "celebration": {"vibe": {"trendy": 0.8, "date-night": 0.5}, "price_shift": 1.0},
    "comfort": {"vibe": {"casual": 0.8, "traditional": 0.6}, "price_shift": 0.0},
}

WEIGHTS = {"taste": 0.5, "epicure": 0.15, "price": 0.15, "vibe": 0.1, "novelty": 0.1}
DISLIKE_PENALTY = 0.3
RATING_STEP = 0.25
SURVEY_SIZE = 12


class Catalogue:
    def __init__(self, data_dir: str):
        base = Path(data_dir)
        self.restaurants = {r["id"]: Restaurant(**r) for r in json.loads((base / "restaurants.json").read_text())}
        self.dishes = {d["id"]: Dish(**d) for d in json.loads((base / "dishes.json").read_text())}
        emb_path = base / "dish_embeddings.json"
        emb = json.loads(emb_path.read_text())["dishes"] if emb_path.exists() else {}
        # Epicure dish vectors are optional: missing ones fall back to taste scores only.
        self.vectors = {k: np.array(v["vector"]) for k, v in emb.items() if v.get("vector")}


def taste_array(t: TasteVector) -> np.ndarray:
    return np.array([getattr(t, k) for k in TASTE_KEYS], dtype=float)


# ----- hard filters -----

def hard_filter(dish: Dish, profile: UserProfile) -> tuple[bool, list[str]]:
    """(allowed, warnings). "no" or a listed allergen excludes; "unknown" warns, never passes silently."""
    warnings = []
    for allergy in profile.allergies:
        if allergy in dish.allergens:
            return False, []
    for restriction in profile.dietary:
        status = getattr(dish.dietary, restriction)
        if status == "no":
            return False, []
        if status == "unknown":
            warnings.append(f"{dish.name}: not confirmed {restriction}, check with the restaurant")
    if profile.allergies and not dish.verified:
        warnings.append(f"{dish.name}: allergen info not yet verified")
    for disliked in profile.disliked_ingredients:
        if any(disliked.lower() in ing.lower() for ing in dish.ingredients):
            warnings.append(f"{dish.name}: contains {disliked}")
    return True, warnings


# ----- profile updates -----

def apply_ratings(profile: UserProfile, ratings: list[DishRating], cat: Catalogue) -> None:
    """Stars 4-5 pull the taste vector toward the dish, 1-2 push it away."""
    user = taste_array(profile.taste_vector)
    conf = np.array([getattr(profile.taste_confidence, k) for k in TASTE_KEYS])
    for r in ratings:
        dish = cat.dishes.get(r.dish_id)
        if dish is None:
            continue
        pull = (r.stars - 3) / 2 * RATING_STEP
        user = np.clip(user + pull * (taste_array(dish.taste) - user), 0, 10)
        conf = np.minimum(conf + 0.05, 1)
    profile.taste_vector = TasteVector(**dict(zip(TASTE_KEYS, np.round(user, 2))))
    profile.taste_confidence = type(profile.taste_confidence)(**dict(zip(TASTE_KEYS, np.round(conf, 2))))
    profile.dish_ratings = [*profile.dish_ratings, *ratings]

    liked = [cat.dishes[r.dish_id].adventurousness for r in profile.dish_ratings
             if r.stars >= 4 and r.dish_id in cat.dishes]
    if liked:
        profile.adventurousness = round(float(np.mean([5, *liked])), 1)


def apply_venue_choice(profile: UserProfile, choice: VenueChoice, cat: Catalogue) -> None:
    profile.venue_choices = [*profile.venue_choices, choice]
    chosen = [cat.restaurants[c.chosen] for c in profile.venue_choices if c.chosen in cat.restaurants]
    if not chosen:
        return
    profile.price_pref = round(float(np.mean([r.price_tier for r in chosen])), 2)
    counts = defaultdict(float)
    for r in chosen:
        for tag in r.vibe_tags:
            counts[tag] += 1
    total = sum(counts.values())
    profile.vibe_pref = {k: round(v / total, 2) for k, v in counts.items()}


# ----- survey content -----

def survey_dishes(profile: UserProfile, cat: Catalogue, n: int = SURVEY_SIZE) -> list[Dish]:
    """Greedy farthest-point pick over taste vectors so each rating tells us something new."""
    pool = [d for d in cat.dishes.values() if hard_filter(d, profile)[0] and d.category in ("main", "starter", "dessert")]
    if not pool:
        return []
    picked = [min(pool, key=lambda d: abs(d.adventurousness - 5))]
    while len(picked) < min(n, len(pool)):
        rest = [d for d in pool if d not in picked]
        picked.append(max(rest, key=lambda d: min(
            np.linalg.norm(taste_array(d.taste) - taste_array(p.taste)) for p in picked)))
    return picked


def venue_rounds(profile: UserProfile, cat: Catalogue, n_rounds: int = 4) -> list[VenueRound]:
    """Same dish family at three restaurants with different price tiers."""
    by_family: dict[str, dict[int, Dish]] = defaultdict(dict)
    for d in cat.dishes.values():
        if not hard_filter(d, profile)[0]:
            continue
        tier = cat.restaurants[d.restaurant_id].price_tier
        by_family[d.dish_family].setdefault(tier, d)
    rounds = []
    for family, tiers in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        if len(tiers) < 3:
            continue
        keys = sorted(tiers)
        picks = [tiers[keys[0]], tiers[keys[len(keys) // 2]], tiers[keys[-1]]]
        rounds.append(VenueRound(
            round_id=f"v{len(rounds) + 1}",
            dish_family=family,
            options=[VenueOption(restaurant=cat.restaurants[d.restaurant_id], dish=d) for d in picks],
        ))
        if len(rounds) == n_rounds:
            break
    return rounds


def taste_map(profile: UserProfile) -> TasteMap:
    t = profile.taste_vector
    return TasteMap(
        taste_vector=t,
        taste_confidence=profile.taste_confidence,
        adventurousness=profile.adventurousness,
        price_pref=profile.price_pref,
        vibe_pref=profile.vibe_pref,
        top_dimensions=sorted(TASTE_KEYS, key=lambda k: -getattr(t, k))[:3],
    )


# ----- scoring -----

def epicure_user_vector(profile: UserProfile, cat: Catalogue) -> np.ndarray | None:
    """Star-weighted mean of rated dishes' Epicure vectors (stars-3: liked +, disliked -)."""
    parts = [(r.stars - 3) * cat.vectors[r.dish_id] for r in profile.dish_ratings if r.dish_id in cat.vectors]
    if not parts:
        return None
    v = np.sum(parts, axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def score_dish(dish: Dish, rest: Restaurant, profile: UserProfile, occasion: str,
               user_vec: np.ndarray | None, vectors: dict[str, np.ndarray]) -> float:
    occ = OCCASIONS.get(occasion, {"vibe": {}, "price_shift": 0.0})
    taste = 1 - np.mean(np.abs(taste_array(profile.taste_vector) - taste_array(dish.taste))) / 10
    target_price = np.clip(profile.price_pref + occ["price_shift"], 1, 4)
    price = 1 - abs(rest.price_tier - target_price) / 3
    vibe = sum(profile.vibe_pref.get(t, 0) + occ["vibe"].get(t, 0) for t in rest.vibe_tags) / max(len(rest.vibe_tags), 1)
    novelty = (dish.adventurousness - 5) / 5 * (profile.adventurousness - 5) / 5
    epicure = 0.0
    if user_vec is not None and dish.id in vectors:
        epicure = float(vectors[dish.id] @ user_vec)
    s = (WEIGHTS["taste"] * taste + WEIGHTS["price"] * price + WEIGHTS["vibe"] * min(vibe, 1)
         + WEIGHTS["novelty"] * novelty + WEIGHTS["epicure"] * epicure)
    if any(x.lower() in i.lower() for x in profile.disliked_ingredients for i in dish.ingredients):
        s -= DISLIKE_PENALTY
    return float(s)


def explain(profile: UserProfile, rest: Restaurant, dishes: list[Dish], occasion: str) -> str:
    # TODO: replace with a cached LLM rewrite of these same facts.
    top = sorted(TASTE_KEYS, key=lambda k: -getattr(profile.taste_vector, k))[:2]
    return (f"{dishes[0].name} at {rest.name} matches your love of {top[0]} and {top[1]}, "
            f"and the {', '.join(rest.vibe_tags)} feel suits a {occasion} meal.")


def recommend(profile: UserProfile, cat: Catalogue, occasion: str, budget: int | None = None,
              top_n: int = 5, dishes_per_restaurant: int = 3) -> list[Recommendation]:
    user_vec = epicure_user_vector(profile, cat)
    rated_low = {r.dish_id for r in profile.dish_ratings if r.stars <= 2}

    results = []
    for rest in cat.restaurants.values():
        if budget is not None and rest.price_tier > budget:
            continue
        scored, warnings = [], []
        for dish_id in rest.dish_ids:
            dish = cat.dishes.get(dish_id)
            if dish is None or dish_id in rated_low or dish.category == "drink":
                continue
            ok, w = hard_filter(dish, profile)
            if not ok:
                continue
            scored.append((score_dish(dish, rest, profile, occasion, user_vec, cat.vectors), dish, w))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        top = scored[:dishes_per_restaurant]
        for _, _, w in top:
            warnings.extend(w)
        dishes = [d for _, d, _ in top]
        results.append(Recommendation(
            restaurant=rest,
            dishes=dishes,
            score=round(float(np.mean([s for s, _, _ in top])), 3),
            explanation=explain(profile, rest, dishes, occasion),
            warnings=warnings,
        ))
    results.sort(key=lambda r: -r.score)
    return results[:top_n]
