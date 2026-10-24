"""
Deterministic recommender: profile updates, hard filters, scoring.

Allergen and dietary decisions are plain code over verified data (BRIEF.md 3 and 8.4).
The only LLM touchpoint is `explain()`, which may rewrite facts computed here into nicer
text (backend/llm.py) and falls back to a template when no API key is set.

Epicure (backend/flavour.py) adds a flavour-similarity term to the score, the
"because you liked X" reason, the stretch pick and survey diversity.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend import llm, safety
from backend.flavour import FlavourIndex
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
DISLIKE_FLAVOUR_WEIGHT = 1.0  # a disliked ingredient pulls the flavour vector away like a 1-star dish
ORDER_FLAVOUR_WEIGHT = 2.0  # all imported Uber Eats ingredients together count like one 5-star dish
BECAUSE_MIN_SIM = 0.3  # centred Epicure cosine needed to say "because you liked X"
STRETCH_MIN_SIM = 0.2  # centred Epicure cosine to the user's direction needed for a stretch pick
SURVEY_EPICURE_MIX = 0.5  # share of Epicure distance in survey diversity (rest is taste distance)


class Catalogue:
    def __init__(self, data_dir: str):
        base = Path(data_dir)
        self.restaurants = {r["id"]: Restaurant(**r) for r in json.loads((base / "restaurants.json").read_text())}
        self.dishes = {d["id"]: Dish(**d) for d in json.loads((base / "dishes.json").read_text())}
        # Optional: if the Epicure asset is missing, flavour.available is False and scoring uses taste only.
        self.flavour = FlavourIndex(self.dishes)


def taste_array(t: TasteVector) -> np.ndarray:
    return np.array([getattr(t, k) for k in TASTE_KEYS], dtype=float)


# ----- hard filters -----

def disliked_in(dish: Dish, disliked: list[str]) -> list[str]:
    """Disliked ingredients found in the dish, matched as whole words ("egg" is not "eggplant")."""
    found = []
    for x in disliked:
        if not x.strip():
            continue  # a blank entry would match every word boundary
        pattern = rf"\b{re.escape(x.strip().lower())}(s|es)?\b"
        if any(re.search(pattern, ing.lower()) for ing in [*dish.ingredients, *dish.ingredients_canonical]):
            found.append(x)
    return found


def hard_filter(dish: Dish, profile: UserProfile) -> tuple[bool, list[str]]:
    """(allowed, warnings). A listed or text-implied allergen, or a "no" flag, excludes.

    "unknown" never passes silently: it warns, or excludes when profile.strict is set.
    backend/safety.py can only make a dish less eligible, never more.
    """
    warnings = []
    allergens = set(dish.allergens) | safety.implied_allergens(dish)
    if set(profile.allergies) & allergens:
        return False, []
    for restriction in profile.dietary:
        status, _ = safety.effective_dietary(dish, restriction)
        if status == "no":
            return False, []
        if status == "unknown":
            if profile.strict:
                return False, []
            warnings.append(f"{dish.name}: not confirmed {restriction}, check with the restaurant")
    if profile.allergies and not dish.verified:
        if profile.strict:
            return False, []
        warnings.append(f"{dish.name}: allergen info not yet verified")
    for disliked in disliked_in(dish, profile.disliked_ingredients):
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

def dish_distance(a: Dish, b: Dish, cat: Catalogue) -> float:
    """0-1 distance mixing the 8 taste scores with Epicure flavour (when both dishes have vectors)."""
    taste = np.linalg.norm(taste_array(a.taste) - taste_array(b.taste)) / (10 * np.sqrt(len(TASTE_KEYS)))
    sim = cat.flavour.dish_similarity(a.id, b.id)
    if sim is None:
        return float(taste)
    return float((1 - SURVEY_EPICURE_MIX) * taste + SURVEY_EPICURE_MIX * (1 - sim) / 2)


def survey_dishes(profile: UserProfile, cat: Catalogue, n: int = SURVEY_SIZE) -> list[Dish]:
    """Greedy farthest-point pick over taste + flavour so each rating tells us something new."""
    pool = [d for d in cat.dishes.values() if hard_filter(d, profile)[0] and d.category in ("main", "starter", "dessert")]
    if not pool:
        return []
    picked = [min(pool, key=lambda d: abs(d.adventurousness - 5))]
    while len(picked) < min(n, len(pool)):
        rest = [d for d in pool if d not in picked]
        picked.append(max(rest, key=lambda d: min(dish_distance(d, p, cat) for p in picked)))
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
    """User's direction in centred Epicure space: rated dishes weighted by stars-3, imported
    order ingredients (Uber Eats), minus dislikes."""
    flav = cat.flavour
    if not flav.available:
        return None
    parts = [(r.stars - 3) * flav.dish_vecs[r.dish_id] for r in profile.dish_ratings if r.dish_id in flav.dish_vecs]
    total = sum(profile.ordered_ingredients.values())
    for ing, w in profile.ordered_ingredients.items():
        v = flav.ingredient_vec(ing)
        if v is not None and total > 0:
            parts.append(ORDER_FLAVOUR_WEIGHT * w / total * v)
    for ing in profile.disliked_ingredients:
        v = flav.ingredient_vec(ing)
        if v is not None:
            parts.append(-DISLIKE_FLAVOUR_WEIGHT * v)
    if not parts:
        return None
    v = np.sum(parts, axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else None


def because_you_liked(dish: Dish, profile: UserProfile, cat: Catalogue) -> str | None:
    """Name of the user's 4-5 star dish closest to `dish` in Epicure space, if close enough."""
    best, best_sim = None, BECAUSE_MIN_SIM
    for r in profile.dish_ratings:
        if r.stars < 4 or r.dish_id == dish.id or r.dish_id not in cat.dishes:
            continue
        sim = cat.flavour.dish_similarity(dish.id, r.dish_id)
        if sim is not None and sim > best_sim:
            best, best_sim = cat.dishes[r.dish_id].name, sim
    return best


def score_dish(dish: Dish, rest: Restaurant, profile: UserProfile, occasion: str,
               user_vec: np.ndarray | None, cat: Catalogue) -> float:
    occ = OCCASIONS.get(occasion, {"vibe": {}, "price_shift": 0.0})
    taste = 1 - np.mean(np.abs(taste_array(profile.taste_vector) - taste_array(dish.taste))) / 10
    target_price = np.clip(profile.price_pref + occ["price_shift"], 1, 4)
    price = 1 - abs(rest.price_tier - target_price) / 3
    vibe = sum(profile.vibe_pref.get(t, 0) + occ["vibe"].get(t, 0) for t in rest.vibe_tags) / max(len(rest.vibe_tags), 1)
    novelty = (dish.adventurousness - 5) / 5 * (profile.adventurousness - 5) / 5
    epicure = cat.flavour.similarity(dish.id, user_vec) or 0.0
    s = (WEIGHTS["taste"] * taste + WEIGHTS["price"] * price + WEIGHTS["vibe"] * min(vibe, 1)
         + WEIGHTS["novelty"] * novelty + WEIGHTS["epicure"] * epicure)
    if disliked_in(dish, profile.disliked_ingredients):
        s -= DISLIKE_PENALTY
    return float(s)


def explain(profile: UserProfile, rest: Restaurant, dishes: list[Dish], occasion: str,
            liked: str | None, stretch: bool = False) -> str:
    """Template sentence from computed facts; an LLM may rewrite the same facts (backend/llm.py)."""
    top = sorted(TASTE_KEYS, key=lambda k: -getattr(profile.taste_vector, k))[:2]
    dish = dishes[0]
    if stretch:
        reason = f"since you loved {liked}" if liked else f"for your love of {top[0]}"
        template = (f"Something new: {dish.name} at {rest.name} is outside what you usually pick, "
                    f"but its flavour profile lines up {reason}.")
    elif liked:
        template = (f"{dish.name} at {rest.name} shares a flavour profile with {liked}, which you loved, "
                    f"and the {', '.join(rest.vibe_tags)} feel suits a {occasion} meal.")
    else:
        template = (f"{dish.name} at {rest.name} matches your love of {top[0]} and {top[1]}, "
                    f"and the {', '.join(rest.vibe_tags)} feel suits a {occasion} meal.")
    facts = {
        "restaurant": rest.name, "vibe": rest.vibe_tags, "price_tier": rest.price_tier,
        "dishes": [d.name for d in dishes], "occasion": occasion, "top_tastes": top,
        "similar_to_dish_they_loved": liked, "stretch_pick": stretch,
    }
    return llm.rewrite_explanation(facts, template)


def recommend(profile: UserProfile, cat: Catalogue, occasion: str, budget: int | None = None,
              top_n: int = 5, dishes_per_restaurant: int = 3) -> list[Recommendation]:
    """Top restaurants by mean dish score, plus one Epicure stretch pick (kind="stretch") when possible."""
    user_vec = epicure_user_vector(profile, cat)
    rated_low = {r.dish_id for r in profile.dish_ratings if r.stars <= 2}

    results = []
    safe: list[tuple[Dish, Restaurant, list[str]]] = []
    for rest in cat.restaurants.values():
        if budget is not None and rest.price_tier > budget:
            continue
        scored = []
        for dish_id in rest.dish_ids:
            dish = cat.dishes.get(dish_id)
            if dish is None or dish_id in rated_low or dish.category == "drink":
                continue
            ok, w = hard_filter(dish, profile)
            if not ok:
                continue
            safe.append((dish, rest, w))
            scored.append((score_dish(dish, rest, profile, occasion, user_vec, cat), dish, w))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        top = scored[:dishes_per_restaurant]
        warnings = list(dict.fromkeys(w for _, _, ws in top for w in ws))
        dishes = [d for _, d, _ in top]
        liked = because_you_liked(dishes[0], profile, cat)
        results.append(Recommendation(
            restaurant=rest,
            dishes=dishes,
            score=round(float(np.mean([s for s, _, _ in top])), 3),
            explanation=explain(profile, rest, dishes, occasion, liked),
            warnings=warnings,
            because_you_liked=liked,
        ))
    results.sort(key=lambda r: -r.score)
    results = results[:top_n]

    shown = {d.id for r in results for d in r.dishes}
    stretch = stretch_pick(profile, cat, occasion, user_vec, safe, shown)
    return results + ([stretch] if stretch else [])


def stretch_pick(profile: UserProfile, cat: Catalogue, occasion: str, user_vec: np.ndarray | None,
                 safe: list[tuple[Dish, Restaurant, list[str]]], shown: set[str]) -> Recommendation | None:
    """The safe dish closest to the user's Epicure direction from a dish family they have not rated.

    Only dishes that already passed hard_filter are considered, so this never loosens safety.
    """
    if user_vec is None:
        return None
    rated_families = {cat.dishes[r.dish_id].dish_family for r in profile.dish_ratings if r.dish_id in cat.dishes}
    best = None
    for dish, rest, w in safe:
        if dish.dish_family in rated_families or dish.id in shown or dish.category == "side":
            continue
        if disliked_in(dish, profile.disliked_ingredients):
            continue
        sim = cat.flavour.similarity(dish.id, user_vec)
        if sim is not None and (best is None or sim > best[0]):
            best = (sim, dish, rest, w)
    if best is None or best[0] < STRETCH_MIN_SIM:
        return None
    sim, dish, rest, w = best
    liked = because_you_liked(dish, profile, cat)
    return Recommendation(
        restaurant=rest,
        dishes=[dish],
        score=round(sim, 3),
        explanation=explain(profile, rest, [dish], occasion, liked, stretch=True),
        warnings=list(dict.fromkeys(w)),
        kind="stretch",
        because_you_liked=liked,
    )
