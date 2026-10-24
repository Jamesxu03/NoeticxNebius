"""
Rank a scanned or typed restaurant menu for one user (POST /menu/rank).

Each reviewed menu line goes through the same pipeline as catalogue dishes:
- safety: backend/safety.py text rules + recommender.hard_filter. A listed or text-implied
  allergen, or a ruled-out diet, excludes the item. Everything else gets the usual warnings,
  because menus never state allergens reliably.
- taste: Arda's TasteNet when its service is running (TASTELAYER_URL, e.g. http://127.0.0.1:8002),
  else TasteLayer's lexicon + tagged catalogue dishes with matching names (backend/uber_import.py).
- flavour: Epicure ingredients in the line vs the user's flavour vector (ratings, orders, dislikes).
No LLM is called and nothing is stored.
"""

from __future__ import annotations

import logging
import os
import re

import httpx
import numpy as np

from backend import recommender as rec
from backend import safety
from backend.uber_import import CUISINE_WORDS, NameMatcher, text_ingredients, text_taste
from schema.models import TASTE_KEYS, Dish, MenuItemIn, MenuItemResult, UserProfile

log = logging.getLogger(__name__)

TASTELAYER_URL = os.environ.get("TASTELAYER_URL", "")
TASTELAYER_TIMEOUT_S = 4.0
WEIGHTS = {"taste": 0.5, "flavour": 0.3, "history": 0.2}
STRONG = 6.0  # a taste dim the user scores this high counts as "loved"
# Short menu lines give noisy flavour vectors ("squid, lemon, aioli" ~ lamb chops at 0.34), so a
# "similar to X you loved" claim needs at least this many recognised ingredients and this similarity.
LIKED_MIN_INGREDIENTS = 3
LIKED_MIN_SIM = 0.4


def _as_dish(item: MenuItemIn) -> Dish:
    """A Dish shell so the catalogue safety code can read the menu line. Flags stay "unknown"."""
    ingredients = [x.strip() for x in re.split(r"[,;/]", item.ingredients) if x.strip()]
    return Dish(id="d_0000", restaurant_id="r_00", name=item.name, description="", cuisine="italian",
                category="main", dish_family="menu", price_gbp=0, ingredients=ingredients, allergens=[],
                dietary={}, taste={}, serving_temp="hot", adventurousness=5, tagged_by="menu-scan")


def _guess_cuisine(text: str) -> str | None:
    text = text.lower()
    for cuisine in ("indian", "italian"):  # the only cuisines TasteNet was trained on
        if any(re.search(rf"\b{re.escape(w)}", text) for w in CUISINE_WORDS[cuisine]):
            return cuisine
    return None


def tastenet_batch(items: list[MenuItemIn]) -> dict[int, np.ndarray]:
    """Index -> TasteNet scores for items it can handle; {} when the service is off or fails."""
    if not TASTELAYER_URL:
        return {}
    payload, index = [], []
    for i, it in enumerate(items):
        cuisine = _guess_cuisine(f"{it.name} {it.ingredients}")
        if cuisine:
            index.append(i)
            payload.append({"name": it.name, "description": it.ingredients, "cuisine": cuisine,
                            "ingredients": [x.strip() for x in it.ingredients.split(",") if x.strip()]})
    if not payload:
        return {}
    try:
        r = httpx.post(f"{TASTELAYER_URL.rstrip('/')}/taste/batch", json=payload, timeout=TASTELAYER_TIMEOUT_S)
        r.raise_for_status()
        return {i: np.array([p["taste"][k] for k in TASTE_KEYS], dtype=float) for i, p in zip(index, r.json())}
    except Exception as exc:  # service down or slow: the local estimate is always available
        log.warning("TasteNet service unavailable, using local estimate: %s", exc)
        return {}


def warm_up() -> None:
    """Load TasteNet in its service ahead of the first real menu (its first call takes several seconds)."""
    if not TASTELAYER_URL:
        return
    try:
        httpx.post(f"{TASTELAYER_URL.rstrip('/')}/taste/batch", timeout=120,
                   json=[{"name": "Margherita pizza", "cuisine": "italian", "ingredients": ["tomato", "mozzarella"]}])
    except Exception as exc:
        log.warning("TasteNet warm-up failed (menus will use the local estimate until it is up): %s", exc)


def _exclusion_reason(dish: Dish, profile: UserProfile) -> str | None:
    hit = sorted((set(dish.allergens) | safety.implied_allergens(dish)) & set(profile.allergies))
    if hit:
        return f"Menu text suggests {', '.join(hit)}"
    for r in profile.dietary:
        flag, why = safety.effective_dietary(dish, r)
        if flag == "no":
            return f"Not {r}: mentions {why}"
    return None


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def rank_menu(profile: UserProfile, items: list[MenuItemIn], cat) -> list[MenuItemResult]:
    matcher = NameMatcher(cat.dishes)
    user = rec.taste_array(profile.taste_vector)
    user_vec = rec.epicure_user_vector(profile, cat)
    loved = [cat.dishes[r.dish_id] for r in profile.dish_ratings if r.stars >= 4 and r.dish_id in cat.dishes]
    tastenet = tastenet_batch(items)
    out = []
    for i, item in enumerate(items):
        dish = _as_dish(item)
        allowed, warnings = rec.hard_filter(dish, profile)
        reason = _exclusion_reason(dish, profile) or (None if allowed else "Not confirmed safe for your profile (strict mode)")

        if i in tastenet:
            est, mask, source = tastenet[i], np.ones(len(TASTE_KEYS), dtype=bool), "tastenet"
        else:
            est, mask = text_taste(item.name, item.ingredients, matcher)
            source = "estimate" if mask.any() else "none"
        coverage = float(mask.mean())
        match = 1 - float(np.mean(np.abs(user - est)[mask])) / 10 if mask.any() else 0.5
        taste_part = coverage * match + (1 - coverage) * 0.5

        ings = text_ingredients(f"{item.name} {item.ingredients}", cat.flavour.vocab, cat.flavour.alias) \
            if cat.flavour.available else set()
        item_vec = cat.flavour.vocab_vec(ings) if ings else None
        flavour = float(item_vec @ user_vec) if item_vec is not None and user_vec is not None else 0.0
        matches = [k for k in profile.ordered_ingredients if k in ings]
        history = min(1.0, len(matches) / 3)

        liked, best = None, LIKED_MIN_SIM
        for d in loved if len(ings) >= LIKED_MIN_INGREDIENTS else []:
            if item_vec is not None and d.id in cat.flavour.dish_vecs:
                sim = float(item_vec @ cat.flavour.dish_vecs[d.id])
                if sim > best:
                    liked, best = d.name, sim

        score = WEIGHTS["taste"] * taste_part + WEIGHTS["flavour"] * (flavour + 1) / 2 + WEIGHTS["history"] * history
        if rec.disliked_in(dish, profile.disliked_ingredients):
            score -= rec.DISLIKE_PENALTY
        fits = [k for k, u, e, m in zip(TASTE_KEYS, user, est, mask) if m and u >= STRONG and e >= STRONG]
        if liked:
            why = f"Similar flavour to {liked}, which you loved."
        elif matches:
            why = f"Shares {', '.join(_pretty(m) for m in matches[:3])} with your usual orders."
        elif fits:
            why = f"Fits your love of {' and '.join(fits[:2])}."
        elif source == "none" and item_vec is None:
            why = "Not much to go on from the menu text, so ask about it."
        else:
            why = "A reasonable fit for your profile."

        out.append(MenuItemResult(
            id=item.id or str(i), name=item.name, ingredients=item.ingredients, price=item.price,
            score=None if reason else round(score, 3), excluded=bool(reason), exclusion_reason=reason,
            warnings=[] if reason else warnings, reason=reason or why,
            matches=[] if reason else [_pretty(m) for m in matches], because_you_liked=None if reason else liked,
            taste={k: round(float(e), 1) for k, e, m in zip(TASTE_KEYS, est, mask) if m}, taste_source=source,
        ))
    return sorted(out, key=lambda r: (r.excluded, -(r.score or 0)))
