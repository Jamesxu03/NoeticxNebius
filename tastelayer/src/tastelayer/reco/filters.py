"""Hard filters: pure code over human-verified dish fields.

Reads ONLY: dish.allergens, dish.dietary, dish.verified, dish.ingredients, dish.ingredients_canonical,
dish.name, dish.description. It never reads model or LLM outputs (scores, suggested allergens,
confidences). Unknown never means safe: strict profiles exclude, non-strict profiles get a warning.

Order per dish: allergies -> dietary restrictions -> disliked ingredients. The first exclusion wins.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..schema import Dish, UserProfile

SUPPORTED_DIETARY = ("vegan", "vegetarian", "halal", "kosher")

ALLERGEN_SYNONYMS = {
    "peanuts": "peanut", "groundnut": "peanut", "groundnuts": "peanut",
    "tree nuts": "nuts", "tree nut": "nuts", "nut": "nuts",
    "almond": "nuts", "almonds": "nuts", "cashew": "nuts", "cashews": "nuts", "pistachio": "nuts",
    "pistachios": "nuts", "walnut": "nuts", "walnuts": "nuts", "hazelnut": "nuts", "hazelnuts": "nuts",
    "pecan": "nuts", "pecans": "nuts",
    "dairy": "milk", "lactose": "milk",
    "eggs": "egg",
    "wheat": "gluten",
    "shellfish": "crustaceans", "prawn": "crustaceans", "shrimp": "crustaceans",
    "soy": "soya", "soybean": "soya",
    "sesame seeds": "sesame", "sesame seed": "sesame",
}


def norm_allergen(s: str) -> str:
    s = s.strip().lower()
    return ALLERGEN_SYNONYMS.get(s, s)


def _term_pattern(term: str) -> re.Pattern:
    # whole-word match with optional plural, so "ham" never matches "graham" or "champagne"
    return re.compile(rf"\b{re.escape(term)}(?:e?s)?\b", re.I)


def disliked_hit(dish: Dish, disliked: Iterable[str]) -> str | None:
    fields = list(dish.ingredients) + list(dish.ingredients_canonical) + [dish.name, dish.description or ""]
    text = " | ".join(fields)
    for term in disliked:
        t = term.strip().lower()
        if t and _term_pattern(t).search(text):
            return t
    return None


def _dietary_flag(dish: Dish, restriction: str) -> str:
    flag = getattr(dish.dietary, restriction)
    if restriction == "vegetarian" and flag != "yes" and dish.dietary.vegan == "yes":
        return "yes"  # vegan implies vegetarian
    return flag


def check_dish(dish: Dish, profile: UserProfile, strict: bool = True) -> tuple[bool, list[str], str | None]:
    """Returns (keep, warnings, exclusion_reason)."""
    warnings: list[str] = []

    allergies = {norm_allergen(a) for a in profile.allergies if a.strip()}
    if allergies:
        dish_allergens = {norm_allergen(a) for a in dish.allergens}
        hit = sorted(allergies & dish_allergens)
        if hit:
            return False, [], f"contains allergen: {', '.join(hit)}"
        if not dish.verified:
            if strict:
                return False, [], "allergens not verified"
            warnings.append("allergens not verified")

    for r in profile.dietary:
        r = r.strip().lower()
        if r not in SUPPORTED_DIETARY:
            if strict:
                return False, [], f"cannot check restriction: {r}"
            warnings.append(f"cannot check restriction: {r}")
            continue
        flag = _dietary_flag(dish, r)
        if flag == "no":
            return False, [], f"not {r}"
        if flag == "unknown" or not dish.verified:
            if strict:
                return False, [], f"unverified: {r}"
            warnings.append(f"unverified: {r}")

    term = disliked_hit(dish, profile.disliked_ingredients)
    if term:
        return False, [], f"disliked ingredient: {term}"

    return True, warnings, None


def apply_filters(profile: UserProfile, dishes: Iterable[Dish], strict: bool = True):
    """-> (kept: list[(dish, warnings)], excluded: list[(dish, reason)]), input order preserved."""
    kept, excluded = [], []
    for d in dishes:
        keep, warnings, reason = check_dish(d, profile, strict)
        if keep:
            kept.append((d, warnings))
        else:
            excluded.append((d, reason))
    return kept, excluded
