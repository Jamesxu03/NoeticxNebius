"""Orchestration: filters -> score -> (optional take-me-further) -> aggregate. Deterministic.

Taste source per dish, in order: dish.taste in menu_dishes.json; data/processed/menu_dishes_tagged.json
(TasteNet predictions); lexicon baseline (so this runs before any model exists).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from ..schema import Dish, Restaurant, TasteVector
from . import slerp as tmf
from .filters import apply_filters
from .profile import RecoProfile, as_reco
from .score import IngredientSpace, _unit, aggregate, explain, load_space, rank, score_dish, user_centroid

MENU = config.DATA_PROCESSED / "menu_dishes.json"
TAGGED = config.DATA_PROCESSED / "menu_dishes_tagged.json"
RESTAURANTS = config.DATA_PROCESSED / "restaurants_mock.json"
PERSONAS = config.DATA_PROCESSED / "personas.json"


@dataclass
class Catalog:
    dishes: dict[str, Dish]
    restaurants: dict[str, Restaurant]
    space: IngredientSpace | None
    taste_source: dict[str, str]


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


_CACHE: dict = {}


def load_catalog(menu: Path = MENU, tagged: Path = TAGGED, restaurants: Path = RESTAURANTS) -> Catalog:
    key = (str(menu), _mtime(menu), _mtime(tagged), _mtime(restaurants))
    if key in _CACHE:
        return _CACHE[key]
    from ..lexicon import predict_lookup

    raw = json.loads(menu.read_text())
    preds = {d["id"]: d for d in json.loads(tagged.read_text())} if tagged.exists() else {}
    dishes, source = {}, {}
    for d in raw:
        dish = Dish(**d)
        if dish.taste is not None:
            source[dish.id] = dish.tagged_by
        elif dish.id in preds and preds[dish.id].get("taste"):
            p = preds[dish.id]
            # taste fields only; safety fields always come from menu_dishes.json
            dish = dish.model_copy(update={"taste": TasteVector(**p["taste"]),
                                           "taste_confidence": p.get("taste_confidence"),
                                           "taste_model_version": p.get("taste_model_version")})
            source[dish.id] = p.get("tagged_by", "trained-regressor")
        else:
            dish = dish.model_copy(update={"taste": TasteVector.from_list(predict_lookup(dish.model_dump()))})
            source[dish.id] = "lexicon"
        dishes[dish.id] = dish
    rests = {r["id"]: Restaurant(**r) for r in json.loads(restaurants.read_text())}
    cat = Catalog(dishes, rests, load_space(), source)
    _CACHE.clear()
    _CACHE[key] = cat
    return cat


def load_personas(path: Path = PERSONAS) -> dict[str, RecoProfile]:
    return {p["id"]: RecoProfile(**p) for p in json.loads(path.read_text())} if path.exists() else {}


def recommend(profile, occasion: str | None = None, k: int = 10, theta: float = 0.0, direction: str | None = None,
              strict: bool | None = None, budget: float | None = None, party_size: int | None = None,
              catalog: Catalog | None = None, include_drinks: bool = False) -> dict:
    """-> {results: [{restaurant, dishes[], score, explanation, warnings[]}], top_dishes, excluded, message}.

    party_size is accepted for the API contract but does not change ranking yet.
    """
    cat = catalog or load_catalog()
    p = as_reco(profile)
    strict = p.strict if strict is None else strict
    candidates = [d for _, d in sorted(cat.dishes.items()) if include_drinks or d.category != "drink"]

    # 1) hard filters FIRST; scoring below only ever sees `kept`
    kept, excluded = apply_filters(p, candidates, strict=strict)
    warnings = {d.id: w for d, w in kept}
    reasons = Counter(r.split(":")[0] for _, r in excluded)
    if not kept:
        needs = sorted({*(r.strip().lower() for r in p.dietary), *(["allergens"] if p.allergies else [])})
        msg = ("No dishes pass your restrictions yet: our menu data isn't verified for "
               + ", ".join(needs or sorted(reasons)) + ". Switch off strict mode to see options with warnings.")
        return {"results": [], "top_dishes": [], "excluded": len(excluded), "excluded_reasons": dict(reasons),
                "message": msg}

    # 2) score (optionally rotated toward a taste direction)
    space = cat.space
    u_c = user_centroid(p, space)
    theta = float(np.clip(theta, 0, 90)) if direction else 0.0
    u_over, target, lam = None, None, 0.0
    if theta > 0:
        u_over = tmf.shifted_u(p.u(), direction, theta)
        lam = tmf.blend_lambda(theta)
        if space is not None:
            a = u_c
            if a is None:
                cs = [c for c in (space.dish_centroid(d) for d, _ in kept) if c is not None]
                a = _unit(np.mean(cs, axis=0)) if cs else None
            if a is not None:
                target = tmf.rotated_target(a, direction, theta)
    scored = []
    for d, _ in kept:
        s = score_dish(p, d, cat.restaurants[d.restaurant_id], space, occasion, budget, u_override=u_over, user_c=u_c)
        if theta > 0:
            sim = tmf.similarity(space.dish_centroid(d), target) if target is not None else 0.5
            s["terms"] = {**{k_: (1 - lam) * v for k_, v in s["terms"].items()}, "further": lam * sim}
            s["score"] = float(sum(s["terms"].values()))
        s["explanation"] = explain(s, p, occasion)
        s["taste_source"] = cat.taste_source.get(d.id)
        scored.append(s)
    ranked = rank(scored)

    # 3) aggregate per restaurant
    results = aggregate(ranked, cat.dishes, cat.restaurants, warnings, k=k)
    top = [{"dish_id": s["dish_id"], "name": cat.dishes[s["dish_id"]].name,
            "restaurant_id": cat.dishes[s["dish_id"]].restaurant_id, "score": round(s["score"], 4),
            "explanation": s["explanation"], "warnings": warnings[s["dish_id"]], "taste_source": s["taste_source"]}
           for s in ranked[:k]]
    return {"results": results, "top_dishes": top, "excluded": len(excluded), "excluded_reasons": dict(reasons),
            "message": ""}
