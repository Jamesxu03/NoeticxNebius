"""Deterministic dish scoring. Only ever sees dishes that already passed filters.py.

score = 0.65*taste_fit + 0.15*price_fit + 0.10*vibe_fit + 0.10*(adventurousness/10)*novelty
        - 0.1*(1 - mean taste_confidence)          (0.5 assumed when a dish has no confidence)

taste_fit = 1 - sqrt(sum_d w_d (u_d - x_d)^2) / sqrt(sum_d w_d * 100),  w_d = (0.5 + conf_d) * occasion_d
price_fit = 1 - |price_tier - target_price| / 3     (target = price_pref + occasion shift, clipped 1-4)
vibe_fit  = sum(pref[t] for restaurant tags) / sum(pref)   (0.5 if no preference)
novelty   = (1 - cos(dish ingredient centroid, user centroid)) / 2   (0.5 if either missing)
Ties break by dish id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..schema import TASTE_DIMS, Dish, Restaurant
from .profile import RecoProfile

W_TASTE, W_PRICE, W_VIBE, W_NOVEL, W_CONF = 0.65, 0.15, 0.10, 0.10, 0.10
# novelty is ~0.1-0.6 in practice (centred embeddings); thresholds for explanation phrases
NOVEL_HI, NOVEL_LO = 0.48, 0.25

OCCASIONS: dict[str, dict] = {
    "date-night": {"dim_weights": {"richness": 1.2, "creaminess": 1.1}, "vibe_boost": {"date-night": 1.0, "trendy": 0.5},
                   "price_shift": 1.0, "novelty_mult": 1.0},
    "quick-lunch": {"dim_weights": {"richness": 0.8}, "vibe_boost": {"quick": 1.0, "casual": 0.5},
                    "price_shift": -1.0, "novelty_mult": 0.8},
    "family": {"dim_weights": {"spice": 1.3}, "vibe_boost": {"family": 1.0, "casual": 0.5},
               "price_shift": 0.0, "novelty_mult": 0.8},
    "comfort": {"dim_weights": {"richness": 1.3, "creaminess": 1.3}, "vibe_boost": {"casual": 0.5, "traditional": 0.5},
                "price_shift": 0.0, "novelty_mult": 0.5},
    "celebration": {"dim_weights": {}, "vibe_boost": {"trendy": 0.7, "date-night": 0.5},
                    "price_shift": 1.5, "novelty_mult": 1.2},
    "adventurous": {"dim_weights": {}, "vibe_boost": {"trendy": 0.5},
                    "price_shift": 0.0, "novelty_mult": 2.0},
}

DIM_WORDS = {"spice": "heat", "sweet": "sweetness", "sour": "tang", "bitter": "bitter notes",
             "umami": "savoury depth", "richness": "richness", "crunch": "crunch", "creaminess": "creaminess"}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class IngredientSpace:
    """Ingredient embedding lookup (artifacts/ingredient_emb.npy, rows aligned with ingredient_emb_vocab.json)."""
    vectors: np.ndarray
    index: dict[str, int]
    _cache: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def centroid(self, ingredients) -> np.ndarray | None:
        key = tuple(ingredients)
        if key not in self._cache:
            rows = [self.vectors[self.index[i]] for i in key if i in self.index]
            self._cache[key] = _unit(np.mean([_unit(r) for r in rows], axis=0)) if rows else None
        return self._cache[key]

    def dish_centroid(self, dish: Dish) -> np.ndarray | None:
        return self.centroid(list(dish.ingredients_canonical) or [i.lower() for i in dish.ingredients])


@lru_cache(maxsize=1)
def load_space(emb_path: str = str(config.ARTIFACTS / "ingredient_emb.npy"),
               vocab_path: str = str(config.ARTIFACTS / "ingredient_emb_vocab.json")) -> IngredientSpace | None:
    if not (Path(emb_path).exists() and Path(vocab_path).exists()):
        return None
    vocab = json.loads(Path(vocab_path).read_text())
    return IngredientSpace(np.load(emb_path).astype(float), {w: i for i, w in enumerate(vocab)})


def novelty(dish_c: np.ndarray | None, user_c: np.ndarray | None) -> float:
    if dish_c is None or user_c is None:
        return 0.5
    return float((1.0 - float(np.clip(_unit(dish_c) @ _unit(np.asarray(user_c, float)), -1, 1))) / 2.0)


def taste_fit(u: list[float], x: list[float], conf: list[float], dim_weights: dict[str, float] | None = None) -> float:
    dw = dim_weights or {}
    w = np.array([(0.5 + c) * dw.get(d, 1.0) for d, c in zip(TASTE_DIMS, conf)])
    diff = np.asarray(u, float) - np.asarray(x, float)
    return float(1.0 - np.sqrt(np.sum(w * diff ** 2)) / np.sqrt(np.sum(w * 100.0)))


def price_fit(tier: int, price_pref: float, shift: float = 0.0) -> float:
    target = float(np.clip(price_pref + shift, 1, 4))
    return float(1.0 - abs(tier - target) / 3.0)


def vibe_fit(tags: list[str], pref: dict[str, float]) -> float:
    total = sum(v for v in pref.values() if v > 0)
    if total <= 0:
        return 0.5
    return float(min(1.0, sum(max(0.0, pref.get(t, 0.0)) for t in set(tags)) / total))


def vibe_pref_for(profile: RecoProfile, occasion: str | None) -> dict[str, float]:
    pref = dict(profile.vibe_pref)
    for t, b in OCCASIONS.get(occasion or "", {}).get("vibe_boost", {}).items():
        pref[t] = pref.get(t, 0.0) + b
    return pref


def user_centroid(profile: RecoProfile, space: IngredientSpace | None) -> np.ndarray | None:
    if profile.ingredient_centroid:
        return _unit(np.asarray(profile.ingredient_centroid, float))
    if space is not None and profile.liked_ingredients:
        return space.centroid([i.lower() for i in profile.liked_ingredients])
    return None


def dish_taste(dish: Dish) -> list[float]:
    return dish.taste.to_list() if dish.taste is not None else [5.0] * len(TASTE_DIMS)


def score_dish(profile: RecoProfile, dish: Dish, restaurant: Restaurant, space: IngredientSpace | None = None,
               occasion: str | None = None, budget: float | None = None, u_override: list[float] | None = None,
               user_c: np.ndarray | None = None) -> dict:
    occ = OCCASIONS.get(occasion or "", {})
    u = u_override if u_override is not None else profile.u()
    x = dish_taste(dish)
    tf = taste_fit(u, x, profile.conf(), occ.get("dim_weights"))
    pf = price_fit(restaurant.price_tier, profile.price_pref, occ.get("price_shift", 0.0))
    if budget is not None and dish.price_gbp is not None and dish.price_gbp > budget:
        pf = 0.0
    vf = vibe_fit(restaurant.vibe_tags, vibe_pref_for(profile, occasion))
    if user_c is None:
        user_c = user_centroid(profile, space)
    nv = novelty(space.dish_centroid(dish) if space is not None else None, user_c)
    adv = float(np.clip(profile.adventurousness / 10.0 * occ.get("novelty_mult", 1.0), 0, 2))
    tc = dish.taste_confidence
    mean_conf = float(np.mean([tc.get(d, 0.5) for d in TASTE_DIMS])) if tc else 0.5
    terms = {
        "taste": W_TASTE * tf,
        "price": W_PRICE * pf,
        "vibe": W_VIBE * vf,
        "novelty": W_NOVEL * adv * nv,
        "uncertainty": -W_CONF * (1.0 - mean_conf),
    }
    return {
        "dish_id": dish.id,
        "score": float(sum(terms.values())),
        "terms": terms,
        "raw": {"taste_fit": tf, "price_fit": pf, "vibe_fit": vf, "novelty": nv, "adventurousness": adv,
                "mean_taste_confidence": mean_conf},
        "u": list(u),
        "x": x,
    }


def explain(s: dict, profile: RecoProfile, occasion: str | None = None) -> str:
    u, x, raw = s["u"], s["x"], s["raw"]
    parts: list[str] = []
    loves = [d for d, a, b in zip(TASTE_DIMS, u, x) if a >= 6 and b >= 6]
    loves.sort(key=lambda d: -min(u[TASTE_DIMS.index(d)], x[TASTE_DIMS.index(d)]))
    if loves:
        parts.append("matches your love of " + " and ".join(DIM_WORDS[d] for d in loves[:2]))
    else:
        close = sorted(TASTE_DIMS, key=lambda d: (abs(u[TASTE_DIMS.index(d)] - x[TASTE_DIMS.index(d)]), d))
        parts.append("close to your profile on " + " and ".join(DIM_WORDS[d] for d in close[:2]))
    si = TASTE_DIMS.index("spice")
    if u[si] <= 3 and x[si] <= 2:
        parts.append("keeps the heat low")
    if raw["price_fit"] >= 0.8:
        parts.append("within your usual price range")
    elif raw["price_fit"] <= 0.4:
        parts.append("a step away from your usual spend")
    if occasion and raw["vibe_fit"] >= 0.5:
        parts.append(f"fits a {occasion.replace('-', ' ')} vibe")
    if raw["novelty"] >= NOVEL_HI and profile.adventurousness >= 5:
        parts.append("something a bit new for you")
    elif raw["novelty"] <= NOVEL_LO:
        parts.append("built on ingredients you already like")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."


def rank(scored: list[dict]) -> list[dict]:
    return sorted(scored, key=lambda s: (-round(s["score"], 12), s["dish_id"]))


def aggregate(ranked: list[dict], dishes: dict[str, Dish], restaurants: dict[str, Restaurant],
              warnings: dict[str, list[str]], k: int = 10, per_restaurant: int = 3) -> list[dict]:
    """Group ranked dishes by restaurant. Restaurant score = mean of its top-3 dish scores."""
    groups: dict[str, list[dict]] = {}
    for s in ranked:
        rid = dishes[s["dish_id"]].restaurant_id
        groups.setdefault(rid, [])
        if len(groups[rid]) < per_restaurant:
            groups[rid].append(s)
    out = []
    for rid, top in groups.items():
        r = restaurants[rid]
        dish_items = [{
            "dish": dishes[s["dish_id"]].model_dump(),
            "score": round(s["score"], 4),
            "breakdown": {k_: round(v, 4) for k_, v in s["terms"].items()},
            "explanation": s["explanation"],
            "warnings": warnings.get(s["dish_id"], []),
        } for s in top]
        ws = sorted({w for s in top for w in warnings.get(s["dish_id"], [])})
        out.append({
            "restaurant": r.model_dump(),
            "dishes": dish_items,
            "score": round(float(np.mean([s["score"] for s in top])), 4),
            "explanation": f"{r.name}: {top[0]['explanation']}",
            "warnings": ws,
        })
    out.sort(key=lambda o: (-o["score"], o["restaurant"]["id"]))
    return out[:k]
