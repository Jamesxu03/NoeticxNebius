"""Tagging rubric (prompts, few-shots, strict JSON schemas). Bump RUBRIC_VERSION on any prompt change."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from tastelayer.schema import TASTE_DIMENSIONS, TASTE_DIMS, IntTaste

RUBRIC_VERSION = "v1"


class DishTag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str
    taste: IntTaste
    confidence: float
    evidence: str  # one short phrase, <=12 words


class DishTagBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[DishTag]


class IngredientTag(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    taste: IntTaste
    confidence: float


class IngredientTagBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[IngredientTag]


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema with $refs inlined, titles dropped, every object strict."""
    s = model.model_json_schema()
    defs = s.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            out = {k: resolve(v) for k, v in node.items() if k != "title"}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}))
            return out
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(s)


def _anchor_lines() -> str:
    lines = []
    for d in TASTE_DIMS:
        info = TASTE_DIMENSIONS[d]
        lo, mid, hi = info["anchors"]
        lines.append(f"- {d} ({info['meaning']}): low = {lo}, mid = {mid}, high = {hi}")
    return "\n".join(lines)


_SCALE = "Use the anchors exactly: 0 = absent, 2-3 = low anchor, 5 = mid anchor, 8-9 = high anchor, 10 = extreme."
_RULES = (
    "Rules: rate each dimension independently. spice = chilli heat only (not aromatic spices like cumin or "
    "cinnamon). sour includes tomato acidity, tamarind, lime/lemon and yoghurt tang. richness = fat/heaviness. "
    "crunch = textural crispness of the main components. creaminess = smooth/creamy mouthfeel. Use the full scale "
    "and avoid defaulting to 5. Restaurant versions of Indian curries are usually richer than home recipes."
)

DISH_SYSTEM = (
    "You are a professional chef and sensory panel lead. Rate how the DISH AS SERVED tastes to a typical UK diner "
    "on 8 independent 0-10 integer scales.\n"
    f"{_SCALE}\nAnchors:\n{_anchor_lines()}\n{_RULES}\n"
    "For each dish also give confidence (0-1, how sure you are given the information) and evidence: ONE short "
    "phrase of at most 12 words naming the key drivers. Return only JSON."
)

INGREDIENT_SYSTEM = (
    "You are a professional chef and sensory panel lead. Rate the INGREDIENT ITSELF (raw, or as it is typically "
    "used in cooking if it is never eaten raw) on 8 independent 0-10 integer scales.\n"
    f"{_SCALE}\nAnchors:\n{_anchor_lines()}\n"
    "Rules: rate each dimension independently. spice = chilli heat only. Texture dims (crunch, creaminess) describe "
    "the ingredient's own texture as typically eaten: ground spices, liquids, pastes and condiments are 0-1 crunch; "
    "only genuinely creamy things (cream, mascarpone, avocado, hummus) score high creaminess. Use the full scale and "
    "avoid defaulting to 5. For each ingredient also give confidence (0-1). Use the ingredient name exactly as "
    "given. Return only JSON."
)

REVIEW_NOTE = (
    "\n\nNOTE: independent raters strongly disagreed on these dishes. Reason carefully about each dimension "
    "separately, using the anchors, before committing to a number."
)

FEW_SHOTS = [
    ("Margherita pizza", "italian", [0, 2, 4, 1, 6, 5, 4, 4]),
    ("Spaghetti carbonara", "italian", [0, 1, 1, 1, 7, 8, 1, 7]),
    ("Rocket & parmesan salad", "italian", [1, 1, 3, 5, 6, 3, 3, 1]),
    ("Chicken tikka masala", "indian", [4, 3, 3, 1, 5, 7, 0, 7]),
    ("Lamb vindaloo", "indian", [9, 2, 5, 1, 6, 6, 0, 3]),
    ("Dal makhani", "indian", [2, 1, 1, 1, 4, 9, 0, 8]),
    ("Papadum with mango chutney", "indian", [1, 6, 2, 0, 1, 3, 9, 0]),
]


def _few_shot_block() -> str:
    rows = [{"name": n, "cuisine": c, "taste": dict(zip(TASTE_DIMS, v))} for n, c, v in FEW_SHOTS]
    return "Reference ratings (calibration examples):\n" + "\n".join(json.dumps(r) for r in rows)


FEW_SHOT_BLOCK = _few_shot_block()

Kind = Literal["dish", "ingredient"]


def dish_payload(rows: list[dict]) -> str:
    items = []
    for r in rows:
        ings = list(r.get("ingredients") or [])[:25]
        items.append(
            {
                "uid": r["uid"],
                "cuisine": r.get("cuisine", ""),
                "name": r.get("name", ""),
                "description": (r.get("description") or "")[:400],
                "ingredients": ings,
            }
        )
    return json.dumps(items, ensure_ascii=False)


def dish_user_message(payload: str, review: bool = False) -> str:
    msg = f"{FEW_SHOT_BLOCK}\n\nRate these dishes. Return one item per uid, uids exactly as given.\nDishes:\n{payload}"
    return msg + (REVIEW_NOTE if review else "")


def ingredient_payload(names: list[str]) -> str:
    return json.dumps(names, ensure_ascii=False)


def ingredient_user_message(payload: str) -> str:
    return f"Rate these ingredients. Return one item per name, names exactly as given.\nIngredients:\n{payload}"
