"""Ingredient-lookup baseline: hand-written lexicon -> 8 taste dims, no learning, no LLM.

predict_lookup(row) accepts a dict, pandas row, or schema.Dish with name/description/ingredients.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .schema import TASTE_DIMS

LEXICON_PATH = Path(__file__).with_name("lexicon.yaml")
KEYWORD_FLAG_NAMES: list[str] = []  # filled on load, one per keyword rule


def _get(row, key, default=None):
    if hasattr(row, "model_dump"):
        row = row.model_dump()
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        v = getattr(row, key, default)
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


@lru_cache(maxsize=1)
def _lexicon():
    lx = yaml.safe_load(LEXICON_PATH.read_text())
    ings = lx["ingredients"]
    keys = sorted(ings, key=len, reverse=True)  # longest match first
    big = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")s?\b", re.I)
    kws = [(re.compile(k["pattern"], re.I), k["delta"]) for k in lx["keywords"]]
    KEYWORD_FLAG_NAMES[:] = [f"kw_{i}" for i in range(len(kws))]
    return lx["defaults"], set(lx["max_pool"]), {k.lower(): v for k, v in ings.items()}, big, kws


def _as_list(v) -> list:
    if v is None or isinstance(v, (str, bytes)):
        return [v] if v else []
    try:
        return list(v)
    except TypeError:
        return []


def dish_text(row) -> str:
    ings = _as_list(_get(row, "ingredients_canonical")) or _as_list(_get(row, "ingredients"))
    return " ".join([str(_get(row, "name", "")), str(_get(row, "description", "")), " ; ".join(map(str, list(ings)))])


def _nutrition(row):
    extra = _get(row, "extra")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = None
    if isinstance(extra, dict):
        n = extra.get("nutrition")
        if isinstance(n, str):
            try:
                n = json.loads(n)
            except json.JSONDecodeError:
                n = None
        if isinstance(n, (list, tuple)) and len(n) >= 7:
            return [float(x) for x in n]
    return None


def lookup_features(row) -> tuple[np.ndarray, np.ndarray]:
    """Returns (8 lexicon scores, keyword flag vector)."""
    defaults, max_pool, ings, big, kws = _lexicon()
    text = dish_text(row)
    hits = {m.group(1).lower() for m in big.finditer(text)}
    per_dim: dict[str, list[float]] = {d: [] for d in TASTE_DIMS}
    for h in hits:
        for d, s in ings[h].items():
            per_dim[d].append(float(s))
    out = []
    for d in TASTE_DIMS:
        vals = sorted(per_dim[d], reverse=True)
        if not vals:
            out.append(float(defaults[d]))
        elif d in max_pool:
            out.append(vals[0])
        else:  # strength-weighted mean: strongest match counts most
            w = np.array([0.5**i for i in range(len(vals))])
            out.append(float(np.dot(w, vals) / w.sum()))
    out = np.array(out)
    name_desc = f"{_get(row, 'name', '')} {_get(row, 'description', '')}"
    flags = np.zeros(len(kws))
    for i, (pat, delta) in enumerate(kws):
        if pat.search(name_desc):
            flags[i] = 1
            for d, dv in delta.items():
                out[TASTE_DIMS.index(d)] += dv
    nut = _nutrition(row)
    if nut is not None:  # Food.com: [cal, fat PDV, sugar PDV, sodium, protein, satfat, carbs]
        ri, sw = TASTE_DIMS.index("richness"), TASTE_DIMS.index("sweet")
        out[ri] = 0.5 * out[ri] + 0.5 * min(10.0, nut[1] / 8.0)
        out[sw] = 0.5 * out[sw] + 0.5 * min(10.0, nut[2] / 12.0)
    return np.clip(out, 0, 10), flags


def predict_lookup(row) -> list[float]:
    return [round(float(x), 2) for x in lookup_features(row)[0]]


def predict_lookup_df(df) -> np.ndarray:
    return np.array([lookup_features(r)[0] for r in df.to_dict("records")])


def lookup_feature_matrix(df) -> np.ndarray:
    rows = [np.concatenate(lookup_features(r)) for r in df.to_dict("records")]
    return np.vstack(rows) if rows else np.zeros((0, len(TASTE_DIMS)))
