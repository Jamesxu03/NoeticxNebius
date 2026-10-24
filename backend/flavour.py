"""
Epicure flavour space for the recommender (BACKEND_GOALS.md, milestone 2).

Loads the slim committed copy in data/epicure/ (numpy only, no network), builds one vector per
dish from its ingredients, and centres them on the catalogue mean. Raw Epicure dish vectors
are all alike (mean cosine 0.5-0.8), so cosines are only meaningful after centring.

If the asset is missing, `FlavourIndex.available` is False and every method returns None, so
the recommender falls back to the 8 taste scores alone.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET = ROOT / "data/epicure/epicure-chem.npz"
DEFAULT_ALIAS = ROOT / "data/epicure_ingredient_map.json"
MIN_MATCHED = 2


def resolve(name: str, vocab: dict[str, int], alias: dict[str, str | list[str]]) -> list[str]:
    """Map a menu ingredient to Epicure vocab names; [] if it cannot be resolved."""
    key = name.strip().lower()
    if key in alias:
        target = alias[key]
        names = [target] if isinstance(target, str) else target
        return [n for n in names if n in vocab]
    snake = key.replace(" ", "_").replace("-", "_")
    candidates = [snake]
    if snake.endswith("ies"):
        candidates.append(snake[:-3] + "y")
    if snake.endswith("es"):
        candidates.append(snake[:-2])
    if snake.endswith("s"):
        candidates.append(snake[:-1])
    for c in candidates:
        if c in vocab:
            return [c]
    return []


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


class FlavourIndex:
    def __init__(self, dishes: dict, asset: str | Path | None = None, alias: str | Path | None = None):
        """`dishes` maps dish id -> object with an `ingredients` list (schema.models.Dish)."""
        self.available = False
        self.dish_vecs: dict[str, np.ndarray] = {}
        asset = Path(asset or os.environ.get("EPICURE_ASSET", DEFAULT_ASSET))
        if not asset.exists():
            log.warning("Epicure asset %s not found; recommending on taste scores only", asset)
            return
        data = np.load(asset)
        E = data["embeddings"].astype(np.float32)
        self.E = E / np.linalg.norm(E, axis=1, keepdims=True)
        self.vocab = {str(n): i for i, n in enumerate(data["vocab"])}
        self.alias = json.loads(Path(alias or DEFAULT_ALIAS).read_text())

        raw = {}
        for dish_id, dish in dishes.items():
            v = self._mean_vec(dish.ingredients, MIN_MATCHED)
            if v is not None:
                raw[dish_id] = v
        if not raw:
            log.warning("no dish could be embedded with Epicure; recommending on taste scores only")
            return
        self.mean = np.mean(list(raw.values()), axis=0)
        for dish_id, v in raw.items():
            c = _unit(v - self.mean)
            if c is not None:
                self.dish_vecs[dish_id] = c
        self.available = True
        log.info("Epicure: %d/%d dishes embedded from %s", len(self.dish_vecs), len(dishes), asset.name)

    def _mean_vec(self, ingredients: list[str], min_matched: int = 1) -> np.ndarray | None:
        names = [n for ing in ingredients for n in resolve(ing, self.vocab, self.alias)]
        if len({ing for ing in ingredients if resolve(ing, self.vocab, self.alias)}) < min_matched:
            return None
        return _unit(np.mean([self.E[self.vocab[n]] for n in names], axis=0))

    def ingredient_vec(self, name: str) -> np.ndarray | None:
        """Centred unit vector for one free-text ingredient (e.g. a user's dislike)."""
        if not self.available:
            return None
        v = self._mean_vec([name])
        return None if v is None else _unit(v - self.mean)

    def similarity(self, dish_id: str, vec: np.ndarray | None) -> float | None:
        if vec is None or dish_id not in self.dish_vecs:
            return None
        return float(self.dish_vecs[dish_id] @ vec)

    def dish_similarity(self, a: str, b: str) -> float | None:
        if a not in self.dish_vecs or b not in self.dish_vecs:
            return None
        return float(self.dish_vecs[a] @ self.dish_vecs[b])
