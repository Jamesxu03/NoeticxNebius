"""'Take me further' slider: rotate the user's ingredient centroid toward a taste direction.

a = user ingredient centroid (falls back to the mean of candidate dishes' centroids)
b = normalize(a + taste_directions[dim]); rotated = slerp(a, b) by theta degrees (embed/slerp.py)
Candidates are ALREADY filtered. Final score = (1 - lam) * base_score(u') + lam * sim,
lam = 0.5 * theta / 90, sim = (1 + cos(dish_centroid, rotated)) / 2, u'_dim = min(10, u_dim + 3 theta/90).
theta = 0 gives lam = 0 and u' = u, i.e. exactly the base ranking.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config
from ..embed.slerp import rotate_toward, shifted_taste_target, slerp  # noqa: F401  (re-exported)
from ..schema import TASTE_DIMS

MAX_LAMBDA = 0.5


@lru_cache(maxsize=1)
def load_directions(path: str = str(config.ARTIFACTS / "taste_directions.json"), embedding: str = "ours") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    block = data.get("embeddings", {}).get(embedding, {})
    return {d: np.asarray(v, float) for d, v in block.get("directions", {}).items()}


def rotated_target(a: np.ndarray, direction: str, theta: float, directions: dict | None = None) -> np.ndarray:
    dirs = directions if directions is not None else load_directions()
    if direction not in dirs:
        raise ValueError(f"unknown direction {direction!r}; choose from {sorted(dirs)}")
    return rotate_toward(a, dirs[direction], theta)


def shifted_u(u: list[float], direction: str, theta: float) -> list[float]:
    shifted = shifted_taste_target(dict(zip(TASTE_DIMS, u)), direction, theta)
    return [shifted[d] for d in TASTE_DIMS]


def blend_lambda(theta: float) -> float:
    return MAX_LAMBDA * float(np.clip(theta, 0, 90)) / 90.0


def similarity(dish_c: np.ndarray | None, target: np.ndarray) -> float:
    if dish_c is None:
        return 0.5
    return float((1.0 + float(np.clip(dish_c @ target, -1, 1))) / 2.0)
