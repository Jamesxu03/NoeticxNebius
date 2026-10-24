"""Recommender-side profile: the contract UserProfile plus optional reco state.

Extra fields (all optional, listed in CONTRACT_CHANGES.md as proposals):
  strict                 True = exclude unverified/unknown dishes for restrictions; False = warn
  ingredient_centroid    unit vector in artifacts/ingredient_emb.npy space (liked dishes, EMA 0.8)
  liked_ingredients      seed ingredients used to initialise the centroid when it is missing
  persona                short label for synthetic test personas
"""
from __future__ import annotations

from typing import Optional

from ..schema import TASTE_DIMS, UserProfile

DEFAULT_TASTE = 5.0
DEFAULT_CONF = 0.3


class RecoProfile(UserProfile):
    strict: bool = True
    ingredient_centroid: Optional[list[float]] = None
    liked_ingredients: list[str] = []
    persona: str = ""

    def u(self) -> list[float]:
        return [float(self.taste_vector.get(d, DEFAULT_TASTE)) for d in TASTE_DIMS]

    def conf(self) -> list[float]:
        return [float(self.taste_confidence.get(d, DEFAULT_CONF)) for d in TASTE_DIMS]


def as_reco(p: UserProfile | dict) -> RecoProfile:
    if isinstance(p, RecoProfile):
        return p
    if isinstance(p, UserProfile):
        return RecoProfile(**p.model_dump())
    return RecoProfile(**p)
