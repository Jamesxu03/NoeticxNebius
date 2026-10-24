"""
Compare the three Epicure siblings on our dishes, to pick the one the recommender uses.

For each model it reports:
  coverage      share of menu ingredients that resolve to the Epicure vocab
  family_gap    mean cosine(same dish_family) - mean cosine(different family), centred vectors
  cuisine_gap   same for cuisine (indian vs italian)
  nn_family     leave-one-out: share of dishes whose nearest other dish has the same family
  taste_rho     Spearman rho between the dish vector's projection on Epicure's taste poles
                and our tagged taste scores (spice~pungent, richness~fatty, sweet, sour, bitter, umami)

Usage
-----
    python data/scripts/compare_epicure.py
    python data/scripts/compare_epicure.py --dishes data/dishes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_dish_embeddings import resolve
from epicure import Epicure

MODELS = {
    "core": ("Kaikaku/epicure-core", "d31ebb5af8e92bbaf5cb67381d5006d4ea8368b7"),
    "chem": ("Kaikaku/epicure-chem", "2461ef3fbafab36d2b1111187a3df98721146861"),
    "cooc": ("Kaikaku/epicure-cooc", "03edd311adde6e39a2eb6f9f3fa78f7396be6b53"),
}
# our taste key -> Epicure supervised property whose "high" pole we project onto
TASTE_POLES = {
    "spice": "pungent_score", "richness": "fatty_score", "sweet": "sweet_score",
    "sour": "sour_score", "bitter": "bitter_score", "umami": "umami_score",
}


def high_pole(m: Epicure, prop: str) -> np.ndarray | None:
    """Direction from the vocab mean toward the ingredients that score high on a property.

    Epicure's modes for a continuous property are clusters of its high-scoring
    ingredients (prop_z_mean > 1), so their members define the "high" end.
    """
    members = {n for mode in m.modes if mode.property == prop for n in mode.members if n in m.vocab}
    if not members:
        return None
    v = np.mean([m.vec(n) for n in members], axis=0) - m.E.mean(axis=0)
    return v / np.linalg.norm(v)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(m: Epicure, dishes: list[dict], alias: dict) -> dict:
    total = matched = 0
    vecs, keep = [], []
    for d in dishes:
        names = []
        for ing in d["ingredients"]:
            total += 1
            r = resolve(ing, m.vocab, alias)
            if r:
                matched += 1
                names.extend(r)
        if len(names) >= 2:
            v = np.mean([m.vec(n) for n in names], axis=0)
            vecs.append(v / np.linalg.norm(v))
            keep.append(d)
    X = np.stack(vecs)
    raw_mean_cos = float((X @ X.T)[np.triu_indices(len(X), 1)].mean())
    Xc = X - X.mean(axis=0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    S = Xc @ Xc.T
    iu = np.triu_indices(len(keep), 1)

    def gap(field: str) -> float:
        same = np.array([keep[i][field] == keep[j][field] for i, j in zip(*iu)])
        return float(S[iu][same].mean() - S[iu][~same].mean())

    np.fill_diagonal(S, -np.inf)
    nn = S.argmax(axis=1)
    nn_family = float(np.mean([keep[i]["dish_family"] == keep[j]["dish_family"] for i, j in enumerate(nn)]))

    rhos = {}
    for key, prop in TASTE_POLES.items():
        pole = high_pole(m, prop)
        if pole is not None:
            rhos[key] = round(spearman(X @ pole, np.array([d["taste"][key] for d in keep])), 2)
    return {
        "coverage": round(matched / total, 2),
        "embedded": f"{len(keep)}/{len(dishes)}",
        "raw_mean_cos": round(raw_mean_cos, 2),
        "family_gap": round(gap("dish_family"), 3),
        "cuisine_gap": round(gap("cuisine"), 3),
        "nn_family": round(nn_family, 2),
        "taste_rho": rhos,
        "taste_rho_mean": round(float(np.mean(list(rhos.values()))), 2) if rhos else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dishes", default="data/mock/dishes.json")
    p.add_argument("--alias", default="data/epicure_ingredient_map.json")
    args = p.parse_args()
    dishes = json.loads(Path(args.dishes).read_text())
    alias = json.loads(Path(args.alias).read_text())
    for name, (repo, rev) in MODELS.items():
        r = evaluate(Epicure.from_pretrained(repo, revision=rev), dishes, alias)
        print(f"{name:5s}", json.dumps(r))


if __name__ == "__main__":
    main()
