"""
Precompute one Epicure flavour vector per dish and report unresolved ingredients.

A dish vector is the mean of its ingredients' unit Epicure vectors, re-normalised.
Dishes with fewer than MIN_MATCHED resolvable ingredients get "vector": null and the
recommender falls back to the 8-dim taste scores for them.

The backend now builds these vectors itself at startup (backend/flavour.py), so the main
use of this script is the unresolved-ingredient list: add those to the alias map.

Usage
-----
    python data/scripts/build_dish_embeddings.py
    python data/scripts/build_dish_embeddings.py --dishes data/dishes.json --out data/dish_embeddings.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from epicure import Epicure

# One ingredient resolver for this script and the backend, so they never disagree.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.flavour import MIN_MATCHED, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dishes", default="data/mock/dishes.json")
    parser.add_argument("--out", default="data/mock/dish_embeddings.json")
    parser.add_argument("--alias", default="data/epicure_ingredient_map.json")
    parser.add_argument("--model", default="Kaikaku/epicure-core")
    args = parser.parse_args()

    m = Epicure.from_pretrained(args.model)
    alias = json.loads(Path(args.alias).read_text())
    dishes = json.loads(Path(args.dishes).read_text())
    factor_modes = [mode for mode in m.modes if mode.kind == "factor"]
    factor_poles = np.stack([mode.pole / np.linalg.norm(mode.pole) for mode in factor_modes])

    out: dict[str, dict] = {}
    unresolved: dict[str, int] = {}
    for d in dishes:
        matched: dict[str, list[str]] = {}
        missing: list[str] = []
        for ing in d["ingredients"]:
            names = resolve(ing, m.vocab, alias)
            if names:
                matched[ing] = names
            else:
                missing.append(ing)
                unresolved[ing] = unresolved.get(ing, 0) + 1

        entry = {
            "matched": matched,
            "unmatched": missing,
            "coverage": round(len(matched) / max(len(d["ingredients"]), 1), 2),
            "vector": None,
            "flavour_group": None,
        }
        if len(matched) >= MIN_MATCHED:
            v = np.mean([m.vec(n) for names in matched.values() for n in names], axis=0)
            v = v / np.linalg.norm(v)
            entry["vector"] = [round(float(x), 5) for x in v]
            entry["flavour_group"] = factor_modes[int(np.argmax(factor_poles @ v))].label
        out[d["id"]] = entry

    payload = {"model": args.model, "dim": int(m.E.shape[1]), "dishes": out}
    Path(args.out).write_text(json.dumps(payload) + "\n")

    embedded = sum(e["vector"] is not None for e in out.values())
    print(f"{embedded}/{len(dishes)} dishes embedded -> {args.out}")
    if unresolved:
        print("Unresolved ingredients (add to the alias map if they matter):")
        for ing, n in sorted(unresolved.items(), key=lambda x: -x[1]):
            print(f"  {n:3d}  {ing}")


if __name__ == "__main__":
    main()
