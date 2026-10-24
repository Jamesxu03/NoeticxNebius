"""
Precompute one Epicure flavour vector per dish (offline, run once per dishes.json).

A dish vector is the mean of its ingredients' unit Epicure vectors, re-normalised.
Dishes with fewer than MIN_MATCHED resolvable ingredients get "vector": null and the
recommender falls back to the 8-dim taste scores for them.

Usage
-----
    python data/scripts/build_dish_embeddings.py
    python data/scripts/build_dish_embeddings.py --dishes data/dishes.json --out data/dish_embeddings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epicure import Epicure

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
