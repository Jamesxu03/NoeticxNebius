"""Build gold/gold_candidates.json (append-only, so re-running never reshuffles labelled dishes).

Base set: 25 Indian + 25 Italian from data/processed/menu_dishes.json, stratified by dish_family
(round-robin across families), 20 of them (10+10) marked overlap for a second annotator.
Extra: up to 10 dishes from reports/high_disagreement.csv (looked up in corpus.parquet), added
whenever that file exists. Only label-free fields are written, so the tool stays blind.

Usage: .venv/bin/python gold/select_candidates.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tastelayer import config  # noqa: E402

OUT = config.GOLD / "gold_candidates.json"
PER_CUISINE = 25
OVERLAP_PER_CUISINE = 10
MAX_DISAGREEMENT = 10


def blind(d: dict, origin: str) -> dict:
    ingredients = d.get("ingredients") or d.get("ingredients_canonical") or d.get("ingredients_raw") or []
    return {
        "uid": d.get("id") or d.get("uid"),
        "name": d["name"],
        "description": d.get("description") or "",
        "cuisine": d["cuisine"],
        "dish_family": d.get("dish_family") or "other",
        "restaurant": d.get("restaurant_name") or d.get("restaurant_id") or d.get("source") or "",
        "ingredients": list(ingredients)[:25],
        "origin": origin,
        "overlap": False,
    }


def stratified(dishes: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for d in dishes:
        by_family[d.get("dish_family") or "other"].append(d)
    for fam in by_family.values():
        rng.shuffle(fam)
    families = sorted(by_family)
    rng.shuffle(families)
    picked: list[dict] = []
    while len(picked) < n and any(by_family.values()):
        for f in families:
            if by_family[f] and len(picked) < n:
                picked.append(by_family[f].pop())
    return picked


def main() -> None:
    rng = random.Random(config.SEED)
    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    have = {c["uid"] for c in existing}
    new: list[dict] = []

    menu_path = config.DATA_PROCESSED / "menu_dishes.json"
    if not any(c["origin"] == "menu" for c in existing):
        if not menu_path.exists():
            sys.exit(f"{menu_path} not found yet: the data block is still building it.")
        menu = json.loads(menu_path.read_text())
        rest_names = {}
        rpath = config.DATA_PROCESSED / "restaurants_mock.json"
        if rpath.exists():
            rest_names = {r["id"]: r["name"] for r in json.loads(rpath.read_text())}
        for cuisine in ("indian", "italian"):
            pool = [d for d in menu if d["cuisine"] == cuisine and d.get("category") != "drink"]
            picks = stratified(pool, PER_CUISINE, rng)
            for i, d in enumerate(picks):
                d = {**d, "restaurant_name": rest_names.get(d.get("restaurant_id"), d.get("restaurant_id"))}
                c = blind(d, "menu")
                c["overlap"] = i < OVERLAP_PER_CUISINE
                new.append(c)
        # interleave cuisines so a short session still covers both
        ind = [c for c in new if c["cuisine"] == "indian"]
        ita = [c for c in new if c["cuisine"] == "italian"]
        new = [x for pair in zip(ind, ita) for x in pair] + ind[len(ita):] + ita[len(ind):]

    hd_path = config.REPORTS / "high_disagreement.csv"
    n_hd = sum(c["origin"] == "high_disagreement" for c in existing)
    if hd_path.exists() and n_hd < MAX_DISAGREEMENT:
        import pandas as pd

        hd = pd.read_csv(hd_path)
        corpus = pd.read_parquet(config.DATA_PROCESSED / "corpus.parquet").set_index("uid")
        std_cols = [c for c in hd.columns if c.endswith("_std")]
        if std_cols:
            hd = hd.assign(_max=hd[std_cols].max(axis=1)).sort_values("_max", ascending=False)
        for uid in hd["uid"]:
            if n_hd >= MAX_DISAGREEMENT:
                break
            if uid in have or uid not in corpus.index or any(c["uid"] == uid for c in new):
                continue
            row = corpus.loc[uid].to_dict()
            row["uid"] = uid
            new.append(blind(row, "high_disagreement"))
            n_hd += 1

    OUT.write_text(json.dumps(existing + new, indent=1, ensure_ascii=False))
    total = existing + new
    print(f"added {len(new)}; total {len(total)} "
          f"(indian {sum(c['cuisine']=='indian' for c in total)}, italian {sum(c['cuisine']=='italian' for c in total)}, "
          f"overlap {sum(c['overlap'] for c in total)}, high-disagreement {sum(c['origin']=='high_disagreement' for c in total)})")


if __name__ == "__main__":
    main()
