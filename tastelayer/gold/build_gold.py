"""Merge gold/labels_*.jsonl -> gold/gold.json and reports/human_agreement.{md,json}.

Final label per dish = mean across annotators (rounded); skipped entries are dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tastelayer import config  # noqa: E402
from tastelayer.schema import TASTE_DIMS  # noqa: E402


def load() -> dict[str, dict[str, dict]]:
    """annotator -> uid -> latest record (non-skipped)."""
    out: dict[str, dict[str, dict]] = {}
    for p in sorted(config.GOLD.glob("labels_*.jsonl")):
        recs: dict[str, dict] = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["uid"]] = r
        out[p.stem.removeprefix("labels_")] = {u: r for u, r in recs.items() if not r.get("skip")}
    return out


def main() -> None:
    by_ann = load()
    if not by_ann:
        sys.exit("no gold/labels_*.jsonl yet")
    cands = {c["uid"]: c for c in json.loads((config.GOLD / "gold_candidates.json").read_text())}
    uids = sorted({u for recs in by_ann.values() for u in recs})
    gold = []
    for u in uids:
        recs = [(a, by_ann[a][u]) for a in sorted(by_ann) if u in by_ann[a]]
        taste = {d: int(round(float(np.mean([r["taste"][d] for _, r in recs])))) for d in TASTE_DIMS}
        c = cands.get(u, {})
        gold.append({
            "uid": u, "name": c.get("name"), "cuisine": c.get("cuisine"), "dish_family": c.get("dish_family"),
            "origin": c.get("origin"), "taste": taste, "annotators": [a for a, _ in recs],
            "n_annotators": len(recs), "unsure": any(r.get("unsure") for _, r in recs),
            "tagged_by": "human", "taste_verified": True,
        })
    (config.GOLD / "gold.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))

    print(f"gold.json: {len(gold)} dishes from {len(by_ann)} annotator(s)")


if __name__ == "__main__":
    main()
