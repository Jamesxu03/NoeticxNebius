"""Human gold vs 3-model consensus: per-dim agreement, scale bias, and the largest per-dish gaps.

Writes reports/gold_vs_consensus.md. Read-only on gold: it never changes any human label.
Usage: .venv/bin/python gold/compare_consensus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tastelayer import config  # noqa: E402
from tastelayer.schema import TASTE_DIMS as D  # noqa: E402

BIG_GAP = 3  # |human - consensus| >= this is listed per dish
FLAG_BIAS = 1.25  # |mean bias| >= this flags a dim


def main() -> None:
    gold = json.loads((config.GOLD / "gold.json").read_text())
    cons = pd.read_parquet(config.DATA_PROCESSED / "labels_consensus.parquet").set_index("uid")
    g = pd.DataFrame([{"uid": x["uid"], "name": x["name"], "cuisine": x["cuisine"], **x["taste"]} for x in gold])
    m = g.merge(cons[D], left_on="uid", right_index=True, suffixes=("_h", "_c"))
    rows = []
    for d in D:
        h, c = m[f"{d}_h"], m[f"{d}_c"]
        bias = float((h - c).mean())
        rho = spearmanr(h, c).statistic
        within = float(((h - c).abs() <= 1).mean())
        flag = "⚠️ scale differs" if abs(bias) >= FLAG_BIAS else ("⚠️ ranking differs" if rho < 0.45 else "")
        rows.append(f"| {d} | {rho:.2f} | {(h - c).abs().mean():.2f} | {within:.0%} | {bias:+.2f} | {h.mean():.1f} | {c.mean():.1f} | {flag} |")
    gaps = []
    for _, r in m.iterrows():
        for d in D:
            diff = r[f"{d}_h"] - r[f"{d}_c"]
            if abs(diff) >= BIG_GAP:
                gaps.append((abs(diff), r["name"], r["cuisine"], d, r[f"{d}_h"], r[f"{d}_c"]))
    gaps.sort(key=lambda x: (-x[0], x[1], x[3]))
    tag = f"n={len(m)}, single human labeller"
    L = [f"# Human gold vs 3-model consensus ({tag})", "",
         f"n={len(m)} dishes with consensus labels; annotators: "
         f"{', '.join(sorted({a for x in gold for a in x.get('annotators', [])}))}. "
         "Bias = mean(human - consensus). Labels are reported, never altered.", "",
         f"## Per-dimension agreement ({tag})", "",
         "| dim | Spearman | MAE | within ±1 | bias | human mean | consensus mean | flag |",
         "|---|---|---|---|---|---|---|---|", *rows, "",
         f"## Dish-level gaps (|diff| >= {BIG_GAP}): {len(gaps)} of {len(m) * len(D)} cells ({tag})", "",
         "| dish | cuisine | dim | human | consensus |", "|---|---|---|---|---|"]
    L += [f"| {n} | {c} | {d} | {h:g} | {cv:g} |" for _, n, c, d, h, cv in gaps]
    out = config.REPORTS / "gold_vs_consensus.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(gaps)} large gaps)")


if __name__ == "__main__":
    main()
