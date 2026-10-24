"""reports/tagging_summary.md: progress, spend, failures, rate limits, agreement."""
from __future__ import annotations

import itertools
import json
import time

import pandas as pd
from scipy.stats import spearmanr

from tastelayer import config
from tastelayer.schema import TASTE_DIMS
from tastelayer.tagging.client import CALLS_LOG, LIMITS_PATH, SPEND_PATH
from tastelayer.tagging.tag import LABELS, PROGRESS, SHARDS

SUMMARY = config.REPORTS / "tagging_summary.md"
MIN_FOR_AGREEMENT = 30


def _load_json(p, default):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _live_labels() -> pd.DataFrame:
    files = sorted(SHARDS.glob("*.parquet"))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df.drop_duplicates(subset=["uid", "model", "temperature", "pass"], keep="last")
    return pd.read_parquet(LABELS) if LABELS.exists() else pd.DataFrame()


def _calls() -> pd.DataFrame:
    if not CALLS_LOG.exists():
        return pd.DataFrame()
    return pd.read_json(CALLS_LOG, lines=True)


def _spearman(a, b) -> float:
    if len(a) < 3 or pd.Series(a).nunique() < 2 or pd.Series(b).nunique() < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def agreement_tables(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = labels[labels["pass"] == "main"]
    per_model = main.groupby(["uid", "model"])[TASTE_DIMS].mean().reset_index()
    models = sorted(per_model["model"].unique())
    inter = {}
    for a, b in itertools.combinations(models, 2):
        A = per_model[per_model.model == a].set_index("uid")
        B = per_model[per_model.model == b].set_index("uid")
        common = A.index.intersection(B.index)
        if len(common) < MIN_FOR_AGREEMENT:
            continue
        inter[f"{a} vs {b}"] = {d: _spearman(A.loc[common, d], B.loc[common, d]) for d in TASTE_DIMS}
    inter_df = pd.DataFrame(inter)
    if not inter_df.empty:
        inter_df["mean_pair"] = inter_df.mean(axis=1)
    self_ag = {}
    temps = sorted(main["temperature"].unique())
    if len(temps) >= 2:
        for m in models:
            M = main[main.model == m]
            lo = M[M.temperature == temps[0]].set_index("uid")
            hi = M[M.temperature == temps[-1]].set_index("uid")
            common = lo.index.intersection(hi.index)
            if len(common) < MIN_FOR_AGREEMENT:
                continue
            row = {d: _spearman(lo.loc[common, d], hi.loc[common, d]) for d in TASTE_DIMS}
            diff = (lo.loc[common, TASTE_DIMS] - hi.loc[common, TASTE_DIMS]).abs()
            row["within_1"] = float((diff <= 1).to_numpy().mean())
            self_ag[m] = row
    return inter_df, pd.DataFrame(self_ag)


def _fmt(df: pd.DataFrame) -> str:
    if df.empty:
        return "_not enough overlapping labels yet_"
    return df.round(3).to_markdown()


def write_summary() -> str:
    spend = _load_json(SPEND_PATH, {"models": {}, "total_usd": 0.0})
    prog = _load_json(PROGRESS, {})
    limits = _load_json(LIMITS_PATH, {})
    calls = _calls()
    labels = _live_labels()
    now = time.time()
    L = [f"# Tagging summary\n\n_updated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n"]

    total = spend.get("total_usd", 0.0)
    L.append(f"## Spend\n**Total ${total:.3f}** (ask at ${config.ASK_USD:.0f}, hard stop at ${config.MAX_USD:.0f})\n")
    L.append("| model | calls | in tokens | out tokens | USD |\n|---|---|---|---|---|")
    for m, v in sorted(spend.get("models", {}).items()):
        L.append(f"| {m} | {v['calls']} | {v['in_tokens']:,} | {v['out_tokens']:,} | {v['usd']:.3f} |")

    if prog:
        el = max(1.0, now - prog.get("started", now))
        L.append(f"\n## Progress (current run: {prog.get('kind')}, {el / 60:.1f} min elapsed)\n")
        L.append("| model | batches done | cached | total | % | batches/min | ETA min |\n|---|---|---|---|---|---|---|")
        for m, tot in sorted(prog.get("total_batches", {}).items()):
            done = prog.get("done_batches", {}).get(m, 0)
            cached = prog.get("cached_batches", {}).get(m, 0)
            fresh = max(0, done - cached)
            rate = fresh / (el / 60)
            eta = (tot - done) / rate if rate > 0 else float("nan")
            L.append(f"| {m} | {done} | {cached} | {tot} | {100 * done / max(tot, 1):.1f} | {rate:.1f} | {eta:.1f} |")
        fk = prog.get("failed_keys", {})
        if fk:
            L.append("\nPermanently failed items: " + ", ".join(f"{m}: {len(v)}" for m, v in fk.items()))

    if not calls.empty:
        L.append("\n## Calls\n")
        g = calls.groupby("model")
        tab = pd.DataFrame({
            "calls": g.size(),
            "ok_rate": g["status"].apply(lambda s: s.astype(str).str.startswith("ok").mean()),
            "truncated": g["status"].apply(lambda s: (s == "ok_truncated").sum()),
            "429s": g["status"].apply(lambda s: (s == "429").sum()),
            "p50_latency_s": g["latency"].median(),
            "usd_per_call": g["cost"].mean(),
        })
        L.append(tab.round(4).to_markdown())

    if limits:
        L.append("\n## Rate limits (latest headers)\n\n| model | limit req/min | limit tok/min | our rpm target | our tpm target |\n|---|---|---|---|---|")
        for m, v in sorted(limits.items()):
            h = v.get("headers", {})
            L.append(f"| {m} | {h.get('x-ratelimit-limit-requests', '?')} | {h.get('x-ratelimit-limit-tokens', '?')} "
                     f"| {v.get('rpm_target')} | {v.get('tpm_target')} |")

    if not labels.empty:
        L.append(f"\n## Labels\n{labels['uid'].nunique():,} dishes with at least one label; "
                 f"{len(labels):,} label rows.\n")
        L.append(labels.groupby(["model", "temperature", "pass"]).size().rename("rows").to_frame().to_markdown())
        inter, self_ag = agreement_tables(labels)
        L.append("\n### Inter-model Spearman per dim (labels averaged over temperatures)\n")
        L.append(_fmt(inter))
        L.append("\n### Self-agreement, temperature 0.2 vs 0.7 (Spearman per dim, within ±1 share)\n")
        L.append(_fmt(self_ag))

    text = "\n".join(L) + "\n"
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(text)
    return text


def watch(every: int = 600) -> None:
    while True:
        write_summary()
        print(f"[{time.strftime('%H:%M:%S')}] wrote {SUMMARY}")
        time.sleep(every)
