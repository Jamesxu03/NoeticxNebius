"""Spend tracking for non-tagging Nebius calls (embeddings), process-safe.

The tagging client (tagging/client.py SpendTracker) owns cache/spend.json and rewrites it from memory,
so embedding spend lives in its own file, cache/spend_embeddings.json, in the same format:
  {"models": {model: {in_tokens, out_tokens, usd, calls}}, "total_usd": float, "updated_at": str}
Calls are appended to the shared cache/calls.jsonl with the tagging client's keys
(ts, model, kind, n, in, out, cost, latency, status). total_usd() sums both spend files.
"""
from __future__ import annotations

import json
import os
import time

from filelock import FileLock

from . import config

TAGGING_SPEND = config.CACHE_DIR / "spend.json"
EMBED_SPEND = config.CACHE_DIR / "spend_embeddings.json"
CALLS_PATH = config.CACHE_DIR / "calls.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def _read(p) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"models": {}, "total_usd": 0.0}


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = config.PRICES.get(model, (0.0, 0.0))
    return (in_tok * pin + out_tok * pout) / 1e6


def add_spend(model: str, in_tok: int, out_tok: int, *, kind: str = "embedding", latency_s: float | None = None,
              n_items: int | None = None) -> float:
    """Record one call; returns the combined running total (tagging + embeddings) in USD."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    usd = cost_of(model, in_tok, out_tok)
    with FileLock(str(EMBED_SPEND) + ".lock"):
        s = _read(EMBED_SPEND)
        m = s["models"].setdefault(model, {"in_tokens": 0, "out_tokens": 0, "usd": 0.0, "calls": 0})
        m["in_tokens"] += in_tok
        m["out_tokens"] += out_tok
        m["usd"] += usd
        m["calls"] += 1
        s["total_usd"] = sum(v["usd"] for v in s["models"].values())
        s["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = EMBED_SPEND.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1))
        os.replace(tmp, EMBED_SPEND)
    with FileLock(str(CALLS_PATH) + ".lock"), CALLS_PATH.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "model": model, "temperature": None, "n": n_items, "kind": kind,
                            "in": in_tok, "out": out_tok, "cost": round(usd, 6),
                            "latency": None if latency_s is None else round(latency_s, 2), "status": "ok"}) + "\n")
    return total_usd()


def total_usd() -> float:
    return float(_read(TAGGING_SPEND).get("total_usd", 0.0)) + float(_read(EMBED_SPEND).get("total_usd", 0.0))


def check_budget(hard: float | None = None) -> None:
    hard = config.MAX_USD if hard is None else hard
    t = total_usd()
    if t >= hard:
        raise BudgetExceeded(f"spend ${t:.2f} >= MAX_USD ${hard:.2f}")
