"""Nebius (OpenAI-compatible) async client with adaptive rate limits, spend tracking and a budget guard."""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from tastelayer import config

SPEND_PATH = config.CACHE_DIR / "spend.json"
CALLS_LOG = config.CACHE_DIR / "calls.jsonl"
FAILURES_LOG = config.CACHE_DIR / "failures.jsonl"
LIMITS_PATH = config.CACHE_DIR / "ratelimits.json"


# Per-model request extras. gpt-oss and DeepSeek are reasoning models: at default settings ~half of
# 10-dish batches hit max_tokens mid-reasoning (measured on the fixture). These settings keep all
# 10 items valid at ~1.8k (gpt-oss) / ~1.1k (DeepSeek) output tokens.
MODEL_KWARGS: dict[str, dict] = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "deepseek-ai/DeepSeek-V4-Flash-0731": {
        "extra_body": {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    },
}
MAX_TOKENS = 3500


def model_kwargs_tag(model: str) -> str:
    return json.dumps(MODEL_KWARGS.get(model, {}), sort_keys=True)


class BudgetPause(Exception):
    """Spend passed ASK_USD; human must approve (ALLOW_OVER_ASK=1) before continuing."""


class BudgetHardStop(Exception):
    """Spend passed MAX_USD."""


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- spend
class SpendTracker:
    def __init__(self, path: Path = SPEND_PATH):
        self.path = path
        self.data: dict = {"models": {}, "total_usd": 0.0}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        self._last_flush = 0.0

    @property
    def total(self) -> float:
        return float(self.data.get("total_usd", 0.0))

    def add(self, model: str, tin: int, tout: int) -> float:
        pin, pout = config.PRICES.get(model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1e6
        m = self.data["models"].setdefault(model, {"in_tokens": 0, "out_tokens": 0, "usd": 0.0, "calls": 0})
        m["in_tokens"] += tin
        m["out_tokens"] += tout
        m["usd"] += cost
        m["calls"] += 1
        self.data["total_usd"] = sum(v["usd"] for v in self.data["models"].values())
        if time.time() - self._last_flush > 2:
            self.flush()
        return cost

    def flush(self) -> None:
        self.data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(self.path, self.data)
        self._last_flush = time.time()

    def check(self) -> None:
        if self.total >= config.MAX_USD:
            raise BudgetHardStop(f"spend ${self.total:.2f} >= MAX_USD ${config.MAX_USD:.2f}")
        if self.total >= config.ASK_USD and os.getenv("ALLOW_OVER_ASK") != "1":
            raise BudgetPause(
                f"spend ${self.total:.2f} >= ASK_USD ${config.ASK_USD:.2f}; rerun with ALLOW_OVER_ASK=1 to continue"
            )


# ---------------------------------------------------------------- rate limiting
class Bucket:
    """Continuous token bucket; capacity = one minute's worth of `rate_per_min`."""

    def __init__(self, rate_per_min: float):
        self.rate = rate_per_min
        self.level = rate_per_min * 0.2  # gentle start
        self.t = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.level = min(self.rate, self.level + (now - self.t) * self.rate / 60.0)
        self.t = now

    def wait_time(self, amount: float) -> float:
        self._refill()
        amount = min(amount, self.rate)  # never wait forever on an oversized request
        if self.level >= amount:
            return 0.0
        return (amount - self.level) * 60.0 / self.rate

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= min(amount, self.rate)


@dataclass
class AdaptiveLimiter:
    model: str
    rpm: float = 300.0
    tpm: float = 280_000.0
    concurrency: int = 64
    headroom: float = 0.85
    seen: dict = field(default_factory=dict)

    def __post_init__(self):
        self.req = Bucket(self.rpm)
        self.tok = Bucket(self.tpm)
        self.lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(self.concurrency)
        self.blocked_until = 0.0

    async def acquire(self, est_tokens: float) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                w = max(self.blocked_until - now, self.req.wait_time(1), self.tok.wait_time(est_tokens))
                if w <= 0:
                    self.req.take(1)
                    self.tok.take(est_tokens)
                    return
            await asyncio.sleep(min(w, 5.0))

    def update_from_headers(self, headers) -> None:
        h = {k.lower(): v for k, v in headers.items()}
        try:
            lr = float(h.get("x-ratelimit-limit-requests", 0))
            lt = float(h.get("x-ratelimit-limit-tokens", 0))
        except ValueError:
            return
        if lr:
            self.req.rate = max(30.0, lr * self.headroom)
        if lt:
            self.tok.rate = max(20_000.0, lt * self.headroom)
        try:
            rem_t = float(h.get("x-ratelimit-remaining-tokens", lt or 1))
            if lt and rem_t < 0.05 * lt:
                self.blocked_until = max(self.blocked_until, time.monotonic() + 1.0)
        except ValueError:
            pass
        self.seen = {k: v for k, v in h.items() if "ratelimit" in k}

    def on_429(self, retry_after: float | None) -> None:
        self.req.rate = max(30.0, self.req.rate * 0.7)
        self.tok.rate = max(20_000.0, self.tok.rate * 0.7)
        self.blocked_until = max(self.blocked_until, time.monotonic() + (retry_after or 5.0))


# ---------------------------------------------------------------- client
def _retryable(e: BaseException) -> bool:
    if isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
        return True
    return False


@dataclass
class CallResult:
    content: str
    tin: int
    tout: int
    cost: float
    latency: float


class NebiusClient:
    def __init__(self, concurrency: int = 64, spend: SpendTracker | None = None):
        self.client = AsyncOpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.nebius_api_key(), timeout=180)
        self.spend = spend or SpendTracker()
        self.concurrency = concurrency
        self.limiters: dict[str, AdaptiveLimiter] = {}
        self.no_strict: set[str] = set()  # models that rejected json_schema
        self.avg_out: dict[str, float] = {}

    def limiter(self, model: str) -> AdaptiveLimiter:
        if model not in self.limiters:
            self.limiters[model] = AdaptiveLimiter(model, concurrency=self.concurrency)
        return self.limiters[model]

    def save_limits(self) -> None:
        atomic_write_json(
            LIMITS_PATH,
            {m: {"rpm_target": round(l.req.rate), "tpm_target": round(l.tok.rate), "headers": l.seen}
             for m, l in self.limiters.items()},
        )

    async def complete_json(
        self, model: str, system: str, user: str, schema_name: str, schema: dict,
        temperature: float, seed: int, max_tokens: int = MAX_TOKENS, meta: dict | None = None,
    ) -> CallResult:
        self.spend.check()
        lim = self.limiter(model)
        est = (len(system) + len(user)) / 3.5 + self.avg_out.get(model, 1500)
        async with lim.sem:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_retryable),
                wait=wait_random_exponential(multiplier=1, max=30),
                stop=stop_after_attempt(4),
                reraise=True,
            ):
                with attempt:
                    await lim.acquire(est)
                    self.spend.check()
                    return await self._one_call(model, system, user, schema_name, schema, temperature,
                                                seed, max_tokens, lim, meta or {})
        raise RuntimeError("unreachable")

    async def _one_call(self, model, system, user, schema_name, schema, temperature, seed, max_tokens, lim, meta):
        strict = model not in self.no_strict
        if strict:
            rf = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
            u = user
        else:
            rf = {"type": "json_object"}
            u = user + "\n\nReturn JSON matching this schema:\n" + json.dumps(schema)
        t0 = time.time()
        try:
            raw = await self.client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": u}],
                response_format=rf, temperature=temperature, max_tokens=max_tokens, seed=seed,
                **MODEL_KWARGS.get(model, {}),
            )
        except openai.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            lim.on_429(float(ra) if ra and ra.replace(".", "").isdigit() else None)
            self._log(model, temperature, meta, 0, 0, 0.0, time.time() - t0, "429")
            raise
        except openai.BadRequestError as e:
            if strict and ("response_format" in str(e) or "json_schema" in str(e)):
                self.no_strict.add(model)
                self._log(model, temperature, meta, 0, 0, 0.0, time.time() - t0, "no_strict_fallback")
                return await self._one_call(model, system, user, schema_name, schema, temperature, seed,
                                            max_tokens, lim, meta)
            self._log(model, temperature, meta, 0, 0, 0.0, time.time() - t0, f"400:{str(e)[:80]}")
            raise
        except openai.APIError as e:
            if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
                lim.blocked_until = max(lim.blocked_until, time.monotonic() + 3.0)  # back off during outages
            self._log(model, temperature, meta, 0, 0, 0.0, time.time() - t0, f"err:{type(e).__name__}")
            raise
        lim.update_from_headers(raw.headers)
        resp = raw.parse()
        tin = resp.usage.prompt_tokens if resp.usage else 0
        tout = resp.usage.completion_tokens if resp.usage else 0
        self.avg_out[model] = 0.9 * self.avg_out.get(model, tout) + 0.1 * tout
        cost = self.spend.add(model, tin, tout)
        lat = time.time() - t0
        finish = resp.choices[0].finish_reason
        self._log(model, temperature, meta, tin, tout, cost, lat, "ok" if finish != "length" else "ok_truncated")
        return CallResult(resp.choices[0].message.content or "", tin, tout, cost, lat)

    def _log(self, model, temperature, meta, tin, tout, cost, lat, status):
        append_jsonl(CALLS_LOG, {
            "ts": time.time(), "model": model, "temperature": temperature, "n": meta.get("n"),
            "kind": meta.get("kind"), "in": tin, "out": tout, "cost": round(cost, 6),
            "latency": round(lat, 2), "status": status,
        })

    async def close(self):
        self.spend.flush()
        self.save_limits()
        await self.client.close()
