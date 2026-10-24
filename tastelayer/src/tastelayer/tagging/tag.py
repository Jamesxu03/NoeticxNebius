"""Batch tagging of dishes and ingredients across Nebius models. Cached, resumable, budget-guarded."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import diskcache
import pandas as pd
from pydantic import ValidationError

from tastelayer import config
from tastelayer.schema import TASTE_DIMS
from tastelayer.tagging import rubric
from tastelayer.tagging.client import (
    FAILURES_LOG,
    model_kwargs_tag,
    BudgetHardStop,
    BudgetPause,
    NebiusClient,
    append_jsonl,
    atomic_write_json,
)

CORPUS = config.DATA_PROCESSED / "corpus.parquet"
MENU = config.DATA_PROCESSED / "menu_dishes.json"
VOCAB = config.ARTIFACTS / "ingredient_vocab.json"
LABELS = config.DATA_PROCESSED / "labels.parquet"
ING_LABELS = config.DATA_PROCESSED / "ingredient_taste_llm.parquet"
SHARDS = config.DATA_INTERIM / "labels"
ING_SHARDS = config.DATA_INTERIM / "ingredient_labels"
PROGRESS = config.CACHE_DIR / "progress.json"
DISAGREE = config.REPORTS / "high_disagreement.csv"

MAX_ATTEMPTS = 3
REPORT_EVERY_S = 600


def short(model: str) -> str:
    return config.MODEL_SHORT.get(model, model.split("/")[-1].lower())


# ---------------------------------------------------------------- input loading
def load_dish_rows(corpus: Path = CORPUS, include_menu: bool = True, menu_only: bool = False) -> list[dict]:
    rows: dict[str, dict] = {}
    if include_menu and MENU.exists():
        for d in json.loads(MENU.read_text()):
            rows[d["id"]] = {
                "uid": d["id"], "cuisine": d.get("cuisine", ""), "name": d.get("name", ""),
                "description": d.get("description", ""),
                "ingredients": d.get("ingredients_canonical") or d.get("ingredients") or [],
            }
    if not menu_only and corpus.exists():
        df = pd.read_parquet(corpus, columns=["uid", "cuisine", "name", "description", "ingredients_canonical"])
        for r in df.itertuples(index=False):
            if r.uid in rows:
                continue
            ings = list(r.ingredients_canonical) if r.ingredients_canonical is not None else []
            rows[r.uid] = {"uid": r.uid, "cuisine": r.cuisine, "name": r.name,
                           "description": r.description or "", "ingredients": ings}
    # menu dishes (d_*) first so they finish early; deterministic order otherwise
    return sorted(rows.values(), key=lambda r: (not r["uid"].startswith("d_"), r["uid"]))


def chunk(xs: list, n: int) -> list[list]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


# ---------------------------------------------------------------- parsing
_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def extract_json(content: str):
    s = _FENCE.sub("", content.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            return json.loads(s[i : j + 1])
        raise


def clamp_int(x) -> int:
    try:
        return int(min(10, max(0, round(float(x)))))
    except (TypeError, ValueError):
        return 0


def parse_items(content: str, kind: str, want_keys: list[str]) -> dict[str, dict]:
    """Return {key: item} for valid items whose key is in want_keys. Raises on unparseable output."""
    obj = extract_json(content)
    batch_cls = rubric.DishTagBatch if kind == "dish" else rubric.IngredientTagBatch
    key_field = "uid" if kind == "dish" else "name"
    # clamp before validation so slightly-off numbers don't fail the whole batch
    for it in obj.get("items", []) if isinstance(obj, dict) else []:
        if isinstance(it.get("taste"), dict):
            it["taste"] = {d: clamp_int(it["taste"].get(d, 0)) for d in TASTE_DIMS}
        try:
            it["confidence"] = float(min(1, max(0, float(it.get("confidence", 0.5)))))
        except (TypeError, ValueError):
            it["confidence"] = 0.5
        if kind == "dish" and not isinstance(it.get("evidence"), str):
            it["evidence"] = ""
    batch = batch_cls.model_validate(obj)
    norm = {k.strip().lower(): k for k in want_keys}
    out: dict[str, dict] = {}
    for it in batch.items:
        k = norm.get(getattr(it, key_field).strip().lower())
        if k is not None and k not in out:
            out[k] = it.model_dump()
    return out


# ---------------------------------------------------------------- job execution
@dataclass
class Progress:
    total: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)
    cached: dict = field(default_factory=dict)
    failed_keys: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    kind: str = "dish"

    def bump(self, model, what, n=1):
        d = getattr(self, what)
        d[model] = d.get(model, 0) + n

    def dump(self, client: NebiusClient | None = None) -> None:
        atomic_write_json(PROGRESS, {
            "kind": self.kind, "started": self.started, "updated": time.time(),
            "total_batches": self.total, "done_batches": self.done, "cached_batches": self.cached,
            "failed_keys": self.failed_keys,
            "limits": {m: {"rpm_target": round(l.req.rate), "tpm_target": round(l.tok.rate)}
                       for m, l in (client.limiters.items() if client else [])},
        })


def cache_key(model: str, temperature: float, kind: str, review: bool, keys: list[str], payload: str,
              system: str = "") -> str:
    sys_hash = hashlib.sha256(system.encode()).hexdigest()[:16]  # prompt edits invalidate the cache
    raw = "|".join([model, model_kwargs_tag(model), rubric.RUBRIC_VERSION, sys_hash, f"{temperature:.2f}", kind,
                    str(review), ",".join(sorted(keys)), payload])
    return hashlib.sha256(raw.encode()).hexdigest()


class Tagger:
    def __init__(self, client: NebiusClient, cache: diskcache.Cache, progress: Progress):
        self.client, self.cache, self.progress = client, cache, progress

    def _messages(self, kind: str, items: list, review: bool) -> tuple[str, str, str, list[str]]:
        if kind == "dish":
            payload = rubric.dish_payload(items)
            return rubric.DISH_SYSTEM, rubric.dish_user_message(payload, review), payload, [r["uid"] for r in items]
        payload = rubric.ingredient_payload(items)
        return rubric.INGREDIENT_SYSTEM, rubric.ingredient_user_message(payload), payload, list(items)

    async def tag(self, model: str, temperature: float, kind: str, items: list, review: bool = False,
                  depth: int = 0) -> dict[str, dict]:
        system, user, payload, keys = self._messages(kind, items, review)
        key = cache_key(model, temperature, kind, review, keys, payload, system)
        hit = self.cache.get(key)
        if hit is not None:
            if depth == 0:
                self.progress.bump(model, "cached")
            return hit
        schema = rubric.strict_schema(rubric.DishTagBatch if kind == "dish" else rubric.IngredientTagBatch)
        got: dict[str, dict] = {}
        remaining = list(items)
        for attempt in range(MAX_ATTEMPTS):
            if not remaining:
                break
            system, user, _, rkeys = self._messages(kind, remaining, review)
            try:
                res = await self.client.complete_json(
                    model, system, user, f"{kind}_batch", schema, temperature,
                    seed=config.SEED + attempt, meta={"n": len(remaining), "kind": kind},
                )
                parsed = parse_items(res.content, kind, rkeys)
            except (BudgetPause, BudgetHardStop):
                raise
            except (ValidationError, json.JSONDecodeError, ValueError, AttributeError) as e:
                append_jsonl(FAILURES_LOG, {"model": model, "kind": kind, "keys": rkeys, "attempt": attempt,
                                            "error": f"parse:{type(e).__name__}"})
                continue
            except Exception as e:  # noqa: BLE001  network errors after client retries
                append_jsonl(FAILURES_LOG, {"model": model, "kind": kind, "keys": rkeys, "attempt": attempt,
                                            "error": f"{type(e).__name__}:{str(e)[:120]}"})
                continue
            got.update(parsed)
            key_of = (lambda r: r["uid"]) if kind == "dish" else (lambda r: r)
            remaining = [r for r in remaining if key_of(r) not in got]
        if remaining:
            if len(remaining) > 1:
                half = len(remaining) // 2
                for part in (remaining[:half], remaining[half:]):
                    got.update(await self.tag(model, temperature, kind, part, review, depth + 1))
            else:
                k = remaining[0]["uid"] if kind == "dish" else remaining[0]
                append_jsonl(FAILURES_LOG, {"model": model, "kind": kind, "keys": [k], "error": "gave_up"})
                self.progress.failed_keys.setdefault(short(model), []).append(k)
        if len(got) == len(keys):
            self.cache.set(key, got)
        return got


# ---------------------------------------------------------------- shard writer
class ShardWriter:
    def __init__(self, folder: Path, prefix: str, flush_every: int = 2000):
        self.folder, self.prefix, self.flush_every = folder, prefix, flush_every
        self.buf: list[dict] = []
        self.n = 0
        folder.mkdir(parents=True, exist_ok=True)

    def add(self, rows: list[dict]) -> None:
        self.buf.extend(rows)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        path = self.folder / f"{self.prefix}_{int(time.time() * 1000)}_{self.n:05d}.parquet"
        pd.DataFrame(self.buf).to_parquet(path, index=False)
        self.buf, self.n = [], self.n + 1


def consolidate(folder: Path, out: Path, key_cols: list[str]) -> pd.DataFrame:
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def dish_records(got: dict[str, dict], model_name: str, temperature: float, pass_name: str) -> list[dict]:
    rows = []
    for uid, it in got.items():
        r = {"uid": uid, "model": model_name, "temperature": temperature, "pass": pass_name}
        r.update({d: int(it["taste"][d]) for d in TASTE_DIMS})
        r.update({"confidence": float(it["confidence"]), "evidence": it.get("evidence", ""),
                  "rubric_version": rubric.RUBRIC_VERSION})
        rows.append(r)
    return rows


def ingredient_records(got: dict[str, dict], model_name: str) -> list[dict]:
    rows = []
    for name, it in got.items():
        r = {"ingredient": name, "model": model_name}
        r.update({d: int(it["taste"][d]) for d in TASTE_DIMS})
        r["confidence"] = float(it["confidence"])
        r["rubric_version"] = rubric.RUBRIC_VERSION
        rows.append(r)
    return rows


# ---------------------------------------------------------------- runners
@dataclass
class Job:
    model: str
    temperature: float
    items: list
    model_name: str
    pass_name: str = "main"
    review: bool = False


async def _run_jobs(jobs: list[Job], kind: str, writer: ShardWriter, concurrency: int) -> int:
    """Run jobs concurrently (per-model semaphores inside the client). Returns exit code."""
    from tastelayer.tagging import report

    client = NebiusClient(concurrency=concurrency)
    cache = diskcache.Cache(str(config.CACHE_DIR / "llm"), size_limit=8 * 2**30)
    progress = Progress(kind=kind)
    for j in jobs:
        progress.bump(j.model_name, "total")
    tagger = Tagger(client, cache, progress)
    stop: dict = {"reason": None}

    async def one(j: Job):
        if stop["reason"]:
            return
        try:
            got = await tagger.tag(j.model, j.temperature, kind, j.items, j.review)
        except (BudgetPause, BudgetHardStop) as e:
            stop["reason"] = stop["reason"] or e
            return
        if kind == "dish":
            writer.add(dish_records(got, j.model_name, j.temperature, j.pass_name))
        else:
            writer.add(ingredient_records(got, j.model_name))
        progress.bump(j.model_name, "done")

    async def heartbeat():
        last_report = time.time()
        while True:
            await asyncio.sleep(30)
            progress.dump(client)
            client.save_limits()
            client.spend.flush()
            if time.time() - last_report >= REPORT_EVERY_S:
                writer.flush()
                try:
                    report.write_summary()
                except Exception as e:  # noqa: BLE001
                    print(f"[report] failed: {e}")
                last_report = time.time()

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(one(j) for j in jobs))
    finally:
        hb.cancel()
        writer.flush()
        progress.dump(client)
        await client.close()
        cache.close()
    total = client.spend.total
    if isinstance(stop["reason"], BudgetHardStop):
        print(f"HARD STOP: {stop['reason']}")
        return 4
    if isinstance(stop["reason"], BudgetPause):
        print(f"PAUSED FOR APPROVAL: {stop['reason']}")
        return 3
    print(f"done. total spend so far ${total:.4f}")
    return 0


def tag_dishes(limit: int | None = None, uids_file: Path | None = None, menu_only: bool = False,
               models: list[str] | None = None, temps: list[float] | None = None,
               concurrency: int = 64, corpus: Path = CORPUS) -> int:
    rows = load_dish_rows(corpus, menu_only=menu_only)
    if uids_file:
        want = {u.strip() for u in Path(uids_file).read_text().split() if u.strip()}
        rows = [r for r in rows if r["uid"] in want]
    if limit:
        rows = rows[:limit]
    batches = chunk(rows, config.DISHES_PER_REQUEST)
    models = models or config.TAGGING_MODELS
    temps = temps or config.TAG_TEMPERATURES
    # interleave: batch-major so partial runs cover whole dishes across all models/temps
    jobs = [Job(m, t, b, short(m)) for b in batches for m in models for t in temps]
    print(f"tag-dishes: {len(rows)} dishes, {len(batches)} batches x {len(models)} models x {len(temps)} temps "
          f"= {len(jobs)} requests (before cache)")
    code = asyncio.run(_run_jobs(jobs, "dish", ShardWriter(SHARDS, "main"), concurrency))
    consolidate(SHARDS, LABELS, ["uid", "model", "temperature", "pass"])
    return code


def tag_ingredients(limit: int | None = None, models: list[str] | None = None, concurrency: int = 64,
                    vocab_path: Path = VOCAB) -> int:
    vocab = json.loads(Path(vocab_path).read_text())
    names = [k for k, _ in sorted(vocab.items(), key=lambda kv: (-kv[1], kv[0]))][: limit or config.N_INGREDIENTS_TO_TAG]
    batches = chunk(names, config.INGREDIENTS_PER_REQUEST)
    models = models or config.TAGGING_MODELS
    jobs = [Job(m, 0.2, b, short(m)) for b in batches for m in models]
    print(f"tag-ingredients: {len(names)} ingredients, {len(jobs)} requests (before cache)")
    code = asyncio.run(_run_jobs(jobs, "ingredient", ShardWriter(ING_SHARDS, "ing"), concurrency))
    consolidate(ING_SHARDS, ING_LABELS, ["ingredient", "model"])
    return code


def second_pass(concurrency: int = 64, corpus: Path = CORPUS) -> int:
    if not DISAGREE.exists():
        print("no reports/high_disagreement.csv; run consensus first")
        return 1
    want = set(pd.read_csv(DISAGREE)["uid"].astype(str))
    rows = [r for r in load_dish_rows(corpus) if r["uid"] in want]
    batches = chunk(rows, config.DISHES_PER_REQUEST)
    m = config.SECOND_PASS_MODEL
    jobs = [Job(m, 0.2, b, short(m) + "-review", pass_name="review", review=True) for b in batches]
    print(f"second-pass: {len(rows)} dishes, {len(jobs)} requests with {m}")
    code = asyncio.run(_run_jobs(jobs, "dish", ShardWriter(SHARDS, "review"), concurrency))
    consolidate(SHARDS, LABELS, ["uid", "model", "temperature", "pass"])
    return code
