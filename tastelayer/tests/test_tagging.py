"""Offline tests for tagging validation, split-on-failure, caching, budget guard and consensus."""
import asyncio
import json

import diskcache
import numpy as np
import pandas as pd
import pytest

from tastelayer.schema import TASTE_DIMS
from tastelayer.tagging import rubric
from tastelayer.tagging.client import BudgetHardStop, BudgetPause, CallResult, SpendTracker
from tastelayer.tagging.consensus import compute_consensus, disagreement_table
from tastelayer.tagging.tag import Progress, Tagger, chunk, parse_items


def _item(uid, v=3, **kw):
    return {"uid": uid, "taste": {d: v for d in TASTE_DIMS}, "confidence": 0.8, "evidence": "x", **kw}


def test_strict_schema_is_inlined_and_strict():
    s = rubric.strict_schema(rubric.DishTagBatch)
    txt = json.dumps(s)
    assert "$ref" not in txt and "$defs" not in txt

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object":
                assert n["additionalProperties"] is False
                assert set(n["required"]) == set(n["properties"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(s)


def test_parse_clamps_and_matches_keys():
    content = "```json\n" + json.dumps({"items": [
        {"uid": "A1", "taste": {**{d: 4 for d in TASTE_DIMS}, "spice": 14}, "confidence": 1.7, "evidence": "e"},
        {"uid": "zz", "taste": {d: 1 for d in TASTE_DIMS}, "confidence": 0.5, "evidence": "e"},
    ]}) + "\n```"
    got = parse_items(content, "dish", ["a1", "b2"])
    assert list(got) == ["a1"]  # case-insensitive, unknown uid dropped
    assert got["a1"]["taste"]["spice"] == 10 and got["a1"]["confidence"] == 1.0


def test_parse_ingredients_by_name():
    content = json.dumps({"items": [{"name": "Habanero ", "taste": {d: 9 for d in TASTE_DIMS}, "confidence": 0.9}]})
    assert list(parse_items(content, "ingredient", ["habanero"])) == ["habanero"]


def test_parse_rejects_garbage():
    with pytest.raises(Exception):
        parse_items("not json at all", "dish", ["a"])


class FakeClient:
    """Fails (invalid JSON) for batches bigger than `max_ok`; otherwise returns valid items."""

    def __init__(self, max_ok=4, drop=None):
        self.max_ok, self.drop, self.calls = max_ok, set(drop or []), 0

    async def complete_json(self, model, system, user, name, schema, temperature, seed, max_tokens=3000, meta=None):
        self.calls += 1
        payload = json.loads(user.split("Dishes:\n", 1)[1].split("\n\nNOTE")[0])
        if len(payload) > self.max_ok:
            return CallResult("{broken", 10, 10, 0.0, 0.1)
        items = [_item(p["uid"]) for p in payload if p["uid"] not in self.drop]
        return CallResult(json.dumps({"items": items}), 10, 10, 0.0, 0.1)


def _rows(n):
    return [{"uid": f"u{i:02d}", "cuisine": "indian", "name": f"d{i}", "description": "", "ingredients": []}
            for i in range(n)]


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    import tastelayer.tagging.tag as tag_mod

    monkeypatch.setattr(tag_mod, "FAILURES_LOG", tmp_path / "failures.jsonl")


def test_split_on_failure_recovers_all(tmp_path):
    fake = FakeClient(max_ok=4)
    t = Tagger(fake, diskcache.Cache(str(tmp_path)), Progress())
    got = asyncio.run(t.tag("m", 0.2, "dish", _rows(10)))
    assert set(got) == {f"u{i:02d}" for i in range(10)}
    calls = fake.calls
    # second run is a pure cache hit
    got2 = asyncio.run(t.tag("m", 0.2, "dish", _rows(10)))
    assert got2 == got and fake.calls == calls


def test_missing_uid_retried_then_given_up(tmp_path):
    fake = FakeClient(max_ok=10, drop={"u03"})
    prog = Progress()
    t = Tagger(fake, diskcache.Cache(str(tmp_path)), prog)
    got = asyncio.run(t.tag("m", 0.2, "dish", _rows(10)))
    assert "u03" not in got and len(got) == 9
    assert prog.failed_keys["m"] == ["u03"]
    # incomplete batches are not cached at the top level, so a rerun retries
    before = fake.calls
    asyncio.run(t.tag("m", 0.2, "dish", _rows(10)))
    assert fake.calls > before


def test_budget_guard(tmp_path, monkeypatch):
    from tastelayer import config

    s = SpendTracker(tmp_path / "spend.json")
    monkeypatch.setattr(config, "ASK_USD", 0.001)
    monkeypatch.setattr(config, "MAX_USD", 0.002)
    monkeypatch.delenv("ALLOW_OVER_ASK", raising=False)
    s.check()
    s.add("openai/gpt-oss-120b", 10_000, 0)  # $0.0015
    with pytest.raises(BudgetPause):
        s.check()
    monkeypatch.setenv("ALLOW_OVER_ASK", "1")
    s.check()
    s.add("openai/gpt-oss-120b", 10_000, 0)
    with pytest.raises(BudgetHardStop):
        s.check()
    assert json.loads((tmp_path / "spend.json").read_text())["total_usd"] > 0


def _labels(rows):
    out = []
    for uid, model, temp, pas, vals in rows:
        r = {"uid": uid, "model": model, "temperature": temp, "pass": pas}
        r.update(dict(zip(TASTE_DIMS, vals)))
        out.append(r)
    return pd.DataFrame(out)


def test_consensus_median_std_conf_and_review():
    base = [0] * 8
    lab = _labels([
        ("a", "m1", 0.2, "main", [2] + base[1:]),
        ("a", "m2", 0.2, "main", [4] + base[1:]),
        ("a", "m3", 0.2, "main", [9] + base[1:]),
        ("b", "m1", 0.2, "main", [5] * 8),
        ("b", "m2", 0.2, "main", [5] * 8),
        ("a", "q-review", 0.2, "review", [8] + base[1:]),
    ])
    c = compute_consensus(lab).set_index("uid")
    # a.spice pool = [2,4,9] + review 8 twice -> median of [2,4,8,8,9] = 8
    assert c.loc["a", "spice"] == 8
    assert c.loc["a", "spice_std"] == pytest.approx(np.std([2, 4, 9]), abs=1e-3)
    assert c.loc["a", "spice_conf"] == pytest.approx(1 - min(np.std([2, 4, 9]) / 3, 1), abs=1e-3)
    assert bool(c.loc["a", "reviewed"]) and not bool(c.loc["b", "reviewed"])
    # std 0 but only 2 models -> coverage factor 0.75
    assert c.loc["b", "spice_conf"] == 0.75 and c.loc["b", "n_labels"] == 2 and c.loc["b", "n_models"] == 2
    dis = disagreement_table(c.reset_index(), {"a": {"name": "A", "cuisine": "indian"}})
    assert list(dis["uid"]) == ["a"] and dis.iloc[0]["dims"] == "spice"


def test_chunk_deterministic():
    assert chunk(list(range(25)), 10) == [list(range(10)), list(range(10, 20)), list(range(20, 25))]
