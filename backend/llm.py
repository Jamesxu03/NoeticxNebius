"""
Optional LLM rewrite of explanation text via Nebius Token Factory (OpenAI-compatible API).

    NEBIUS_API_KEY=...                       # unset -> every call returns the template unchanged
    NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast
    LLM_CACHE=backend/.cache/llm.json        # answers are cached, so a repeated demo makes no calls

The model only sees facts the recommender already computed. It never sees or decides
allergens or dietary restrictions; warnings are returned separately and shown verbatim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
MODEL = os.environ.get("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-fast")
CACHE_PATH = Path(os.environ.get("LLM_CACHE", Path(__file__).resolve().parent / ".cache/llm.json"))
TIMEOUT_S = 4.0
MAX_CHARS = 300

SYSTEM = (
    "You write one short, warm sentence (max 30 words) telling a diner why a restaurant and dish "
    "were picked for them. Use only the facts given. Do not mention allergens, diets or safety. "
    "Do not invent ingredients, prices or claims. Reply with the sentence only."
)

_lock = threading.Lock()
_cache: dict[str, str] | None = None


def enabled() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY"))


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=1, ensure_ascii=False))


def chat(system: str, user: str, max_tokens: int = 80) -> str | None:
    """One cached chat completion; None if disabled or the call fails."""
    if not enabled():
        return None
    key = hashlib.sha256(json.dumps([MODEL, system, user]).encode()).hexdigest()
    with _lock:
        cache = _load_cache()
        if key in cache:
            return cache[key]
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}"},
            json={
                "model": MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.4,
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # network, auth, quota, bad JSON: the template is always good enough
        log.warning("LLM call failed, using template: %s", exc)
        return None
    with _lock:
        _load_cache()[key] = text
        _save_cache()
    return text


def rewrite_explanation(facts: dict, template: str) -> str:
    text = chat(SYSTEM, f"Facts: {json.dumps(facts, ensure_ascii=False)}\nDraft: {template}")
    if not text or len(text) > MAX_CHARS:
        return template
    return text.strip('"')
