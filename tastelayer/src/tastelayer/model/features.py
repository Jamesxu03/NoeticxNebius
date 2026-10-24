"""Feature blocks for TasteNet.

Blocks (each a dense float32 matrix, one row per dish):
  bge       local sentence-transformers BAAI/bge-base-en-v1.5 (768-d, L2-normalised)
  qwen      Qwen/Qwen3-Embedding-8B via Nebius /v1/embeddings, truncated to QWEN_DIM (MRL) + renormalised
  ing_ours  IDF-weighted mean of mini-Epicure ingredient vectors + 8 probe projections (mean, max)
  ing_fg    same with FlavorGraph vectors
  lex       lexicon baseline 8 scores + keyword flags + cuisine/menu-style indicators

Text for embeddings: "{cuisine} | {name} | {description} | ingredients: a, b, c".
Embeddings are cached per text (sha256) in artifacts/emb_cache/<model>_{keys.json,vecs.npy}.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..lexicon import lookup_feature_matrix

BGE_MODEL = "BAAI/bge-base-en-v1.5"
QWEN_MODEL = config.EMBEDDING_MODEL
QWEN_DIM = 1024  # native 4096; Nebius accepts dimensions= (Matryoshka truncation server-side)
EMB_CACHE = config.ARTIFACTS / "emb_cache"
ALL_BLOCKS = ["bge", "qwen", "ing_ours", "ing_fg", "lex"]


def _listify(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    return [str(i) for i in list(x)]


def dish_text(row: dict) -> str:
    ings = _listify(row.get("ingredients_canonical")) or _listify(row.get("ingredients"))
    desc = str(row.get("description") or "")[:400]
    return f"{row.get('cuisine', '')} | {row.get('name', '')} | {desc} | ingredients: {', '.join(ings[:25])}"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


class EmbeddingStore:
    """Append-only per-text vector cache on disk."""

    def __init__(self, name: str):
        self.name = name
        self.kp, self.vp = EMB_CACHE / f"{name}_keys.json", EMB_CACHE / f"{name}_vecs.npy"
        self.keys: list[str] = json.loads(self.kp.read_text()) if self.kp.exists() else []
        self.vecs = np.load(self.vp) if self.vp.exists() else None
        n_vecs = 0 if self.vecs is None else len(self.vecs)
        if len(self.keys) != n_vecs:
            # e.g. a fresh clone: the keys file was committed but the >50MB vectors file was not.
            # Never trust an index that doesn't match its vectors; start the cache over instead.
            print(f"[emb_cache] {name}: {len(self.keys)} keys but {n_vecs} vectors; ignoring the stale cache")
            self.keys, self.vecs = [], None
        self.index = {k: i for i, k in enumerate(self.keys)}

    def get(self, texts: list[str], compute) -> np.ndarray:
        hs = [_h(t) for t in texts]
        missing = list(dict.fromkeys(h for h in hs if h not in self.index))
        if missing:
            by_h = {}
            for t, h in zip(texts, hs):
                by_h.setdefault(h, t)
            new = np.asarray(compute([by_h[h] for h in missing]), dtype=np.float32)
            self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])
            for h in missing:
                self.index[h] = len(self.keys)
                self.keys.append(h)
            EMB_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(self.vp, self.vecs)
            self.kp.write_text(json.dumps(self.keys))
        return self.vecs[[self.index[h] for h in hs]]


_BGE = None


def load_bge(allow_download: bool = False):
    """Load bge-base from the local Hugging Face cache only, so serving makes no network call
    (without this, every load sends HEAD requests to huggingface.co to check for updates).
    Fetch the weights once with `make fetch-models` (allow_download=True)."""
    from sentence_transformers import SentenceTransformer

    if allow_download:
        return SentenceTransformer(BGE_MODEL)
    try:
        return SentenceTransformer(BGE_MODEL, local_files_only=True)
    except OSError as e:
        raise RuntimeError(f"{BGE_MODEL} is not in the local cache; run `make fetch-models` once (needs network)") from e


def _bge_compute(texts: list[str]) -> np.ndarray:
    global _BGE
    if _BGE is None:
        _BGE = load_bge()
    return _BGE.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 2000)


def embed_bge(texts: list[str]) -> np.ndarray:
    return EmbeddingStore("bge").get(texts, _bge_compute)


async def _qwen_async(texts: list[str], batch: int = 96, concurrency: int = 8) -> np.ndarray:
    from openai import AsyncOpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    from .. import spend

    client = AsyncOpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.nebius_api_key(), timeout=120)
    sem = asyncio.Semaphore(concurrency)
    out: list[np.ndarray | None] = [None] * math.ceil(len(texts) / batch)

    @retry(stop=stop_after_attempt(8), wait=wait_random_exponential(min=2, max=60), reraise=True)
    async def one(i: int):
        chunk = texts[i * batch:(i + 1) * batch]
        async with sem:
            spend.check_budget()
            t0 = time.time()
            r = await client.embeddings.create(model=QWEN_MODEL, input=chunk, dimensions=QWEN_DIM)
            spend.add_spend(QWEN_MODEL, r.usage.prompt_tokens if r.usage else 0, 0, kind="embedding",
                            latency_s=time.time() - t0, n_items=len(chunk))
        E = np.array([d.embedding for d in sorted(r.data, key=lambda d: d.index)], dtype=np.float32)
        out[i] = E

    await asyncio.gather(*(one(i) for i in range(len(out))))
    return np.vstack(out)


def _qwen_compute(texts: list[str]) -> np.ndarray:
    E = asyncio.run(_qwen_async(texts))
    E = E[:, :QWEN_DIM]
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def embed_qwen(texts: list[str]) -> np.ndarray:
    return EmbeddingStore(f"qwen{QWEN_DIM}").get(texts, _qwen_compute)


@dataclass
class FeatureBuilder:
    """Holds the fitted state needed to featurise new dishes identically at predict time."""
    idf: dict[str, float] = field(default_factory=dict)
    emb_names: list[str] = field(default_factory=list)  # which ingredient embeddings are loaded

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        lists = [_listify(x) for x in df["ingredients_canonical"]] if "ingredients_canonical" in df else []
        n = max(1, len(lists))
        dfc = Counter(i for l in lists for i in set(l))
        self.idf = {i: math.log((1 + n) / (1 + c)) + 1 for i, c in dfc.items()}
        return self

    # --- ingredient-embedding block ---
    def _ing_block(self, rows: list[dict], name: str) -> np.ndarray | None:
        from ..embed.cooc import load_embedding
        from ..embed.probes import load_directions

        emb = load_embedding(name)
        if emb is None:
            return None
        E, vocab = emb
        idx = {v: i for i, v in enumerate(vocab)}
        dirs = (load_directions() or {}).get("embeddings", {}).get(name)
        from ..schema import TASTE_DIMS

        W = np.array([dirs["coef"][d] for d in TASTE_DIMS]).T if dirs else None  # (dim, 8)
        b = np.array([dirs["intercept"][d] for d in TASTE_DIMS]) if dirs else None
        default_idf = max(self.idf.values()) if self.idf else 1.0
        n_proj = 16 if W is not None else 0
        out = np.zeros((len(rows), E.shape[1] + n_proj + 1), dtype=np.float32)
        for r, row in enumerate(rows):
            ings = _listify(row.get("ingredients_canonical"))
            ids = [idx[i] for i in ings if i in idx]
            if not ids:
                continue
            w = np.array([self.idf.get(vocab[i], default_idf) for i in ids])
            V = E[ids]
            out[r, :E.shape[1]] = (w[:, None] * V).sum(0) / w.sum()
            if W is not None:
                P = V @ W + b  # per-ingredient predicted taste (n_ing, 8)
                out[r, E.shape[1]:E.shape[1] + 8] = (w[:, None] * P).sum(0) / w.sum()
                out[r, E.shape[1] + 8:E.shape[1] + 16] = P.max(0)
            out[r, -1] = len(ids) / max(1, len(ings))  # vocab coverage
        return out

    def coverage(self, rows: list[dict]) -> np.ndarray:
        from ..embed.cooc import load_embedding

        emb = load_embedding("ours")
        vocab = set(emb[1]) if emb else set()
        cov = []
        for row in rows:
            ings = _listify(row.get("ingredients_canonical"))
            cov.append(sum(i in vocab for i in ings) / len(ings) if ings else 0.0)
        return np.array(cov)

    def blocks(self, df: pd.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
        rows = df.to_dict("records")
        texts = [dish_text(r) for r in rows]
        out: dict[str, np.ndarray] = {}
        for n in names:
            if n == "bge":
                out[n] = embed_bge(texts)
            elif n == "qwen":
                out[n] = embed_qwen(texts)
            elif n in ("ing_ours", "ing_fg"):
                m = self._ing_block(rows, "ours" if n == "ing_ours" else "flavorgraph")
                if m is not None:
                    out[n] = m
            elif n == "lex":
                lx = lookup_feature_matrix(df)
                cui = np.array([[r.get("cuisine") == "indian", r.get("cuisine") == "italian",
                                 bool(r.get("menu_style", False))] for r in rows], dtype=np.float32)
                out[n] = np.hstack([lx, cui]).astype(np.float32)
            else:
                raise ValueError(f"unknown block {n}")
        return out

    def transform(self, df: pd.DataFrame, names: list[str]) -> np.ndarray:
        b = self.blocks(df, names)
        missing = [n for n in names if n not in b]
        if missing:
            raise RuntimeError(f"feature blocks unavailable: {missing}")
        return np.hstack([b[n] for n in names]).astype(np.float32)


def available_ing_blocks() -> list[str]:
    from ..embed.cooc import load_embedding

    return [b for b, n in [("ing_ours", "ours"), ("ing_fg", "flavorgraph")] if load_embedding(n) is not None]
