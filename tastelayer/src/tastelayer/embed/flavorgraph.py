"""Optional FlavorGraph (Park et al. 2021) pretrained ingredient vectors, mapped onto our vocab.

Source files (fetched from github.com/lamypark/FlavorGraph into data/raw/flavorgraph/):
  nodes_191120.csv              node_id,name,id,node_type,is_hub
  kitchenette_embeddings.pkl    dict node_id -> np.ndarray  (loaded with a numpy-only restricted unpickler)
Output: artifacts/flavorgraph_emb.npy + artifacts/flavorgraph_emb_vocab.json (rows aligned to our vocab subset).
"""
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

FG_DIR = config.DATA_RAW / "flavorgraph"
REPO_RAW = "https://raw.githubusercontent.com/lamypark/FlavorGraph/master"
FILES = {"nodes_191120.csv": f"{REPO_RAW}/input/nodes_191120.csv",
         "kitchenette_embeddings.pkl": f"{REPO_RAW}/output/kitchenette_embeddings.pkl"}
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar")}


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def download() -> bool:
    import httpx

    FG_DIR.mkdir(parents=True, exist_ok=True)
    for fn, url in FILES.items():
        p = FG_DIR / fn
        if p.exists() and p.stat().st_size > 1000:
            continue
        try:
            r = httpx.get(url, timeout=120, follow_redirects=True)
            r.raise_for_status()
            p.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            print(f"[flavorgraph] download failed for {fn}: {e}")
            return False
    return True


def load_raw() -> dict[str, np.ndarray] | None:
    """Return {ingredient name (spaces): vector} or None if files are missing."""
    nodes_p, emb_p = FG_DIR / "nodes_191120.csv", FG_DIR / "kitchenette_embeddings.pkl"
    if not emb_p.exists() or emb_p.stat().st_size < 1000:
        return None
    emb = _NumpyOnly(io.BytesIO(emb_p.read_bytes())).load()
    if not isinstance(emb, dict):
        return None
    # The published pickle is keyed by ingredient name (e.g. "beef_suet"); older dumps use node_id.
    id2name = {}
    if nodes_p.exists():
        nodes = pd.read_csv(nodes_p)
        nodes = nodes[nodes.node_type == "ingredient"]
        id2name = {str(i): str(n) for i, n in zip(nodes.node_id, nodes.name)}
    out = {}
    for k, v in emb.items():
        name = id2name.get(str(k), str(k))
        out[name.replace("_", " ").lower().strip()] = np.asarray(v, dtype=np.float32)
    return out


def build(vocab: list[str], min_score: int = 92) -> tuple[np.ndarray, list[str]] | None:
    """Map our canonical vocab onto FlavorGraph names (exact, then rapidfuzz). Saves artifacts."""
    from rapidfuzz import fuzz, process

    raw = load_raw()
    if raw is None and download():
        raw = load_raw()
    if not raw:
        print("[flavorgraph] not available, skipping")
        return None
    names = list(raw)
    rows, kept = [], []
    for ing in vocab:
        alias = next((a for a in [ing, f"{ing} cheese", f"fresh {ing}", f"{ing}s", ing.rstrip("s"),
                                  f"{ing} paste", f"{ing} leaf", f"{ing} powder"] if a in raw), None)
        if alias:
            rows.append(raw[alias]); kept.append(ing); continue
        m = process.extractOne(ing, names, scorer=fuzz.ratio, score_cutoff=min_score)
        if m:
            rows.append(raw[m[0]]); kept.append(ing)
    if not rows:
        return None
    X = np.vstack(rows)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    config.ARTIFACTS.mkdir(exist_ok=True)
    np.save(config.ARTIFACTS / "flavorgraph_emb.npy", X)
    (config.ARTIFACTS / "flavorgraph_emb_vocab.json").write_text(json.dumps(kept))
    print(f"[flavorgraph] mapped {len(kept)}/{len(vocab)} vocab ingredients, dim={X.shape[1]}")
    return X, kept
