"""Mini-Epicure ingredient embeddings: co-occurrence stats + skip-gram over shuffled ingredient sets.

Epicure (Radzikowski & Chen) trains skip-gram/Metapath2Vec on recipe ingredient sets; we do the
cheap version: Word2Vec sg=1 over 5 random permutations of each recipe's canonical ingredients,
optionally ("core" variant) plus random walks over the positive-NPMI graph.

workers=1 by default so the embedding is bit-for-bit reproducible with a fixed seed; more workers
are ~Nx faster but gensim then gives slightly different vectors run to run.

Outputs: artifacts/ingredient_emb.npy (L2-normalised rows), artifacts/ingredient_emb_vocab.json
(row order), artifacts/ingredient_npmi.npz (sparse positive NPMI, same order).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from .. import config

MIN_COUNT = 20


def load_ingredient_lists() -> list[list[str]]:
    lists: list[list[str]] = []
    for fn in ["corpus.parquet", "cooc_extra.parquet"]:
        p = config.DATA_PROCESSED / fn
        if p.exists():
            df = pd.read_parquet(p, columns=["ingredients_canonical"])
            lists += [list(x) for x in df.ingredients_canonical if x is not None and len(x) > 0]
    return lists


def build_vocab(lists: list[list[str]], min_count: int = MIN_COUNT) -> list[str]:
    df = Counter(i for l in lists for i in set(l))
    return sorted([i for i, c in df.items() if c >= min_count], key=lambda i: (-df[i], i))


def npmi_matrix(lists: list[list[str]], vocab: list[str]) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Document-level co-occurrence -> (PPMI, positive NPMI), both sparse |V|x|V|."""
    idx = {v: i for i, v in enumerate(vocab)}
    rows, cols = [], []
    for r, l in enumerate(lists):
        ids = sorted({idx[i] for i in l if i in idx})
        rows += [r] * len(ids)
        cols += ids
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lists), len(vocab)))
    n = X.shape[0]
    C = (X.T @ X).tocoo()
    df = np.asarray(X.sum(0)).ravel()
    mask = C.row != C.col
    r, c, v = C.row[mask], C.col[mask], C.data[mask]
    pij = v / n
    pmi = np.log(pij / ((df[r] / n) * (df[c] / n)))
    npmi = pmi / (-np.log(pij))
    shape = (len(vocab), len(vocab))
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep], (r[keep], c[keep])), shape=shape)
    pnpmi = sparse.csr_matrix((npmi[keep], (r[keep], c[keep])), shape=shape)
    return ppmi, pnpmi


def random_walks(G: sparse.csr_matrix, vocab: list[str], walks_per_node: int = 20, walk_length: int = 50,
                 seed: int = config.SEED) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    G = G.tocsr()
    walks = []
    for _ in range(walks_per_node):
        for start in range(G.shape[0]):
            w = [start]
            for _ in range(walk_length - 1):
                cur = w[-1]
                s, e = G.indptr[cur], G.indptr[cur + 1]
                if s == e:
                    break
                p = G.data[s:e]
                w.append(int(G.indices[s:e][rng.choice(e - s, p=p / p.sum())]))
            if len(w) > 1:
                walks.append([vocab[i] for i in w])
    return walks


def train(lists: list[list[str]] | None = None, dim: int = 128, core: bool = False, workers: int = 1,
          epochs: int = 20, permutations: int = 5, min_count: int = MIN_COUNT, seed: int = config.SEED,
          save: bool = True):
    from gensim.models import Word2Vec

    lists = load_ingredient_lists() if lists is None else lists
    vocab = build_vocab(lists, min_count)
    if len(vocab) < 5:
        raise RuntimeError(f"vocab too small ({len(vocab)}) — need corpus.parquet with ingredient lists")
    vs = set(vocab)
    rng = np.random.default_rng(seed)
    sents = []
    for l in lists:
        l = [i for i in dict.fromkeys(l) if i in vs]
        if len(l) < 2:
            continue
        for _ in range(permutations):
            sents.append(list(rng.permutation(l)))
    ppmi, pnpmi = npmi_matrix(lists, vocab)
    if core:
        sents += random_walks(pnpmi, vocab, seed=seed)
    m = Word2Vec(sentences=sents, vector_size=dim, window=10, sg=1, negative=5, min_count=1, epochs=epochs,
                 seed=seed, workers=workers)
    E = np.vstack([m.wv[v] for v in vocab]).astype(np.float32)
    E -= E.mean(0, keepdims=True)  # remove the common direction skip-gram vectors share (cosines otherwise ~0.9+)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    if save:
        config.ARTIFACTS.mkdir(exist_ok=True)
        np.save(config.ARTIFACTS / "ingredient_emb.npy", E)
        (config.ARTIFACTS / "ingredient_emb_vocab.json").write_text(json.dumps(vocab))
        sparse.save_npz(config.ARTIFACTS / "ingredient_npmi.npz", pnpmi)
        print(f"[cooc] {len(lists)} recipes, {len(sents)} sentences, vocab {len(vocab)}, dim {dim}, core={core}")
    return E, vocab, pnpmi


def load_embedding(name: str = "ours") -> tuple[np.ndarray, list[str]] | None:
    f = {"ours": "ingredient_emb", "flavorgraph": "flavorgraph_emb"}[name]
    p, pv = config.ARTIFACTS / f"{f}.npy", config.ARTIFACTS / f"{f}_vocab.json"
    if not (p.exists() and pv.exists()):
        return None
    return np.load(p), json.loads(pv.read_text())


def neighbours(E: np.ndarray, vocab: list[str], word: str, k: int = 8) -> list[tuple[str, float]]:
    if word not in vocab:
        return []
    i = vocab.index(word)
    sims = E @ E[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(vocab[j], round(float(sims[j]), 3)) for j in order]
