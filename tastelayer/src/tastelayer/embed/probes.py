"""Linear taste probes: per dim, ridge from ingredient embedding -> LLM ingredient taste score.

Direction = normalised ridge coefficient. Sanity check: the 3 benchmark anchors of each dim must
project in order low < mid < high. Fallback (PROMPTS.md): if our embedding fails the anchor check
on more dims than FlavorGraph, FlavorGraph becomes the primary embedding.

Output: artifacts/taste_directions.json, reports/probes.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from .. import config
from ..schema import TASTE_DIMS
from .cooc import load_embedding, neighbours

# Canonical-name candidates for each benchmark anchor (first one found in vocab is used).
ANCHOR_CANDIDATES = {
    "spice": [["black pepper", "pepper"], ["jalapeno", "jalapeno pepper", "green chilli"], ["habanero", "scotch bonnet", "habanero pepper"]],
    "sweet": [["roasted red pepper", "red bell pepper", "red pepper", "bell pepper"], ["caramelised onion", "caramelized onion", "onion"], ["maple syrup", "honey", "sugar"]],
    "sour": [["natural yoghurt", "yoghurt", "yogurt", "plain yogurt"], ["lemon juice", "lemon"], ["tamarind", "tamarind paste"]],
    "bitter": [["rocket", "arugula"], ["dark chocolate", "bittersweet chocolate"], ["radicchio", "endive"]],
    "umami": [["tomato"], ["parmesan", "parmesan cheese", "parmigiano reggiano"], ["anchovy", "miso", "anchovy fillet"]],
    "richness": [["olive oil", "extra virgin olive oil"], ["cream", "heavy cream", "double cream"], ["ghee", "butter"]],
    "crunch": [["bread", "white bread"], ["nuts", "almond", "walnut", "cashew"], ["papadum", "poppadom", "papad"]],
    "creaminess": [["broth", "stock", "chicken stock", "vegetable stock"], ["hummus", "chickpea"], ["mascarpone", "mascarpone cheese"]],
}
SANITY_WORDS = ["paneer", "basil", "tamarind", "guanciale"]


def ingredient_targets() -> pd.DataFrame | None:
    p = config.DATA_PROCESSED / "ingredient_taste_llm.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.groupby("ingredient")[TASTE_DIMS].mean()


def _pick(cands, vocab_set):
    return next((c for c in cands if c in vocab_set), None)


def fit_probes(E: np.ndarray, vocab: list[str], targets: pd.DataFrame, seed: int = config.SEED) -> dict:
    idx = {v: i for i, v in enumerate(vocab)}
    common = [i for i in targets.index if i in idx]
    if len(common) < 20:
        raise RuntimeError(f"only {len(common)} ingredients overlap between embedding and LLM scores")
    X = E[[idx[i] for i in common]]
    Y = targets.loc[common, TASTE_DIMS].to_numpy(float)
    alphas = np.logspace(-2, 3, 12)
    kf = KFold(5, shuffle=True, random_state=seed)
    out = {"n_ingredients": len(common), "dim": int(E.shape[1]), "directions": {}, "coef": {}, "intercept": {},
           "cv_spearman": {}, "anchor_check": {}}
    vs = set(vocab)
    for j, d in enumerate(TASTE_DIMS):
        y = Y[:, j]
        pred = np.zeros_like(y)
        for tr, te in kf.split(X):
            pred[te] = RidgeCV(alphas=alphas).fit(X[tr], y[tr]).predict(X[te])
        rho = spearmanr(pred, y).correlation
        out["cv_spearman"][d] = None if np.isnan(rho) else round(float(rho), 3)
        m = RidgeCV(alphas=alphas).fit(X, y)
        coef = m.coef_.astype(float)
        out["coef"][d] = coef.tolist()
        out["intercept"][d] = float(m.intercept_)
        out["directions"][d] = (coef / (np.linalg.norm(coef) + 1e-12)).tolist()
        anchors = [_pick(c, vs) for c in ANCHOR_CANDIDATES[d]]
        proj = [None if a is None else round(float(E[idx[a]] @ coef + m.intercept_), 2) for a in anchors]
        ok = None if None in proj else bool(proj[0] < proj[1] < proj[2])
        out["anchor_check"][d] = {"anchors": anchors, "pred_score": proj, "ordered": ok}
    return out


def run() -> dict:
    targets = ingredient_targets()
    if targets is None:
        raise RuntimeError("data/processed/ingredient_taste_llm.parquet missing — run ingredient tagging first")
    results, lines = {}, ["# Taste probes", ""]
    for name in ["ours", "flavorgraph"]:
        emb = load_embedding(name)
        if emb is None:
            continue
        E, vocab = emb
        try:
            r = fit_probes(E, vocab, targets)
        except RuntimeError as e:
            print(f"[probes] {name}: {e}")
            continue
        results[name] = r
        lines += [f"## {name} (dim {r['dim']}, {r['n_ingredients']} ingredients)", "",
                  "| dim | CV Spearman | low / mid / high anchor | predicted score | ordered |", "|---|---|---|---|---|"]
        for d in TASTE_DIMS:
            a = r["anchor_check"][d]
            lines.append(f"| {d} | {r['cv_spearman'][d]} | {' / '.join(str(x) for x in a['anchors'])} | "
                         f"{a['pred_score']} | {a['ordered']} |")
        lines += ["", "Nearest neighbours:", ""]
        for w in SANITY_WORDS:
            lines.append(f"- **{w}**: {neighbours(E, vocab, w, 6) or '(not in vocab)'}")
        lines.append("")
    if not results:
        raise RuntimeError("no embeddings available for probes")
    n_fail = {k: sum(v["anchor_check"][d]["ordered"] is False for d in TASTE_DIMS) for k, v in results.items()}
    primary = "ours"
    if "ours" not in results or ("flavorgraph" in results and n_fail["ours"] > n_fail["flavorgraph"]):
        primary = "flavorgraph" if "flavorgraph" in results else "ours"
    lines.insert(2, f"Primary embedding: **{primary}** (anchor-order failures: {n_fail})\n")
    payload = {"primary": primary, "embeddings": results}
    (config.ARTIFACTS / "taste_directions.json").write_text(json.dumps(payload))
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "probes.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return payload


def load_directions() -> dict | None:
    p = config.ARTIFACTS / "taste_directions.json"
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == "__main__":
    run()
