"""TasteNet: dish text + ingredient features -> 8 taste dims, distilled from the Nebius 3-model consensus.

- targets: labels_consensus.parquet per-dim medians; sample_weight = mean per-dim confidence
- never trains on gold uids
- holdout: 10% of name-groups within each source (near-duplicate names never straddle the split)
- feature-set comparison: 5-fold grouped CV with RidgeCV on the train split -> reports/feature_cv.md
- final candidates: RidgeCV and MLPRegressor(512, 256) on the best feature set; pick by held-out mean Spearman
- artifact: artifacts/tastenet.joblib
"""
from __future__ import annotations

import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..schema import TASTE_DIMS
from .features import FeatureBuilder, available_ing_blocks

VERSION = "tastenet-v1"
ARTIFACT = config.ARTIFACTS / "tastenet.joblib"
ALPHAS = np.logspace(-1, 4, 16)
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------- data
def load_dishes() -> pd.DataFrame:
    """Corpus rows + menu dishes (menu uid = dish id), one row per uid."""
    frames = []
    p = config.DATA_PROCESSED / "corpus.parquet"
    if p.exists():
        frames.append(pd.read_parquet(p))
    mp = config.DATA_PROCESSED / "menu_dishes.json"
    if mp.exists():
        menu = json.loads(mp.read_text())
        frames.append(pd.DataFrame([{
            "uid": d["id"], "source": "menu", "cuisine": d.get("cuisine"), "name": d.get("name", ""),
            "description": d.get("description", ""),
            "ingredients_canonical": d.get("ingredients_canonical") or d.get("ingredients") or [],
            "menu_style": True, "extra": "{}"} for d in menu]))
    if not frames:
        raise FileNotFoundError("no corpus.parquet or menu_dishes.json in data/processed")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("uid", keep="last").reset_index(drop=True)
    for c, default in [("source", "unknown"), ("menu_style", False), ("description", ""), ("extra", "{}")]:
        if c not in df:
            df[c] = default
    df["description"] = df["description"].fillna("")
    return df


def gold_uids() -> set[str]:
    """Every dish that is (or may become) gold: labelled gold plus all labelling candidates."""
    uids: set[str] = set()
    for name in ("gold.json", "gold_candidates.json"):
        p = config.GOLD / name
        if p.exists():
            uids |= {g["uid"] for g in json.loads(p.read_text())}
    return uids


MIN_LABELS = int(os.getenv("TASTENET_MIN_LABELS", "4"))  # of 6 (3 models x 2 temps); 1-model rows have fake std=0


def load_training_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cons = pd.read_parquet(config.DATA_PROCESSED / "labels_consensus.parquet")
    df = load_dishes().merge(cons, on="uid", how="inner", suffixes=("", "_lbl"))
    df = df[~df.uid.isin(gold_uids())]
    if "n_labels" in df:
        df = df[df.n_labels >= MIN_LABELS]
    df = df.reset_index(drop=True)
    Y = df[TASTE_DIMS].to_numpy(float)
    conf_cols = [f"{d}_conf" for d in TASTE_DIMS]
    w = df[conf_cols].mean(axis=1).to_numpy(float) if all(c in df for c in conf_cols) else np.ones(len(df))
    return df, Y, np.clip(w, 0.05, 1.0)


def name_groups(df: pd.DataFrame) -> np.ndarray:
    norm = df["name"].fillna("").str.lower().map(lambda s: re.sub(r"[^a-z ]", "", s).strip())
    return (df["cuisine"].fillna("") + "|" + norm).to_numpy()


def holdout_split(df: pd.DataFrame, frac: float = 0.10, seed: int = config.SEED) -> np.ndarray:
    """Boolean mask of held-out rows: frac of name-groups within each source."""
    rng = np.random.default_rng(seed)
    groups = name_groups(df)
    test = np.zeros(len(df), bool)
    for src in sorted(df["source"].astype(str).unique()):
        m = (df["source"].astype(str) == src).to_numpy()
        g = np.unique(groups[m])
        if len(g) < 10:
            continue
        pick = set(rng.choice(g, size=max(1, int(round(frac * len(g)))), replace=False))
        test |= m & np.isin(groups, list(pick))
    return test


# ---------------------------------------------------------------- metrics
def spearman_per_dim(Y: np.ndarray, P: np.ndarray) -> list[float]:
    out = []
    for j in range(Y.shape[1]):
        r = spearmanr(Y[:, j], P[:, j]).correlation
        out.append(0.0 if np.isnan(r) else float(r))
    return out


def mean_spearman(Y, P) -> float:
    return float(np.mean(spearman_per_dim(Y, P)))


# ---------------------------------------------------------------- models
def make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def make_mlp(seed: int = config.SEED):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(512, 256), early_stopping=True,
                                                        validation_fraction=0.1, max_iter=200, alpha=1e-3,
                                                        learning_rate_init=1e-3, batch_size=256, random_state=seed))


def fit(model, X, Y, w):
    """Fit on 0-1 scaled targets; pass sample_weight where the estimator supports it."""
    last = model.steps[-1][0]
    try:
        model.fit(X, Y / 10.0, **{f"{last}__sample_weight": w})
    except TypeError:
        model.fit(X, Y / 10.0)
    return model


def predict(model, X) -> np.ndarray:
    return np.clip(model.predict(X) * 10.0, 0, 10)


def candidate_feature_sets(text_blocks: list[str]) -> list[list[str]]:
    ing = available_ing_blocks()
    sets = [["lex"], *[[t] for t in text_blocks]]
    for t in text_blocks:
        sets.append([t, "lex"] + ing)
    if len(text_blocks) > 1:
        sets.append(text_blocks + ["lex"] + ing)
    if ing:
        sets.append(["lex"] + ing)
    out, seen = [], set()
    for s in sets:
        k = tuple(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def feature_cv(blocks: dict[str, np.ndarray], Y, w, groups, sets, n_splits: int = 5) -> list[dict]:
    rows = []
    gkf = GroupKFold(n_splits=n_splits)
    for s in sets:
        if any(b not in blocks for b in s):
            continue
        X = np.hstack([blocks[b] for b in s])
        P = np.zeros_like(Y)
        t0 = time.time()
        for tr, te in gkf.split(X, Y, groups):
            P[te] = predict(fit(make_ridge(), X[tr], Y[tr], w[tr]), X[te])
        per = spearman_per_dim(Y, P)
        rows.append({"features": "+".join(s), "n_features": X.shape[1], "mean_spearman": float(np.mean(per)),
                     "mae": float(np.abs(P - Y).mean()), **dict(zip(TASTE_DIMS, per)), "secs": time.time() - t0})
        print(f"[cv] {rows[-1]['features']:<40} rho={rows[-1]['mean_spearman']:.3f} mae={rows[-1]['mae']:.2f}")
    return rows


def write_cv_report(rows: list[dict], n_train: int) -> None:
    lines = [f"# Feature-set comparison (5-fold grouped CV, RidgeCV, n={n_train})", "",
             "| features | #feat | mean Spearman | MAE | " + " | ".join(TASTE_DIMS) + " |",
             "|---|---|---|---|" + "---|" * len(TASTE_DIMS)]
    for r in sorted(rows, key=lambda r: -r["mean_spearman"]):
        lines.append(f"| {r['features']} | {r['n_features']} | **{r['mean_spearman']:.3f}** | {r['mae']:.2f} | "
                     + " | ".join(f"{r[d]:.2f}" for d in TASTE_DIMS) + " |")
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "feature_cv.md").write_text("\n".join(lines) + "\n")


def run(text_blocks: list[str] | None = None, try_mlp: bool = True, max_cv_rows: int = 20000) -> dict:
    config.seed_everything()
    text_blocks = ["bge", "qwen"] if text_blocks is None else text_blocks
    df, Y, w = load_training_frame()
    if len(df) < 50:
        raise RuntimeError(f"only {len(df)} labelled dishes — need consensus labels first")
    test = holdout_split(df)
    builder = FeatureBuilder().fit(df[~test])
    wanted = sorted({b for s in candidate_feature_sets(text_blocks) for b in s})
    blocks = {}
    for b in wanted:
        try:
            blocks.update(builder.blocks(df, [b]))
        except Exception as e:  # noqa: BLE001  (e.g. no Nebius key / network for qwen)
            print(f"[train] block {b} unavailable: {type(e).__name__}: {e}")
    sets = [s for s in candidate_feature_sets([t for t in text_blocks if t in blocks]) if all(b in blocks for b in s)]

    tr_idx = np.where(~test)[0]
    rng = np.random.default_rng(config.SEED)
    cv_idx = tr_idx if len(tr_idx) <= max_cv_rows else np.sort(rng.choice(tr_idx, max_cv_rows, replace=False))
    groups = name_groups(df)
    cv_rows = feature_cv({k: v[cv_idx] for k, v in blocks.items()}, Y[cv_idx], w[cv_idx], groups[cv_idx], sets)
    write_cv_report(cv_rows, len(cv_idx))
    best_set = max(cv_rows, key=lambda r: r["mean_spearman"])["features"].split("+")
    print(f"[train] best feature set: {best_set}")

    X = np.hstack([blocks[b] for b in best_set])
    Xtr, Ytr, wtr, Xte, Yte = X[~test], Y[~test], w[~test], X[test], Y[test]
    cands = {"ridge": make_ridge()}
    if try_mlp:
        cands["mlp"] = make_mlp()
    results = {}
    for name, m in cands.items():
        t0 = time.time()
        fit(m, Xtr, Ytr, wtr)
        P = predict(m, Xte)
        per = spearman_per_dim(Yte, P)
        results[name] = {"model": m, "mean_spearman": float(np.mean(per)), "per_dim": dict(zip(TASTE_DIMS, per)),
                         "mae_per_dim": dict(zip(TASTE_DIMS, np.abs(P - Yte).mean(0).tolist())),
                         "train_secs": time.time() - t0}
        print(f"[train] {name}: held-out mean Spearman {results[name]['mean_spearman']:.3f} "
              f"MAE {np.abs(P - Yte).mean():.2f} ({results[name]['train_secs']:.0f}s)")
    # Fallback rule: ship ridge unless the MLP is strictly better.
    best = max(results, key=lambda k: (results[k]["mean_spearman"], k == "ridge"))
    art = {
        "version": VERSION, "model_type": best, "pipeline": results[best]["model"], "feature_set": best_set,
        "builder": builder, "dims": TASTE_DIMS,
        "holdout": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()},
        "holdout_uids": df.loc[test, "uid"].tolist(), "n_train": int((~test).sum()), "n_test": int(test.sum()),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config.ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(art, ARTIFACT)
    print(f"[train] saved {ARTIFACT.name}: {best} on {'+'.join(best_set)} (n_train={art['n_train']})")
    return art


@app.command()
def main(text: str = typer.Option("bge,qwen", help="comma list of text blocks to compare: bge,qwen"),
         mlp: bool = typer.Option(True, help="also try the MLP")):
    run([t for t in text.split(",") if t], try_mlp=mlp)


if __name__ == "__main__":
    app()
