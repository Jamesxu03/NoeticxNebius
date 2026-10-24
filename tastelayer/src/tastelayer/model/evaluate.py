"""Model comparison on the human gold set -> reports/eval.md, eval.csv, eval.png, next_to_label.csv.

Columns: claude (if labelled), each Nebius model
(temperature 0.2, main pass), 3-model vote (consensus), vote calibrated (cross-fitted), Qwen review pass,
lexicon baseline, TasteNet raw, TasteNet calibrated (5-fold cross-fitted isotonic).
Metrics per dim + mean: Spearman (1000x bootstrap 95% CI), MAE, % within ±1.
Fallback without gold: reference = consensus on TasteNet's held-out split (LLM columns are then circular).
Side effect: fits artifacts/calibrators.joblib on all gold (TasteNet raw -> human).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr

from .. import config
from ..lexicon import predict_lookup_df
from ..schema import TASTE_DIMS
from . import calibrate
from .predict import load_model, predict_batch
from .train import load_dishes

app = typer.Typer(add_completion=False)
N_BOOT = 1000


# ---------------------------------------------------------------- metrics
def _rho(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def metrics(Y: np.ndarray, P: np.ndarray, n_boot: int = N_BOOT, seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(Y)
    rho = np.array([_rho(Y[:, j], P[:, j]) for j in range(8)])
    boots = np.full((n_boot, 8), np.nan)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boots[b] = [_rho(Y[i, j], P[i, j]) for j in range(8)]
    mae = np.abs(Y - P).mean(0)
    w1 = (np.abs(Y - P) <= 1.0).mean(0) * 100
    mean_boot = np.nanmean(boots, 1)
    out = {"n": n}
    for j, d in enumerate(TASTE_DIMS):
        out[d] = {"spearman": rho[j], "ci": tuple(np.nanpercentile(boots[:, j], [2.5, 97.5])) if n >= 3 else (np.nan, np.nan),
                  "mae": float(mae[j]), "within1": float(w1[j])}
    out["mean"] = {"spearman": float(np.nanmean(rho)), "ci": tuple(np.nanpercentile(mean_boot, [2.5, 97.5])),
                   "mae": float(mae.mean()), "within1": float(w1.mean())}
    return out


# ---------------------------------------------------------------- inputs
def load_gold() -> pd.DataFrame | None:
    p = config.GOLD / "gold.json"
    if not p.exists():
        return None
    g = json.loads(p.read_text())
    if not g:
        return None
    return pd.DataFrame([{"uid": x["uid"], **{d: float(x["taste"][d]) for d in TASTE_DIMS}} for x in g])


def _labels() -> pd.DataFrame | None:
    p = config.DATA_PROCESSED / "labels.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _consensus() -> pd.DataFrame | None:
    p = config.DATA_PROCESSED / "labels_consensus.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _short(m: str) -> str:
    return config.MODEL_SHORT.get(m, m.split("/")[-1].split(":")[-1].lower())


def method_predictions(ref: pd.DataFrame, dishes: pd.DataFrame, gold_mode: bool) -> dict[str, pd.DataFrame]:
    """Returns {method: DataFrame(uid + 8 dims)} restricted to uids in ref."""
    uids = set(ref.uid)
    out: dict[str, pd.DataFrame] = {}
    lab = _labels()
    if lab is not None:
        lab = lab[lab.uid.isin(uids)]
        main = lab[lab["pass"] == "main"] if "pass" in lab else lab
        for m in sorted(main.model.unique()):
            sub = main[main.model == m]
            if "temperature" in sub and (sub.temperature.round(2) == 0.2).any():
                sub = sub[sub.temperature.round(2) == 0.2]
            out["claude" if "claude" in m.lower() else _short(m)] = sub.groupby("uid")[TASTE_DIMS].mean().reset_index()
        if "pass" in lab and (lab["pass"] == "review").any():
            out["qwen-review"] = lab[lab["pass"] == "review"].groupby("uid")[TASTE_DIMS].mean().reset_index()
    cons = _consensus()
    if cons is not None:
        out["3-model vote"] = cons[cons.uid.isin(uids)][["uid", *TASTE_DIMS]]
    rows = dishes[dishes.uid.isin(uids)].reset_index(drop=True)
    if len(rows):
        out["lexicon"] = pd.DataFrame(predict_lookup_df(rows), columns=TASTE_DIMS).assign(uid=rows.uid.values)
        try:
            t0 = time.time()
            preds = predict_batch(rows.to_dict("records"), calibrated=False)
            _TIMING["tastenet_s_per_dish"] = (time.time() - t0) / len(rows)
            out["TasteNet raw"] = pd.DataFrame([p["taste"] for p in preds]).assign(uid=rows.uid.values)
        except FileNotFoundError:
            pass
    return out


MIN_GOLD_FOR_CALIBRATION = 40
_TIMING: dict[str, float] = {}


def aligned(ref: pd.DataFrame, pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    m = ref.merge(pred, on="uid", suffixes=("_y", "_p"))
    return (m[[f"{d}_y" for d in TASTE_DIMS]].to_numpy(float), m[[f"{d}_p" for d in TASTE_DIMS]].to_numpy(float),
            m.uid.tolist())


# ---------------------------------------------------------------- cost / latency
def cost_rows(feature_set: list[str]) -> pd.DataFrame:
    p = config.CACHE_DIR / "calls.jsonl"
    rows = []
    if p.exists():
        calls = pd.read_json(p, lines=True)
        if "kind" in calls and len(calls):
            ok = calls[calls.get("status", "ok").astype(str).str.startswith("ok")] if "status" in calls else calls
            dish = ok[ok.kind == "dish"]
            for m, g in dish.groupby("model"):
                n = g["n"].fillna(10).sum()
                lat = (g["latency"] / g["n"].fillna(10)).mean()
                rows.append({"method": _short(m), "usd_per_1k_dishes": 1000 * g["cost"].sum() / max(n, 1),
                             "runtime_llm_calls_per_dish": 1 / max(g["n"].fillna(10).mean(), 1),
                             "latency_s_per_dish": lat})
            lab = _labels()
            if lab is not None and len(dish):
                n_uid = lab[lab["pass"] == "main"].uid.nunique() if "pass" in lab else lab.uid.nunique()
                tot = dish["cost"].sum()
                rows.append({"method": "3-model vote", "usd_per_1k_dishes": 1000 * tot / max(n_uid, 1),
                             "runtime_llm_calls_per_dish": float("nan"), "latency_s_per_dish": float("nan")})
            emb = ok[ok.kind == "embedding"]
            emb_cost = 1000 * emb["cost"].sum() / max(emb["n"].fillna(1).sum(), 1) if len(emb) else 0.0
        else:
            emb_cost = 0.0
    else:
        emb_cost = 0.0
    rows.append({"method": "TasteNet", "usd_per_1k_dishes": emb_cost if "qwen" in feature_set else 0.0,
                 "runtime_llm_calls_per_dish": 0.0, "latency_s_per_dish": _TIMING.get("tastenet_s_per_dish", float("nan"))})
    rows.append({"method": "lexicon", "usd_per_1k_dishes": 0.0, "runtime_llm_calls_per_dish": 0.0,
                 "latency_s_per_dish": float("nan")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Indian Food 101 sanity
def indian101_auroc(dishes: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    def fp(x):
        try:
            return (json.loads(x) if isinstance(x, str) else (x or {})).get("flavor_profile")
        except (json.JSONDecodeError, AttributeError):
            return None

    d = dishes.copy()
    d["fp"] = d["extra"].map(fp)
    d = d[d.fp.notna() & (d.fp.astype(str) != "-1")].reset_index(drop=True)
    if len(d) < 20:
        return {}
    res = {"n": len(d)}
    preds = {"lexicon": pd.DataFrame(predict_lookup_df(d), columns=TASTE_DIMS)}
    try:
        preds["TasteNet raw"] = pd.DataFrame([p["taste"] for p in predict_batch(d.to_dict("records"), calibrated=False)])
    except FileNotFoundError:
        pass
    cons = _consensus()
    if cons is not None:
        m = d[["uid"]].merge(cons, on="uid", how="left")
        if m[TASTE_DIMS].notna().all(axis=1).sum() > 20:
            preds["3-model vote"] = m[TASTE_DIMS]
    for name, P in preds.items():
        r = {}
        for dim, flav in [("spice", "spicy"), ("sweet", "sweet")]:
            y = (d.fp == flav).to_numpy()
            ok = P[dim].notna().to_numpy()
            if 0 < y[ok].sum() < ok.sum():
                r[f"{dim}_vs_{flav}"] = round(float(roc_auc_score(y[ok], P[dim].to_numpy()[ok])), 3)
        res[name] = r
    return res


# ---------------------------------------------------------------- active learning
def next_to_label(gold_uids: set[str]) -> pd.DataFrame | None:
    mp = config.DATA_PROCESSED / "menu_dishes.json"
    cons = _consensus()
    if not mp.exists() or cons is None:
        return None
    menu = [d for d in json.loads(mp.read_text()) if d["id"] not in gold_uids]
    if not menu:
        return None
    df = pd.DataFrame([{"uid": d["id"], "name": d["name"], "cuisine": d["cuisine"]} for d in menu])
    df = df.merge(cons, on="uid", how="left", suffixes=("", "_c"))
    df["ens_std"] = df[[f"{d}_std" for d in TASTE_DIMS]].mean(axis=1).fillna(3.0)
    try:
        preds = predict_batch(menu, calibrated=False)
        P = np.array([[p["taste"][d] for d in TASTE_DIMS] for p in preds])
        V = df[TASTE_DIMS].to_numpy(float)
        df["tastenet_vs_vote"] = np.nan_to_num(np.abs(P - V).mean(1), nan=3.0)
    except FileNotFoundError:
        df["tastenet_vs_vote"] = 0.0
    df["score"] = df.ens_std + df.tastenet_vs_vote
    out = df.sort_values(["score", "uid"], ascending=[False, True])[
        ["uid", "name", "cuisine", "ens_std", "tastenet_vs_vote", "score"]]
    out.to_csv(config.REPORTS / "next_to_label.csv", index=False)
    return out


# ---------------------------------------------------------------- report
def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


ALT_MODEL = config.ARTIFACTS / "tastenet_qwen.joblib"
ALT_EVAL = config.REPORTS / "eval_qwen_alt.csv"


def served_model_line() -> str:
    p = config.ARTIFACTS / "tastenet.joblib"
    if not p.exists():
        return ""
    import joblib

    a = joblib.load(p)
    net = "one Qwen3-Embedding API call per unseen dish" if "qwen" in a["feature_set"] else "no network calls (bge-base loaded from the local cache)"
    ho = a["holdout"][a["model_type"]]["mean_spearman"]
    return (f"Served model: **TasteNet {a['model_type']} on {'+'.join(a['feature_set'])}**, held-out ρ vs consensus "
            f"{ho:.2f}; runtime: {net}.")


def alternative_section(tag: str) -> list[str]:
    """Not-served alternative (TasteNet with the Qwen3-Embedding feature), from its saved evaluation."""
    if not (ALT_MODEL.exists() and ALT_EVAL.exists()):
        return []
    import joblib

    a = joblib.load(ALT_MODEL)
    df = pd.read_csv(ALT_EVAL)
    t = df[df.method == "TasteNet raw"].set_index("dim")
    if t.empty:
        return []
    ho = a["holdout"][a["model_type"]]["mean_spearman"]
    L = ["", f"## Alternative, not served: TasteNet with Qwen3-Embedding features{tag}", "",
         f"{a['model_type']} on {'+'.join(a['feature_set'])} (`artifacts/tastenet_qwen.joblib`); held-out ρ vs consensus "
         f"{ho:.2f}. Needs one Qwen3-Embedding API call per unseen dish. Full report: `reports/eval_qwen_alt.md`.", "",
         "| dim | Spearman (95% CI) | MAE | % within ±1 |", "|---|---|---|---|"]
    for d in TASTE_DIMS + ["mean"]:
        if d in t.index:
            r = t.loc[d]
            L.append(f"| {'**mean**' if d == 'mean' else d} | {r.spearman:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] | {r.mae:.2f} | {r.within1_pct:.0f} |")
    return L


def write_report(results: dict[str, dict], reference: str, costs: pd.DataFrame, auroc: dict, human: dict | None,
                 notes: list[str]) -> None:
    order = (["claude"] if "claude" in results else []) + [k for k in results if k != "claude"]
    recs = []
    for meth in order:
        r = results[meth]
        for d in TASTE_DIMS + ["mean"]:
            x = r[d]
            ci = x.get("ci", (np.nan, np.nan))
            recs.append({"method": meth, "dim": d, "n": r["n"], "spearman": x["spearman"], "ci_lo": ci[0],
                         "ci_hi": ci[1], "mae": x["mae"], "within1_pct": x["within1"]})
    long = pd.DataFrame(recs)
    long.to_csv(config.REPORTS / "eval.csv", index=False)

    L = [f"# TasteNet evaluation", "", f"Reference: **{reference}**. Generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    sm = served_model_line()
    if sm:
        L += [sm, ""]
    gp = config.GOLD / "gold.json"
    if reference.startswith("human gold") and gp.exists():
        g = json.loads(gp.read_text())
        anns = sorted({a for x in g for a in x.get("annotators", [])})
        cz = {c: sum(x.get("cuisine") == c for x in g) for c in ("indian", "italian")}
        L += [f"> **Caveat:** the gold set is small (n={len(g)}: {cz['indian']} Indian, {cz['italian']} Italian) and labelled by "
              f"a single human labeller. Per-dim 95% bootstrap CIs are wide (often ±0.3 ρ): treat differences between methods"
              f" smaller than the CI width as noise. The labeller's scale also runs lower than the rubric's on richness/umami/"
              f"creaminess/sour (see reports/gold_vs_consensus.md), which inflates MAE for every method but not Spearman.", ""]
    tag = f" — n={results[order[0]]['n']}, single human labeller" if reference.startswith("human gold") else ""
    L += [f"## Spearman ρ (95% bootstrap CI){tag}", "", "| dim | " + " | ".join(f"{m} (n={results[m]['n']})" for m in order) + " |",
          "|---|" + "---|" * len(order)]
    for d in TASTE_DIMS + ["mean"]:
        cells = []
        for m in order:
            x = results[m][d]
            ci = x.get("ci", (np.nan, np.nan))
            cells.append(f"{_fmt(x['spearman'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if not np.isnan(ci[0]) else _fmt(x["spearman"]))
        L.append(f"| {'**mean**' if d == 'mean' else d} | " + " | ".join(cells) + " |")
    for metric, title, nd in [("mae", "MAE (0-10 scale, lower is better)", 2), ("within1", "% within ±1", 0)]:
        L += ["", f"## {title}{tag}", "", "| dim | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for d in TASTE_DIMS + ["mean"]:
            L.append(f"| {'**mean**' if d == 'mean' else d} | " + " | ".join(_fmt(results[m][d][metric], nd) for m in order) + " |")
    L += ["", "## Cost and runtime (not gold-based: from this run's token logs and timings)", "", "| method | $ per 1k dishes | runtime LLM calls / dish | latency s / dish |",
          "|---|---|---|---|"]
    for r in costs.itertuples():
        L.append(f"| {r.method} | {_fmt(r.usd_per_1k_dishes, 3)} | {_fmt(r.runtime_llm_calls_per_dish, 2)} | "
                 f"{_fmt(r.latency_s_per_dish, 3)} |")
    if auroc:
        L += ["", f"## Indian Food 101 sanity check (n={auroc['n']}, AUROC; not gold-based: dataset flavour labels)", "", "| method | spice vs 'spicy' | sweet vs 'sweet' |",
              "|---|---|---|"]
        for k, v in auroc.items():
            if k != "n":
                L.append(f"| {k} | {v.get('spice_vs_spicy', '–')} | {v.get('sweet_vs_sweet', '–')} |")
    L += alternative_section(tag)
    L += ["", "## Interpretation", ""] + [f"{i + 1}. {n}" for i, n in enumerate(notes[:5])]
    (config.REPORTS / "eval.md").write_text("\n".join(L) + "\n")
    _plot(results, order)


def _plot(results, order):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.2))
    k = len(order)
    w = 0.8 / max(k, 1)
    x = np.arange(len(TASTE_DIMS) + 1)
    for i, m in enumerate(order):
        ys = [results[m][d]["spearman"] for d in TASTE_DIMS + ["mean"]]
        ax.bar(x + (i - (k - 1) / 2) * w, np.nan_to_num(ys), w, label=m)
    ax.set_xticks(x, TASTE_DIMS + ["mean"])
    ax.set_ylabel("Spearman ρ vs reference")
    ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
    ax.axhline(0, color="#888", lw=0.6)
    ax.legend(ncol=min(k, 5), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.REPORTS / "eval.png", dpi=150)
    plt.close(fig)


def interpretation(results: dict, costs: pd.DataFrame, reference: str) -> list[str]:
    ms = {m: r["mean"]["spearman"] for m, r in results.items() if m != "human-human"}
    notes = []
    if not ms:
        return ["No methods could be evaluated."]
    best = max(ms, key=lambda m: ms[m])
    notes.append(f"Best mean Spearman vs {reference}: **{best}** ({ms[best]:.2f}).")
    served = "TasteNet calibrated" if _TIMING.get("calibration_used") else "TasteNet raw"
    tn = next((m for m in [served, "TasteNet raw", "TasteNet calibrated"] if m in ms), None)
    if tn and "3-model vote" in ms:
        gap = ms["3-model vote"] - ms[tn]
        notes.append(f"{tn} is {abs(gap):.2f} ρ {'below' if gap > 0 else 'above'} the 3-model vote it was distilled from, "
                     f"with zero runtime LLM calls"
                     + (" and no network calls." if not _TIMING.get("uses_qwen", 1.0) else "."))
    if "lexicon" in ms and tn:
        notes.append(f"Learning adds {ms[tn] - ms['lexicon']:+.2f} ρ over the hand-written lexicon baseline ({ms['lexicon']:.2f}).")
    ref = results.get(tn or best)
    if ref:
        per = {d: ref[d]["spearman"] for d in TASTE_DIMS}
        lo, hi = min(per, key=lambda d: np.nan_to_num(per[d], nan=9)), max(per, key=lambda d: np.nan_to_num(per[d], nan=-9))
        notes.append(f"Easiest dim for {tn or best}: {hi} ({per[hi]:.2f}); hardest: {lo} ({per[lo]:.2f}).")
    tc = costs[costs.method == "TasteNet"]
    vc = costs[costs.method == "3-model vote"]
    if len(tc) and len(vc):
        notes.append(f"Cost per 1k dishes: TasteNet ${tc.usd_per_1k_dishes.iloc[0]:.3f} vs 3-model vote "
                     f"${vc.usd_per_1k_dishes.iloc[0]:.2f}.")
    return notes


def run() -> dict:
    config.seed_everything()
    config.REPORTS.mkdir(exist_ok=True)
    dishes = load_dishes()
    gold = load_gold()
    try:
        art = load_model()
    except FileNotFoundError:
        art = None
    if gold is not None:
        ref, reference = gold, f"human gold set (n={len(gold)})"
    else:
        cons = _consensus()
        if art is None or cons is None:
            raise RuntimeError("no gold/gold.json and no trained model + consensus to fall back on")
        ref = cons[cons.uid.isin(art["holdout_uids"])][["uid", *TASTE_DIMS]]
        reference = f"held-out 3-model consensus (n={len(ref)}; no gold yet — LLM columns are circular)"
    preds = method_predictions(ref, dishes, gold is not None)
    results: dict[str, dict] = {}
    for m, P in preds.items():
        Y, Pm, _ = aligned(ref, P)
        if len(Y) >= 3:
            results[m] = metrics(Y, Pm)
    if gold is not None:
        for base, name in [("TasteNet raw", "TasteNet calibrated"), ("3-model vote", "vote calibrated")]:
            if base in preds:
                Y, Pm, _ = aligned(ref, preds[base])
                if len(Y) >= 10:
                    results[name] = metrics(Y, calibrate.cross_fit(Pm, Y))
                    if base == "TasteNet raw" and art is not None:
                        # Serve calibration only if it lowers MAE without hurting the ranking, and the gold
                        # set is big enough to trust (isotonic on ~30 noisy labels mostly learns the annotator's scale).
                        cal, raw = results[name]["mean"], results[base]["mean"]
                        use = (cal["mae"] < raw["mae"] and cal["spearman"] >= raw["spearman"] - 0.02
                               and len(Y) >= MIN_GOLD_FOR_CALIBRATION and config.SERVE_CALIBRATION)
                        calibrate.fit_final(Pm, Y, fitted_for=art["trained_at"], use=use)
                        _TIMING["calibration_used"] = float(use)
    human = None  # single-labeller gold: no human-vs-human comparison is reported
    _TIMING["uses_qwen"] = float(bool(art and "qwen" in art["feature_set"]))
    costs = cost_rows(art["feature_set"] if art else [])
    auroc = indian101_auroc(dishes)
    notes = interpretation(results, costs, "gold" if gold is not None else "held-out consensus")
    if "calibration_used" in _TIMING:
        notes.insert(2, "predict.py serves " + ("calibrated" if _TIMING["calibration_used"] else "raw")
                     + " TasteNet scores. Calibration is reported (cross-fitted) but switched off for serving"
                     + " (config.SERVE_CALIBRATION=False)." if not config.SERVE_CALIBRATION else
                     "predict.py serves " + ("calibrated" if _TIMING["calibration_used"] else "raw")
                     + f" TasteNet scores (calibration is used only if it lowers cross-fitted MAE, keeps Spearman within"
                     + f" 0.02, and gold n >= {MIN_GOLD_FOR_CALIBRATION}).")
    write_report(results, reference, costs, auroc, human, notes)
    nl = next_to_label(set(gold.uid) if gold is not None else set())
    print((config.REPORTS / "eval.md").read_text())
    if nl is not None:
        print(f"[eval] next_to_label.csv: top 5 -> {nl.head(5).name.tolist()}")
    return results


@app.command()
def main():
    run()


if __name__ == "__main__":
    app()
