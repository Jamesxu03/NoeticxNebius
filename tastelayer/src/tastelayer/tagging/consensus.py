"""Consensus labels across models/temperatures + disagreement report.

Per uid, per dim:
  label      = median of all main-pass labels (3 models x 2 temps -> up to 6 values)
  taste_std  = population std of those labels
  confidence = 1 - min(std / 3, 1)
Review pass (Qwen3-235B on high-disagreement dishes): the review label is added to the pool of
main labels with weight 2 (two copies) and the median is recomputed. Taking median(main_median,
review, review) instead would always just return the review label, discarding the panel; adding
it twice to the panel lets one careful re-rating pull the median without overriding 6 votes.
std/confidence stay those of the main panel (honest about the original disagreement); `reviewed`=True.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tastelayer import config
from tastelayer.schema import TASTE_DIMS, Dish, TasteVector
from tastelayer.tagging.tag import DISAGREE, LABELS, MENU, load_dish_rows

CONSENSUS = config.DATA_PROCESSED / "labels_consensus.parquet"
MENU_BY_MODEL = config.DATA_PROCESSED / "menu_dishes_by_model.json"
STD_FLAG = 2.0
TAGGED_BY = "nebius:consensus"


def compute_consensus(labels: pd.DataFrame) -> pd.DataFrame:
    main = labels[labels["pass"] == "main"]
    review = labels[labels["pass"] == "review"]
    g = main.groupby("uid")[TASTE_DIMS]
    med = g.median()
    std = g.std(ddof=0).fillna(0.0)
    n = main.groupby("uid").size().rename("n_labels")
    out = med.copy()
    if not review.empty:
        rv = review.groupby("uid")[TASTE_DIMS].mean()
        for uid in rv.index.intersection(out.index):
            pool = main.loc[main["uid"] == uid, TASTE_DIMS].to_numpy(dtype=float)
            r = rv.loc[uid].to_numpy(dtype=float)
            out.loc[uid] = np.median(np.vstack([pool, r, r]), axis=0)
    res = pd.DataFrame(index=out.index)
    for d in TASTE_DIMS:
        res[d] = out[d].round(1)
        res[f"{d}_std"] = std[d].round(3)
        res[f"{d}_conf"] = (1 - np.minimum(std[d] / 3.0, 1.0)).round(3)
    res["n_labels"] = n
    res["reviewed"] = res.index.isin(review["uid"].unique()) if not review.empty else False
    res["tagged_by"] = TAGGED_BY
    return res.reset_index()


def disagreement_table(cons: pd.DataFrame, names: dict[str, dict]) -> pd.DataFrame:
    std_cols = [f"{d}_std" for d in TASTE_DIMS]
    flagged = cons[(cons[std_cols] > STD_FLAG).any(axis=1)].copy()
    flagged["name"] = flagged["uid"].map(lambda u: names.get(u, {}).get("name", ""))
    flagged["cuisine"] = flagged["uid"].map(lambda u: names.get(u, {}).get("cuisine", ""))
    flagged["dims"] = flagged.apply(lambda r: ",".join(d for d in TASTE_DIMS if r[f"{d}_std"] > STD_FLAG), axis=1)
    flagged["max_std"] = flagged[std_cols].max(axis=1)
    return flagged[["uid", "name", "cuisine", "dims", "max_std", *std_cols]].sort_values("max_std", ascending=False)


def menu_by_model(labels: pd.DataFrame) -> dict[str, list[dict]]:
    if not MENU.exists():
        return {}
    menu = {d["id"]: d for d in json.loads(MENU.read_text())}
    main = labels[(labels["pass"] == "main") & labels["uid"].isin(menu.keys())]
    full_id = {v: k for k, v in config.MODEL_SHORT.items()}
    out: dict[str, list[dict]] = {}
    for m, grp in main.groupby("model"):
        model_id = full_id.get(m, m)
        avg = grp.groupby("uid")[TASTE_DIMS].mean()  # mean over the two temperatures
        dishes = []
        for uid, row in avg.iterrows():
            d = dict(menu[uid])
            d["taste"] = TasteVector.from_list(row.round(1).tolist()).model_dump()
            d["tagged_by"] = f"nebius:{model_id}"
            d["verified"] = False
            dishes.append(Dish.model_validate(d).model_dump())
        out[model_id] = dishes
    return out


def run() -> pd.DataFrame:
    labels = pd.read_parquet(LABELS)
    cons = compute_consensus(labels)
    CONSENSUS.parent.mkdir(parents=True, exist_ok=True)
    cons.to_parquet(CONSENSUS, index=False)
    names = {r["uid"]: r for r in load_dish_rows()}
    dis = disagreement_table(cons, names)
    DISAGREE.parent.mkdir(parents=True, exist_ok=True)
    dis.to_csv(DISAGREE, index=False)
    mbm = menu_by_model(labels)
    if mbm:
        MENU_BY_MODEL.write_text(json.dumps(mbm, indent=1, ensure_ascii=False))
    print(f"consensus: {len(cons)} dishes, {len(dis)} flagged (any dim std>{STD_FLAG}), "
          f"{int(cons['reviewed'].sum())} reviewed; menu-by-model for {len(mbm)} models")
    return cons
