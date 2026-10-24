"""TasteNet inference. No LLM calls at runtime (a Qwen embedding call only if the shipped feature set uses
the qwen block and the dish text is not already cached).

predict_taste(dish) -> {"taste": {8 floats, 1dp, 0-10}, "taste_confidence": {8 floats 0-1}, "taste_model_version"}
taste_confidence = (1 - held-out MAE_d / 3) * (0.6 + 0.4 * ingredient-vocab coverage), clipped to [0.05, 1].

write_menu_predictions() writes data/processed/menu_dishes_tagged.json. It only touches taste fields
(taste, taste_confidence, taste_model_version, tagged_by="trained-regressor"; `verified` is passed through untouched);
allergens, dietary and ingredients are copied through untouched.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import typer

from .. import config
from ..schema import TASTE_DIMS, Dish, TasteVector
from . import calibrate
from .features import _listify
from .train import ARTIFACT, predict as _predict

app = typer.Typer(add_completion=False)


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not ARTIFACT.exists():
        raise FileNotFoundError("artifacts/tastenet.joblib missing — run `make train`")
    return joblib.load(ARTIFACT)


def _row(d) -> dict:
    if isinstance(d, Dish):
        d = d.model_dump()
    d = dict(d)
    canon = _listify(d.get("ingredients_canonical"))
    d["ingredients_canonical"] = canon or _listify(d.get("ingredients"))
    d.setdefault("description", "")
    d.setdefault("menu_style", True)
    return d


def predict_batch(dishes: list, calibrated: bool = True) -> list[dict]:
    art = load_model()
    rows = [_row(d) for d in dishes]
    if not rows:
        return []
    df = pd.DataFrame(rows)
    X = art["builder"].transform(df, art["feature_set"])
    P = _predict(art["pipeline"], X)
    # config.SERVE_CALIBRATION is the master switch: off means raw scores even if calibrators exist.
    cals = calibrate.load(fitted_for=art["trained_at"]) if (calibrated and config.SERVE_CALIBRATION) else None
    if cals is not None:
        P = calibrate.apply(cals, P)
    mae = art["holdout"][art["model_type"]]["mae_per_dim"]
    base = np.array([np.clip(1 - mae[d] / 3.0, 0.05, 1.0) for d in TASTE_DIMS])
    cov = art["builder"].coverage(rows)
    conf = np.clip(base[None, :] * (0.6 + 0.4 * cov[:, None]), 0.05, 1.0)
    out = []
    for p, c in zip(P, conf):
        p = np.clip(p, 0, 10)
        out.append({"taste": {d: round(float(v), 1) for d, v in zip(TASTE_DIMS, p)},
                    "taste_confidence": {d: round(float(v), 2) for d, v in zip(TASTE_DIMS, c)},
                    "taste_model_version": art["version"], "calibrated": cals is not None})
    return out


def predict_taste(dish, calibrated: bool = True) -> dict:
    return predict_batch([dish], calibrated)[0]


def write_menu_predictions(calibrated: bool = True) -> int:
    src = config.DATA_PROCESSED / "menu_dishes.json"
    dishes = json.loads(src.read_text())
    preds = predict_batch(dishes, calibrated)
    out = []
    for d, p in zip(dishes, preds):
        d = dict(d)  # copy: allergens / dietary / ingredients pass through untouched
        d["taste"] = TasteVector(**p["taste"]).model_dump()
        d["taste_confidence"] = p["taste_confidence"]
        d["taste_model_version"] = p["taste_model_version"]
        d["tagged_by"] = "trained-regressor"
        # `verified` belongs to the human allergen/dietary check that filters rely on: never set it
        # to True from a model, and never clear a human's True either. Default False if absent.
        d["verified"] = bool(d.get("verified", False))
        out.append(Dish.model_validate(d).model_dump())
    dst = config.DATA_PROCESSED / "menu_dishes_tagged.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[predict] wrote {len(out)} dishes -> {dst}")
    return len(out)


@app.command()
def main(raw: bool = typer.Option(False, help="skip isotonic calibration")):
    write_menu_predictions(calibrated=not raw)


if __name__ == "__main__":
    app()
