"""Per-dim isotonic calibration of any predictor onto the human gold scale.

cross_fit(): 5-fold cross-fitted predictions on gold (unbiased gold metrics for the calibrated column).
fit_final(): fit on all gold -> artifacts/calibrators.joblib, applied by predict.py.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from .. import config
from ..schema import TASTE_DIMS

CALIBRATORS = config.ARTIFACTS / "calibrators.joblib"


def _iso() -> IsotonicRegression:
    return IsotonicRegression(y_min=0, y_max=10, out_of_bounds="clip", increasing=True)


def cross_fit(P: np.ndarray, Y: np.ndarray, n_splits: int = 5, seed: int = config.SEED) -> np.ndarray:
    """P, Y: (n_gold, 8). Returns out-of-fold calibrated predictions."""
    out = np.zeros_like(P, dtype=float)
    k = min(n_splits, len(P))
    if k < 2:
        return P.astype(float).copy()
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(P):
        for j in range(P.shape[1]):
            out[te, j] = _iso().fit(P[tr, j], Y[tr, j]).predict(P[te, j])
    return out


def fit_final(P: np.ndarray, Y: np.ndarray, fitted_for: str = "", save: bool = True,
              use: bool = True) -> dict[str, IsotonicRegression]:
    """fitted_for = TasteNet trained_at stamp, so predict.py can ignore calibrators from an older model.
    use = False when cross-fitting showed calibration does not help (predict.py then serves raw scores)."""
    cals = {d: _iso().fit(P[:, j], Y[:, j]) for j, d in enumerate(TASTE_DIMS)}
    if save:
        config.ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump({"fitted_for": fitted_for, "n_gold": int(len(P)), "use": use, "cals": cals}, CALIBRATORS)
    return cals


def load(fitted_for: str | None = None) -> dict | None:
    """Returns {dim: IsotonicRegression} or None (missing, or fitted for a different model)."""
    if not CALIBRATORS.exists():
        return None
    obj = joblib.load(CALIBRATORS)
    if fitted_for is not None and (obj.get("fitted_for") != fitted_for or not obj.get("use", True)):
        return None
    return obj["cals"]


def apply(cals: dict, P: np.ndarray) -> np.ndarray:
    P = np.atleast_2d(np.asarray(P, float))
    return np.column_stack([cals[d].predict(P[:, j]) for j, d in enumerate(TASTE_DIMS)])
