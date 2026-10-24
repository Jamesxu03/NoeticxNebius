"""SLERP rotation for the "take me further" slider (Epicure-style).

Rotate the user's ingredient centroid a toward target b = normalize(a + taste_direction[dim])
by an angle theta (degrees). theta=0 returns a unchanged.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    omega = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if omega < 1e-6:
        return a
    t = float(np.clip(t, 0.0, 1.0))
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def rotate_toward(a: np.ndarray, direction: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate unit vector a by theta degrees in the plane spanned by a and direction.

    Uses target b = normalize(a + direction); t = theta / angle(a, b), clipped to [0, 1], so the
    rotation never overshoots the target.
    """
    a = _unit(np.asarray(a, float))
    b = _unit(a + _unit(np.asarray(direction, float)))
    omega = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if omega < 1e-6 or theta_deg <= 0:
        return a
    return slerp(a, b, np.radians(theta_deg) / omega)


def shifted_taste_target(u: dict[str, float], dim: str, theta_deg: float, max_shift: float = 3.0) -> dict[str, float]:
    """u'_dim = u_dim + theta/90 * 3, clipped to [0, 10]."""
    out = dict(u)
    out[dim] = float(np.clip(out.get(dim, 5.0) + theta_deg / 90.0 * max_shift, 0, 10))
    return out
