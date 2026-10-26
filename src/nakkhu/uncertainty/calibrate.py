"""Existence-probability calibration and weighted ECE (U1; prereg v5 ``uncertainty``).

Calibrators map the raw ensemble score s in (0, 1] to P(real building):

- temperature: p = sigmoid(logit(s) / T), T = exp(u), u in ``log_t_bounds``, fitted by
  design-weighted log loss (s clipped to [clip, 1 - clip]);
- isotonic: non-decreasing step function by weighted pool-adjacent-violators, linear between
  knots and flat beyond them.

Held-out evaluation is leave-one-block-out over plan 005 blocks (a unit never sees its own
block). ECE: 10 equal-count bins of the pooled out-of-fold predictions; within a bin the
design-weighted mean prediction and observed rate; ECE = sum_b (W_b / W) |obs_b - pred_b|.
The method with the lower pooled out-of-fold weighted ECE is chosen (tie: temperature).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

CLIP = 1.0e-4
LOG_T_BOUNDS = (-5.0, 5.0)
METHODS = ("temperature", "isotonic")


def _logit(s: np.ndarray, clip: float = CLIP) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), clip, 1.0 - clip)
    return np.log(s / (1.0 - s))


def fit_temperature(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *,
                    clip: float = CLIP, bounds: tuple[float, float] = LOG_T_BOUNDS) -> float:
    z = _logit(s, clip)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)

    def loss(u: float) -> float:
        p = 1.0 / (1.0 + np.exp(-z / np.exp(u)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))

    res = optimize.minimize_scalar(loss, bounds=bounds, method="bounded",
                                   options={"xatol": 1e-8})
    return float(np.exp(res.x))


def predict_temperature(t: float, s: np.ndarray, clip: float = CLIP) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(s, clip) / t))


def fit_isotonic(s: np.ndarray, y: np.ndarray,
                 w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted non-decreasing PAV; knots ``(unique s, fitted p)``."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    ux, inv = np.unique(s, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sy = np.bincount(inv, weights=w * y)
    level: list[float] = []
    weight: list[float] = []
    size: list[int] = []
    for num, den in zip(sy, sw, strict=True):
        level.append(num / den if den > 0 else 0.0)
        weight.append(den)
        size.append(1)
        while len(level) > 1 and level[-2] > level[-1]:
            wn = weight[-2] + weight[-1]
            ln = (level[-2] * weight[-2] + level[-1] * weight[-1]) / wn if wn > 0 else 0.0
            level[-2:], weight[-2:], size[-2:] = [ln], [wn], [size[-2] + size[-1]]
    return ux, np.repeat(np.asarray(level), size)


def predict_isotonic(knots: tuple[np.ndarray, np.ndarray], s: np.ndarray) -> np.ndarray:
    ux, p = knots
    return np.interp(np.asarray(s, dtype=float), ux, p)


Predictor = Callable[[np.ndarray], np.ndarray]


def fit_method(method: str, s: np.ndarray, y: np.ndarray, w: np.ndarray) -> Predictor:
    """Fit one calibration method and return its predictor."""
    if method == "temperature":
        t = fit_temperature(s, y, w)
        return lambda x: predict_temperature(t, x)
    if method == "isotonic":
        knots = fit_isotonic(s, y, w)
        return lambda x: predict_isotonic(knots, x)
    raise ValueError(f"unknown calibration method {method!r}")


def cross_fit(s: np.ndarray, y: np.ndarray, w: np.ndarray, blocks: np.ndarray,
              method: str, fit: Callable[..., Predictor] = fit_method) -> np.ndarray:
    """Leave-one-block-out predictions; NaN where the other blocks lack a class."""
    s, y, w = (np.asarray(v, dtype=float) for v in (s, y, w))
    blocks = np.asarray(blocks)
    out = np.full(len(s), np.nan)
    for blk in np.unique(blocks):
        test = blocks == blk
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        out[test] = fit(method, s[train], y[train], w[train])(s[test])
    return out




def reliability(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of ``p`` (finite only): n, weight, weighted mean p, weighted rate."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(p)
    p, y, w = p[ok], y[ok], w[ok]
    order = np.argsort(p, kind="stable")
    rows = []
    for k, ix in enumerate(np.array_split(order, bins)):
        if not len(ix):
            continue
        wb = w[ix].sum()
        rows.append({"bin": k, "n": int(len(ix)), "weight": float(wb),
                     "p_mean": float(np.sum(w[ix] * p[ix]) / wb),
                     "observed": float(np.sum(w[ix] * y[ix]) / wb),
                     "p_mean_unweighted": float(p[ix].mean()),
                     "observed_unweighted": float(y[ix].mean())})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, bins: int = 10,
        weighted: bool = True) -> float:
    rel = reliability(p, y, w, bins)
    if rel.empty:
        return float("nan")
    if weighted:
        gap = (rel["observed"] - rel["p_mean"]).abs()
        return float(np.sum(rel["weight"] / rel["weight"].sum() * gap))
    gap = (rel["observed_unweighted"] - rel["p_mean_unweighted"]).abs()
    return float(np.sum(rel["n"] / rel["n"].sum() * gap))


def brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(p)
    return float(np.sum(w[ok] * (p[ok] - y[ok]) ** 2) / w[ok].sum())


def calibrate_u1(s: np.ndarray, y: np.ndarray, w: np.ndarray, blocks: np.ndarray, *,
                 bins: int = 10, methods: tuple[str, ...] = METHODS
                 ) -> dict[str, Any]:
    """Per method: out-of-fold ECE (weighted and unweighted), Brier and reliability; the
    chosen method (lowest weighted ECE, tie -> first in ``methods`` = temperature) and its
    predictor fitted on all audited units."""
    s, y, w = (np.asarray(v, dtype=float) for v in (s, y, w))
    out: dict[str, Any] = {"methods": {}}
    for m in methods:
        oof = cross_fit(s, y, w, blocks, m)
        ok = np.isfinite(oof)
        out["methods"][m] = {
            "ece": ece(oof, y, w, bins, weighted=True),
            "ece_unweighted": ece(oof, y, w, bins, weighted=False),
            "brier": brier(oof, y, w), "n_scored": int(ok.sum()), "n": int(len(y)),
            "reliability": reliability(oof, y, w, bins), "oof": oof}
    def key(m: str) -> tuple[float, int]:
        e = out["methods"][m]["ece"]
        return (e if np.isfinite(e) else np.inf, methods.index(m))

    best = min(methods, key=key)
    out["chosen"] = best
    out["predict"] = fit_method(best, s, y, w)
    return out
