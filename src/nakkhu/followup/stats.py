"""Phase-1 statistics of prereg_v9_a3 (plan 014), built on :mod:`nakkhu.flood.validate`.

ROC-AUC, PR-AUC, DeLong and the paired block bootstrap of a delta AUC are validate's own
functions (``common.metric`` / ``paired_test``). Statistics validate has no function for
(partial AUC, Spearman, stratified AUC, 90 % TOST intervals) use :func:`block_bootstrap`,
which draws exactly the resamples of ``validate.block_bootstrap_delta_auc`` (same block ids,
same generator, same seed), so every interval of a run shares its resamples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats

from nakkhu.flood import validate as V


def block_resamples(blocks: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices of each resample of whole blocks, as in validate's bootstrap."""
    ids, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    members = [np.nonzero(inverse == k)[0] for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def block_bootstrap(stat: Callable[[np.ndarray], float], blocks: np.ndarray, n_boot: int,
                    seed: int) -> np.ndarray:
    """Finite values of ``stat(idx)`` over the block resamples."""
    values = np.array([stat(idx) for idx in block_resamples(blocks, n_boot, seed)], dtype=float)
    return values[np.isfinite(values)]


def percentile_ci(values: np.ndarray, level: float = V.LEVEL) -> tuple[float, float]:
    if not np.size(values):
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def scorable(y: np.ndarray, min_cell: int) -> bool:
    y = np.asarray(y).astype(bool)
    return int(y.sum()) >= min_cell and int((~y).sum()) >= min_cell


def single(y: np.ndarray, score: np.ndarray, w: np.ndarray | None = None) -> dict[str, Any]:
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    yy, ww = np.asarray(y)[ok], None if w is None else np.asarray(w)[ok]
    return {"n": int(ok.sum()), "affected": int(np.sum(yy)), "n_nan": int((~ok).sum()),
            "auc": V.roc_auc(score[ok], yy, ww), "pr_auc": V.average_precision(score[ok], yy, ww)}


def pair(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, blocks: np.ndarray,
         w: np.ndarray | None, n_boot: int, seed: int, min_cell: int) -> dict[str, Any]:
    """Paired comparison of two scores (higher = more exposed) on the rows where both are
    finite: AUCs, PR-AUCs, delta AUC with validate's paired block bootstrap, DeLong
    (unweighted only). Below ``min_cell`` only counts are reported (``scored`` False)."""
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    y = np.asarray(y).astype(int)
    ok = np.isfinite(s1) & np.isfinite(s2)
    row: dict[str, Any] = {"n_rows": int(len(y)), "n_nan_candidate": int((~np.isfinite(s1)).sum()),
                           "n_nan_comparator": int((~np.isfinite(s2)).sum()),
                           "n": int(ok.sum()), "affected": int(y[ok].sum()),
                           "n_blocks": int(len(np.unique(np.asarray(blocks)[ok])))}
    if not scorable(y[ok], min_cell):
        return {**row, "scored": False, "reason": f"below min_cell ({min_cell})"}
    yy, a, b, bl = y[ok], s1[ok], s2[ok], np.asarray(blocks)[ok]
    ww = None if w is None else np.asarray(w, dtype=float)[ok]
    bb = V.block_bootstrap_delta_auc(yy, a, b, bl, n_boot, seed, ww)
    dl = V.delong(yy, a, b) if ww is None else {}
    return {**row, "scored": True, "reason": "",
            "auc_candidate": V.roc_auc(a, yy, ww), "auc_comparator": V.roc_auc(b, yy, ww),
            "pr_auc_candidate": V.average_precision(a, yy, ww),
            "pr_auc_comparator": V.average_precision(b, yy, ww),
            "delta_auc": bb["delta"], "boot_lo": bb["boot_lo"], "boot_hi": bb["boot_hi"],
            "n_boot_valid": bb["n_boot_valid"], "delong_se": dl.get("se"),
            "delong_p": dl.get("p")}


# --- natural cubic splines and logistic fits (P1-3, P1-6 M-C) ----------------------------------


def ns_knots(x: np.ndarray) -> np.ndarray:
    """Boundary knots at the 1st / 99th and interior knots at the 25 / 50 / 75th percentiles
    of the training values (prereg_v9_a3 P1-3 ``basis``)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])


def ns_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis without intercept (df = len(knots) - 1 = 4), truncated-power
    form (Hastie et al. 2009, eq. 5.4-5.5) on x rescaled to [0, 1] between the boundary knots;
    values outside the boundary are clamped to it."""
    k = np.asarray(knots, dtype=float)
    lo, hi = k[0], k[-1]
    span = hi - lo if hi > lo else 1.0
    t = (np.clip(np.asarray(x, dtype=float), lo, hi) - lo) / span
    kk = (k - lo) / span

    def d(j: int) -> np.ndarray:
        denom = kk[-1] - kk[j]
        if denom <= 0:
            return np.zeros_like(t)
        return (np.clip(t - kk[j], 0, None) ** 3 - np.clip(t - kk[-1], 0, None) ** 3) / denom

    cols = [t] + [d(j) - d(len(kk) - 2) for j in range(len(kk) - 2)]
    return np.column_stack(cols)


def logistic_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, lam: float = 0.0,
                 *, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Newton-IRLS logistic regression; ``lam`` > 0 adds an L2 penalty lam * ||beta||^2 / 2 on
    every coefficient but the intercept (column 0). ``lam`` = 0 and no weights is
    :func:`nakkhu.flood.validate.logistic`."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if lam == 0:
        return V.logistic(X, y, w, max_iter=max_iter, tol=tol)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pen = np.full(X.shape[1], float(lam))
    pen[0] = 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        v = sw * np.clip(p * (1 - p), 1e-12, None)
        grad = X.T @ (sw * (y - p)) - pen * beta
        hess = (X * v[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=float) @ beta)))


def wald_chi2(beta: np.ndarray, cov: np.ndarray, index: Sequence[int]) -> dict[str, float]:
    idx = np.asarray(index)
    b, c = beta[idx], cov[np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.solve(c, b))
    return {"chi2": chi2, "df": int(len(idx)), "p": float(stats.chi2.sf(chi2, len(idx)))}


def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(loss, weights=w))


def brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(np.average((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2,
                            weights=w))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-mass bins of the predicted probability."""
    table = V.reliability(p, y, bins)
    return float(np.sum(table["n"] * np.abs(table["p_mean"] - table["observed"]))
                 / table["n"].sum())


# --- other metrics -----------------------------------------------------------------------------


def standardised_pauc(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                      max_fpr: float = 0.1) -> float:
    """McClish (1989) standardised partial AUC over FPR in [0, max_fpr]: 0.5 * (1 + (pAUC -
    min) / (max - min)), min = max_fpr^2 / 2, max = max_fpr; the ROC is linear between its
    points (ties as a diagonal step, consistent with AUC ties = 1/2)."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    curve = V.roc_curve(score, y, w)
    fpr = np.r_[0.0, curve["fpr"].to_numpy()]
    tpr = np.r_[0.0, curve["tpr"].to_numpy()]
    stop = np.searchsorted(fpr, max_fpr, side="right")
    xs = np.r_[fpr[:stop], max_fpr]
    ys = np.r_[tpr[:stop], np.interp(max_fpr, fpr, tpr)]
    pauc = float(np.trapezoid(ys, xs))
    lo, hi = max_fpr ** 2 / 2, max_fpr
    return 0.5 * (1 + (pauc - lo) / (hi - lo))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from nakkhu import hazard

    return hazard.spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stratified_auc(score: np.ndarray, y: np.ndarray, band: np.ndarray, bands: Sequence[Any],
                   w: np.ndarray | None = None) -> float:
    """sum_b P_b N_b AUC_b / sum_b P_b N_b over ``bands`` (weighted counts with ``w``)."""
    y = np.asarray(y).astype(bool)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    num = den = 0.0
    for b in bands:
        sel = np.asarray(band) == b
        pos, neg = float(sw[sel & y].sum()), float(sw[sel & ~y].sum())
        if pos == 0 or neg == 0:
            continue
        auc = V.roc_auc(np.asarray(score)[sel], y[sel], None if w is None else sw[sel])
        num += pos * neg * auc
        den += pos * neg
    return num / den if den else float("nan")


def binary_skill(pred: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
                 ) -> dict[str, float]:
    """H (POD), FAR and CSI of a binary rule (design-weighted counts with ``w``)."""
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    tp, fp, fn = (float(sw[pred & y].sum()), float(sw[pred & ~y].sum()),
                  float(sw[~pred & y].sum()))
    return {"tp": tp, "fp": fp, "fn": fn,
            "H": tp / (tp + fn) if tp + fn else float("nan"),
            "FAR": fp / (tp + fp) if tp + fp else float("nan"),
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else float("nan")}
