"""Detection metrics (plan 005 Step 3).

Precision is counted on the prediction side and recall on the reference side, each over the
objects whose centroid lies in the scored region (see :mod:`nakkhu.evaluate.score`). AP uses the
COCO 101-point interpolated precision envelope (``ap_coco101``); ``ap_voc_all`` is the VOC
all-point area under the envelope, kept for cross-checks. Both accept per-prediction weights so
the block bootstrap can reuse them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

COCO_RECALLS = np.linspace(0.0, 1.0, 101)
AP_IOUS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def prf(tp_pred: float, n_pred: float, tp_ref: float, n_ref: float) -> dict[str, float]:
    precision = tp_pred / n_pred if n_pred > 0 else float("nan")
    recall = tp_ref / n_ref if n_ref > 0 else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif np.isfinite(precision) and np.isfinite(recall):
        f1 = 0.0
    else:
        f1 = float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _curve(conf: np.ndarray, tp: np.ndarray, n_ref: float,
           weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    hit = np.asarray(tp, dtype=float)[order]
    w = np.ones(len(order)) if weights is None else np.asarray(weights, dtype=float)[order]
    ctp = np.cumsum(w * hit)
    cfp = np.cumsum(w * (1.0 - hit))
    recall = ctp / n_ref if n_ref > 0 else np.zeros_like(ctp)
    denom = ctp + cfp
    precision = np.divide(ctp, denom, out=np.zeros_like(ctp), where=denom > 0)
    return recall, precision


def ap_coco101(conf: np.ndarray, tp: np.ndarray, n_ref: float,
               weights: np.ndarray | None = None) -> float:
    """COCO-style AP: mean over recall 0, 0.01, ..., 1 of the interpolated precision."""
    if n_ref <= 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    recall, precision = _curve(conf, tp, n_ref, weights)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, COCO_RECALLS, side="left")
    sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(sampled.mean())


def ap_voc_all(conf: np.ndarray, tp: np.ndarray, n_ref: float,
               weights: np.ndarray | None = None) -> float:
    """VOC 2010+ all-point AP: area under the precision envelope over recall steps."""
    if n_ref <= 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    recall, precision = _curve(conf, tp, n_ref, weights)
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    steps = np.flatnonzero(r[1:] != r[:-1])
    return float(np.sum((r[steps + 1] - r[steps]) * p[steps + 1]))


def f1_max_threshold(conf: np.ndarray, tp: np.ndarray, n_ref: int) -> dict[str, float]:
    """Confidence cut maximising F1 = 2 TP / (n_pred + n_ref) over cuts ``conf >= t``.

    Cuts are only at observed confidences (all predictions tied at ``t`` are kept together).
    Ties in F1 go to the higher threshold (fewer predictions).
    """
    conf = np.asarray(conf, dtype=float)
    if len(conf) == 0 or n_ref <= 0:
        return {"threshold": float("nan"), "f1": float("nan"), "n_pred": 0}
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    ctp = np.cumsum(np.asarray(tp, dtype=float)[order])
    last = np.flatnonzero(np.r_[c[1:] != c[:-1], True])   # last index of each tie group
    n_pred = last + 1
    f1 = 2 * ctp[last] / (n_pred + n_ref)
    best = int(np.flatnonzero(f1 == f1.max())[0])          # first = highest threshold
    return {"threshold": float(c[last[best]]), "f1": float(f1[best]),
            "n_pred": int(n_pred[best])}


def count_errors(pred_counts: npt.ArrayLike, ref_counts: npt.ArrayLike) -> dict[str, float]:
    """Per-unit count MAE and bias (mean of prediction - reference)."""
    diff = np.asarray(pred_counts, dtype=float) - np.asarray(ref_counts, dtype=float)
    if not len(diff):
        return {"mae": float("nan"), "bias": float("nan"), "units": 0}
    return {"mae": float(np.abs(diff).mean()), "bias": float(diff.mean()), "units": len(diff)}
