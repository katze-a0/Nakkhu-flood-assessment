"""Block bootstrap (plan 005 Step 4).

Blocks (3 x 3 tiles, ``configs/splits/split_v1.yaml``) are resampled with replacement; each
replicate is a vector of block multiplicities. Additive per-block columns (TP, counts, absolute
count errors) are summed with those weights and the statistic is recomputed (a ratio estimator,
not a mean of per-block ratios). AP is recomputed with per-prediction weights equal to the
multiplicity of the prediction's block. Paired differences use the same replicates for both
models. CIs are percentile intervals.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nakkhu.evaluate.metrics import ap_coco101


def multiplicities(n_blocks: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_blocks)`` counts of each block in each resample."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    out = np.zeros((n_boot, n_blocks), dtype=np.int64)
    rows = np.repeat(np.arange(n_boot), n_blocks)
    np.add.at(out, (rows, draws.ravel()), 1)
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


Statistic = Callable[[dict[str, np.ndarray]], np.ndarray]


def additive(columns: dict[str, np.ndarray], weights: np.ndarray,
             statistic: Statistic) -> np.ndarray:
    """Statistic of weighted column sums for every replicate (``weights``: ``(B, K)``)."""
    sums = {k: weights @ np.asarray(v, dtype=float) for k, v in columns.items()}
    return np.asarray(statistic(sums), dtype=float)


def f1_stat(tp_pred: str, n_pred: str, tp_ref: str, n_ref: str) -> Statistic:
    def stat(s: dict[str, np.ndarray]) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = s[tp_pred] / s[n_pred]
            r = s[tp_ref] / s[n_ref]
            return np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return stat


def ratio_stat(num: str, den: str) -> Statistic:
    def stat(s: dict[str, np.ndarray]) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return s[num] / s[den]
    return stat


def ap_replicates(conf: np.ndarray, tp: np.ndarray, pred_block: np.ndarray,
                  n_ref_block: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AP (COCO 101-point) per replicate; ``pred_block`` indexes the columns of ``weights``."""
    out = np.empty(len(weights))
    for b, w in enumerate(weights):
        out[b] = ap_coco101(conf, tp, float(w @ n_ref_block), weights=w[pred_block])
    return out


def summary(point: float, samples: np.ndarray, level: float = 0.95) -> dict[str, float]:
    lo, hi = percentile_ci(samples, level)
    return {"value": float(point), "ci_low": lo, "ci_high": hi,
            "boot_sd": float(np.nanstd(samples, ddof=1)) if len(samples) > 1 else float("nan")}
