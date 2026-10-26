"""HAND tiers, agreement statistics, building attribution and event calibration (plan 008).

Tiers are water-stage equivalents (plan 008 Step 8): with edges ``[0, 3, 6, 9, 12]`` the classes
are *on drainage* (HAND <= 0), (0, 3], (3, 6], (6, 9], (9, 12] and > 12 m, in manuscript order.

Event calibration (H6) maps HAND to P(affected | HAND) with a logistic or an isotonic
(non-increasing) model, fitted out of fold over spatial blocks. It needs the plan 009 flood
reference and is not run by plan 008. Every H6 function takes optional design weights ``w``
(positional, after the data; ``None`` = unweighted, the same arithmetic as before), used for
the audited reference R1-aud whose strata are sampled at unequal rates. The pinned keyword
defaults of :func:`fit_logistic` (prereg ``validation.h6.logistic_defaults``) are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EDGES = (0.0, 3.0, 6.0, 9.0, 12.0)


# --- tiers and agreement --------------------------------------------------------------------


def tier_labels(edges: Sequence[float] = DEFAULT_EDGES) -> list[str]:
    def fmt(value: float) -> str:
        return f"{value:g}"

    labels = ["on_drainage"]
    labels += [f"{fmt(lo)}-{fmt(hi)}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    labels.append(f">{fmt(edges[-1])}")
    return labels


def tier_index(hand: np.ndarray, edges: Sequence[float] = DEFAULT_EDGES) -> np.ndarray:
    """Ordinal tier per value: 0 = on drainage (<= edges[0]), ..., len(edges) = above the last
    edge; -1 for NaN. Upper bounds are inclusive: 3.0 m is in (0, 3]."""
    values = np.asarray(hand, dtype=np.float64)
    tiers = np.searchsorted(np.asarray(edges, dtype=np.float64), values, side="left")
    tiers = tiers.astype(np.int8)
    tiers[np.isnan(values)] = -1
    return tiers


def tier_counts(tiers: np.ndarray, n_tiers: int) -> np.ndarray:
    valid = tiers[tiers >= 0]
    return np.bincount(valid.astype(np.int64), minlength=n_tiers)[:n_tiers]


def transition_matrix(a: np.ndarray, b: np.ndarray, n_tiers: int) -> np.ndarray:
    """Counts ``m[i, j]`` of items in tier i under ``a`` and tier j under ``b`` (both valid)."""
    ok = (a >= 0) & (b >= 0)
    flat = a[ok].astype(np.int64) * n_tiers + b[ok].astype(np.int64)
    return np.bincount(flat, minlength=n_tiers * n_tiers).reshape(n_tiers, n_tiers)


def weighted_kappa(a: np.ndarray, b: np.ndarray, n_tiers: int, weights: str = "linear") -> float:
    """Cohen's weighted kappa of two ordinal labelings (``-1`` = missing, dropped pairwise)."""
    observed = transition_matrix(a, b, n_tiers).astype(np.float64)
    total = observed.sum()
    if total == 0:
        return float("nan")
    i, j = np.indices((n_tiers, n_tiers))
    if weights == "linear":
        w = np.abs(i - j) / (n_tiers - 1)
    elif weights == "quadratic":
        w = ((i - j) / (n_tiers - 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    denominator = (w * expected).sum()
    if denominator == 0:
        return 1.0 if (w * observed).sum() == 0 else float("nan")
    return float(1.0 - (w * observed).sum() / denominator)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_surfaces(a: np.ndarray, b: np.ndarray, tolerance_m: float,
                     edges: Sequence[float] = DEFAULT_EDGES) -> dict[str, float]:
    """Agreement of two HAND samples (same cells): |difference| statistics and tier kappa."""
    ok = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[ok].astype(np.float64) - b[ok].astype(np.float64))
    n_tiers = len(edges) + 1
    ta, tb = tier_index(a[ok], edges), tier_index(b[ok], edges)
    return {
        "n_cells": int(ok.sum()),
        "n_missing": int((~ok).sum()),
        "max_abs_diff_m": float(diff.max()) if diff.size else float("nan"),
        "mean_abs_diff_m": float(diff.mean()) if diff.size else float("nan"),
        "median_abs_diff_m": float(np.median(diff)) if diff.size else float("nan"),
        "share_abs_diff_gt_tol": float((diff > tolerance_m).mean()) if diff.size else float("nan"),
        "kappa_linear": weighted_kappa(ta, tb, n_tiers),
        "share_same_tier": float((ta == tb).mean()) if diff.size else float("nan"),
        "spearman": spearman(a[ok], b[ok]),
    }


# --- building attribution (H5) --------------------------------------------------------------


@dataclass(frozen=True)
class CellGroups:
    """Flat grid-cell indices per building, concatenated (``offsets[k]:offsets[k+1]``)."""

    cells: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_lists(cls, lists: Sequence[Sequence[int]]) -> CellGroups:
        sizes = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        offsets = np.zeros(len(lists) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        cells = (np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
                 if len(lists) and offsets[-1] else np.zeros(0, dtype=np.int64))
        return cls(cells=cells, offsets=offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def minimum(self, values: np.ndarray) -> np.ndarray:
        """NaN-aware minimum of ``values[cells]`` per building (NaN if no finite cell)."""
        out = np.full(len(self), np.nan, dtype=np.float64)
        sizes = np.diff(self.offsets)
        nonempty = sizes > 0
        if not nonempty.any():
            return out
        sampled = values[self.cells].astype(np.float64)
        sampled[np.isnan(sampled)] = np.inf
        mins = np.minimum.reduceat(sampled, self.offsets[:-1][nonempty])
        mins[np.isinf(mins)] = np.nan
        out[nonempty] = mins
        return out


def cell_groups(geometries, index_raster: str | Path) -> CellGroups:
    """Cells touched by each geometry (exactextract ``cell_id``: any coverage counts).

    ``index_raster`` is any raster on the analysis grid; ``cell_id`` is the row-major flat index.
    """
    import geopandas as gpd
    from exactextract import exact_extract

    frame = gpd.GeoDataFrame({"_k": np.arange(len(geometries))}, geometry=list(geometries),
                             crs=getattr(geometries, "crs", None))
    result = exact_extract(str(index_raster), frame, ["cell_id"], output="pandas",
                           include_cols=["_k"])
    result = result.sort_values("_k")
    lists = [np.asarray(v, dtype=np.int64) for v in result["cell_id"]]
    return CellGroups.from_lists(lists)


def centroid_cells(geometries, grid) -> np.ndarray:
    """Flat index of the cell containing each centroid; -1 outside the grid."""
    cent = geometries.centroid
    rows, cols = grid.rowcol(cent.x.to_numpy(), cent.y.to_numpy())
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    return np.where(inside, rows * grid.width + cols, -1)


def sample_centroids(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(idx.shape, np.nan, dtype=np.float64)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def confidence_cut(frame, thresholds_csv: str | Path,
                   column: str = "confidence_threshold_90%_precision") -> tuple[np.ndarray,
                                                                               np.ndarray]:
    """Google Open Buildings v3 precision cut: ``(keep, threshold)`` per polygon.

    The threshold is that of the S2 level-4 cell containing the polygon's published
    ``longitude``/``latitude`` (Google's ``score_thresholds_s2_level_4.csv``); keep
    ``confidence >= threshold``. Same rule as ``nakkhu.labels.harmonise`` (plan 004). A polygon
    without a cell threshold raises instead of using a fallback.
    """
    import geopandas as gpd

    table = pd.read_csv(thresholds_csv)
    cells = gpd.GeoDataFrame({"t": table[column].astype(float)},
                             geometry=gpd.GeoSeries.from_wkt(table["geometry"]), crs="EPSG:4326")
    points = gpd.GeoDataFrame({"_row": np.arange(len(frame))},
                              geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
                              crs="EPSG:4326")
    joined = gpd.sjoin(points, cells, how="left", predicate="within")
    joined = joined.sort_values("_row").drop_duplicates("_row", keep="first")
    threshold = joined["t"].to_numpy(dtype=np.float64)
    if np.isnan(threshold).any():
        raise ValueError(f"{int(np.isnan(threshold).sum())} polygons have no S2-cell threshold")
    keep = frame["confidence"].to_numpy(dtype=np.float64) >= threshold
    return keep, threshold


# --- event calibration (H6; run in plan 009) ------------------------------------------------


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *,
                 l2: float = 1e-6, max_iter: int = 100,
                 tol: float = 1e-10) -> tuple[float, float]:
    """``(intercept, slope)`` of P(y=1 | x) = 1 / (1 + exp(-(a + b x))) by Newton-IRLS; with
    design weights ``w`` the (pseudo-)log-likelihood is weighted."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        v = np.clip(p * (1 - p), 1e-12, None)
        resid = y - p
        if w is not None:
            v, resid = w * v, w * resid
        gradient = design.T @ resid - l2 * beta
        hessian = (design * v[:, None]).T @ design + l2 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < tol:
            break
    return float(beta[0]), float(beta[1])


def predict_logistic(params: tuple[float, float], x: np.ndarray) -> np.ndarray:
    a, b = params
    return 1.0 / (1.0 + np.exp(-(a + b * np.asarray(x, dtype=np.float64))))


def fit_isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Non-increasing step function of P(y=1 | x) by pool-adjacent-violators (each item
    weighted by ``w`` when given).

    Returns knots ``(x_sorted_unique, p)``; :func:`predict_isotonic` interpolates linearly
    between knots and holds the end values outside them.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    ux, inverse = np.unique(xs, return_inverse=True)
    if w is None:
        sums = np.bincount(inverse, weights=ys)
        weights = np.bincount(inverse).astype(np.float64)
    else:
        ws = np.asarray(w, dtype=np.float64)[order]
        sums = np.bincount(inverse, weights=ws * ys)
        weights = np.bincount(inverse, weights=ws)
    # PAVA for a non-increasing fit = PAVA (non-decreasing) on the negated means.
    level, weight, size = [], [], []
    for s, wt in zip(-sums / weights, weights, strict=True):
        level.append(s)
        weight.append(wt)
        size.append(1)
        while len(level) > 1 and level[-2] > level[-1]:
            w_new = weight[-2] + weight[-1]
            l_new = (level[-2] * weight[-2] + level[-1] * weight[-1]) / w_new
            n_new = size[-2] + size[-1]
            level[-2:], weight[-2:], size[-2:] = [l_new], [w_new], [n_new]
    p = np.repeat(-np.asarray(level), size)
    return ux, p


def predict_isotonic(knots: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    ux, p = knots
    return np.interp(np.asarray(x, dtype=np.float64), ux, p)


def youden_threshold(x: np.ndarray, y: np.ndarray,
                     w: np.ndarray | None = None) -> tuple[float, float]:
    """HAND cut ``t`` maximising Youden's J for the rule "affected if HAND <= t"; ``(t, J)``.
    With design weights ``w`` the true- and false-positive rates are weighted."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    if int(y.sum()) == 0 or int((~y).sum()) == 0:
        raise ValueError("Youden's J needs both classes")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    if w is None:
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        tp = np.cumsum(ys)
        fp = np.cumsum(~ys)
    else:
        sw = np.asarray(w, dtype=np.float64)
        n_pos, n_neg = float(sw[y].sum()), float(sw[~y].sum())
        ws = sw[order]
        tp = np.cumsum(ws * ys)
        fp = np.cumsum(ws * ~ys)
    last = np.r_[xs[1:] != xs[:-1], True]  # evaluate once per distinct cut
    j = tp[last] / n_pos - fp[last] / n_neg
    k = int(np.argmax(j))
    return float(xs[last][k]), float(j[k])


def brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    sq = (np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2
    return float(np.mean(sq)) if w is None else float(np.average(sq, weights=w))


LIKELIHOOD_EDGES = (0.1, 0.25, 0.5)
LIKELIHOOD_LABELS = ("low", "moderate", "high", "very_high")


def likelihood_class(p: np.ndarray, edges: Sequence[float] = LIKELIHOOD_EDGES) -> np.ndarray:
    """0 = low (< 0.1), 1 = moderate [0.1, 0.25), 2 = high [0.25, 0.5), 3 = very high (>= 0.5)."""
    return np.searchsorted(np.asarray(edges), np.asarray(p, dtype=np.float64), side="right")


def calibrate_out_of_fold(x: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                          method: str = "logistic",
                          w: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold P(affected | HAND): fit on all other spatial blocks, predict each block
    (design-weighted fits when ``w`` is given)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    blocks = np.asarray(blocks)
    out = np.full(x.shape, np.nan)
    for block in np.unique(blocks):
        test = blocks == block
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        wt = None if w is None else np.asarray(w, dtype=np.float64)[train]
        if method == "logistic":
            out[test] = predict_logistic(fit_logistic(x[train], y[train], wt), x[test])
        elif method == "isotonic":
            out[test] = predict_isotonic(fit_isotonic(x[train], y[train], wt), x[test])
        else:
            raise ValueError(f"unknown calibration method {method!r}")
    return out
