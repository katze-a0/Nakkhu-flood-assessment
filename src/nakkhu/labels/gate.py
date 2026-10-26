"""Step 6: tile quality gate (L7). No learning: the score is a fixed, pre-registered formula.

Per-tile features (all computed without any reference data):

- ``peak_ratio``: registration peak ratio (Step 3);
- ``mean_iou``: mean, over the Google and Microsoft instances intersecting the tile, of each
  instance's best cross-source IoU (0 for an unmatched instance);
- ``conflict_frac``: share of those instances in a split / merge / complex component;
- ``sam2_mean_iou``: mean SAM2 mask/footprint IoU of the tile's prompts (Step 4);
- ``valid_frac``: share of valid image pixels.

Each feature is turned into a percentile rank in (0, 1] over all scored tiles (average rank for
ties; ``lower`` features are negated first). A missing feature (no footprints, no SAM2 output)
gets the worst rank, 0. The gate score is the unweighted mean of the ranks of the configured
features. Admission keeps the top ``admit_fraction`` of the train tiles by score; the fraction
is chosen on val blocks only (:func:`choose_admit_fraction` refuses test tiles).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from nakkhu.labels.match import CONFLICT_CLASSES

DIRECTIONS = ("higher", "lower")


class GateError(ValueError):
    """The gate cannot be scored or a set-point would read test tiles."""


def match_features(tile_ids: list[str], tile_google: list[np.ndarray],
                   tile_microsoft: list[np.ndarray], google: pd.DataFrame,
                   microsoft: pd.DataFrame) -> pd.DataFrame:
    g_iou, m_iou = google["max_iou"].to_numpy(), microsoft["max_iou"].to_numpy()
    g_conf = np.isin(google["cls"].to_numpy(), CONFLICT_CLASSES)
    m_conf = np.isin(microsoft["cls"].to_numpy(), CONFLICT_CLASSES)
    rows = []
    for tile_id, gi, mi in zip(tile_ids, tile_google, tile_microsoft, strict=True):
        n = len(gi) + len(mi)
        if n == 0:
            rows.append({"tile_id": tile_id, "n_instances": 0, "mean_iou": np.nan,
                         "conflict_frac": np.nan})
            continue
        rows.append({"tile_id": tile_id, "n_instances": n,
                     "mean_iou": float((g_iou[gi].sum() + m_iou[mi].sum()) / n),
                     "conflict_frac": float((g_conf[gi].sum() + m_conf[mi].sum()) / n)})
    return pd.DataFrame(rows)


def percentile_rank(values: np.ndarray, direction: str) -> np.ndarray:
    """Average-tie percentile rank in (0, 1]; NaN -> 0 (worst)."""
    if direction not in DIRECTIONS:
        raise GateError(f"direction must be one of {DIRECTIONS}")
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    finite = np.isfinite(values)
    if not finite.any():
        return out
    signed = values[finite] if direction == "higher" else -values[finite]
    ranks = pd.Series(signed).rank(method="average").to_numpy()
    out[finite] = ranks / finite.sum()
    return out


def gate_scores(features: pd.DataFrame, spec: Mapping[str, str]) -> pd.DataFrame:
    """Add ``rank_<feature>`` columns and ``gate_score`` (mean of the ranks)."""
    missing = [name for name in spec if name not in features.columns]
    if missing:
        raise GateError(f"missing gate features: {missing}")
    out = features.copy()
    rank_cols = []
    for name, direction in spec.items():
        col = f"rank_{name}"
        out[col] = percentile_rank(out[name].to_numpy(), direction)
        rank_cols.append(col)
    out["gate_score"] = out[rank_cols].mean(axis=1)
    return out


def admit(scores: pd.DataFrame, train_ids: Iterable[str], fraction: float) -> set[str]:
    """Top ``fraction`` of the train tiles by gate score (ties broken by tile id)."""
    if not 0 < fraction <= 1:
        raise GateError("admit fraction must be in (0, 1]")
    train = scores.loc[scores["tile_id"].isin(set(train_ids))]
    train = train.sort_values(["gate_score", "tile_id"], ascending=[False, True])
    n = int(np.ceil(fraction * len(train)))
    return set(train["tile_id"].iloc[:n])


def choose_admit_fraction(curve: pd.DataFrame, val_ids: Iterable[str],
                          test_ids: Iterable[str]) -> float:
    """Set-point = the admit fraction with the highest val F1 (ties: the larger fraction).

    ``curve`` has ``fraction, f1`` and ``tile_ids`` (the val tiles the F1 was scored on). It
    must be scored on val tiles only: any test tile id raises :class:`GateError`.
    """
    val, test = set(val_ids), set(test_ids)
    if val & test:
        raise GateError("val and test tile sets overlap")
    for ids in curve.get("tile_ids", pd.Series(dtype=object)):
        used = set(ids)
        if used & test:
            raise GateError("the set-point curve was scored on test tiles")
        if not used <= val:
            raise GateError("the set-point curve uses tiles outside the val blocks")
    best = curve.sort_values(["f1", "fraction"], ascending=[False, False]).iloc[0]
    return float(best["fraction"])
