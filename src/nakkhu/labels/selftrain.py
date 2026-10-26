"""Step 7: one self-training round (L8). Status: code path only; needs the plan 007 L7 model.

Given the L7 seed-0 model's detections on the train tiles (tile pixel corners and conf):

- **add** a detection with ``conf >= add_conf`` that overlaps no footprint label or ignore
  polygon of the tile (``add_conf`` = the conf giving 90 % precision against OSM on val, computed
  in plan 007 with :func:`conf_at_precision`);
- **drop** a footprint label with no supporting detection (no detection with
  ``conf >= support_conf`` overlapping it with IoU > 0).

One round only; it is reported either way.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely


@dataclass(frozen=True)
class SelfTrainResult:
    boxes: np.ndarray          # (n, 4, 2) final label corners
    added: int
    dropped: int


def _polys(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=float).reshape(-1, 4, 2)
    return np.array([shapely.Polygon(c) for c in corners], dtype=object)


def self_train_tile(labels: np.ndarray, detections: np.ndarray, conf: np.ndarray, *,
                    add_conf: float, support_conf: float, overlap_iou: float = 0.0,
                    ignore: object | None = None) -> SelfTrainResult:
    """Apply one self-training round to one tile (all coordinates in tile pixels)."""
    label_polys = _polys(labels)
    det_polys = _polys(detections)
    conf = np.asarray(conf, dtype=float)
    keep = np.ones(len(label_polys), dtype=bool)
    support = det_polys[conf >= support_conf]
    for i, poly in enumerate(label_polys):
        if not len(support) or not shapely.intersects(support, poly).any():
            keep[i] = False
    added = []
    for poly, c in zip(det_polys, conf, strict=True):
        if c < add_conf:
            continue
        if len(label_polys):
            inter = shapely.area(shapely.intersection(label_polys, poly))
            union = shapely.area(shapely.union(label_polys, poly))
            if np.any(inter / np.maximum(union, 1e-12) > overlap_iou):
                continue
        if ignore is not None and shapely.intersects(ignore, poly):
            continue
        added.append(np.asarray(poly.exterior.coords, dtype=float)[:4])
    kept = np.asarray(labels, dtype=float).reshape(-1, 4, 2)[keep]
    boxes = np.concatenate([kept, np.array(added).reshape(-1, 4, 2)])
    return SelfTrainResult(boxes=boxes, added=len(added), dropped=int((~keep).sum()))


def conf_at_precision(conf: npt.ArrayLike, is_tp: npt.ArrayLike,
                      target: float = 0.9) -> float | None:
    """Lowest conf threshold whose precision (TP / detections at or above it) >= ``target``.

    Computed on val detections matched against OSM (plan 007); ``None`` if never reached.
    """
    conf = np.asarray(conf, dtype=float)
    is_tp = np.asarray(is_tp, dtype=bool)
    if not len(conf):
        return None
    thresholds = np.unique(conf)                       # ascending
    n_at = np.array([(conf >= t).sum() for t in thresholds])
    tp_at = np.array([is_tp[conf >= t].sum() for t in thresholds])
    ok = np.flatnonzero(tp_at / n_at >= target)
    return float(thresholds[ok[0]]) if len(ok) else None


def apply_to_arm(l7_dir: Path, out_dir: Path, detections: pd.DataFrame, *, add_conf: float,
                 support_conf: float, overlap_iou: float, tile_px: int) -> pd.DataFrame:
    """Write L8 = L7 + one round into ``out_dir`` (labels, ignore masks and tiles.txt copied).

    ``detections``: plan 003/007 ``detections_px.parquet`` columns ``tile_id, conf, x1..y4`` of
    the L7 seed-0 model on the train tiles. Returns per-tile added/dropped counts.
    """
    from nakkhu.labels.obb import read_yolo_obb, write_yolo_obb

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(l7_dir, out_dir)
    corners_cols = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    by_tile = dict(tuple(detections.groupby("tile_id")))
    rows = []
    for tile_id in (l7_dir / "tiles.txt").read_text(encoding="utf-8").split():
        _, labels = read_yolo_obb(l7_dir / "labels" / f"{tile_id}.txt", tile_px)
        det = by_tile.get(tile_id)
        corners = (det[corners_cols].to_numpy(dtype=float).reshape(-1, 4, 2)
                   if det is not None else np.zeros((0, 4, 2)))
        conf = det["conf"].to_numpy(dtype=float) if det is not None else np.zeros(0)
        result = self_train_tile(labels, corners, conf, add_conf=add_conf,
                                 support_conf=support_conf, overlap_iou=overlap_iou)
        write_yolo_obb(out_dir / "labels" / f"{tile_id}.txt", result.boxes, tile_px)
        rows.append({"tile_id": tile_id, "added": result.added, "dropped": result.dropped,
                     "labels": len(result.boxes)})
    stats = pd.DataFrame(rows)
    stats.to_parquet(out_dir / "selftrain_stats.parquet")
    return stats
