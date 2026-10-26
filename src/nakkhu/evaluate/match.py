"""Rotated-polygon matching (plan 005 Step 3).

IoU is the polygon IoU of shapely geometries (oriented boxes are 4-corner polygons, so this is
the rotated IoU). Matching is the COCO rule: predictions are visited in descending confidence
(stable, so ties keep input order) and each takes the still-unmatched reference with the highest
IoU at or above the threshold. Because earlier (higher-confidence) predictions are matched first,
the matching restricted to ``conf >= t`` is the prefix of the full matching for every ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely


@dataclass(frozen=True)
class Pairs:
    """Candidate pairs with positive intersection: indices into pred / ref and their IoU."""

    pred: np.ndarray
    ref: np.ndarray
    iou: np.ndarray


def iou_pairs(pred: np.ndarray, ref: np.ndarray) -> Pairs:
    pred = np.asarray(pred, dtype=object)
    ref = np.asarray(ref, dtype=object)
    empty = np.array([], dtype=np.int64)
    if len(pred) == 0 or len(ref) == 0:
        return Pairs(empty, empty, np.array([], dtype=float))
    ip, ir = shapely.STRtree(ref).query(pred, predicate="intersects")   # (query, tree) order
    inter = shapely.area(shapely.intersection(pred[ip], ref[ir]))
    union = shapely.area(pred[ip]) + shapely.area(ref[ir]) - inter
    iou = np.divide(inter, union, out=np.zeros(len(ip)), where=union > 0)
    keep = iou > 0
    return Pairs(ip[keep].astype(np.int64), ir[keep].astype(np.int64), iou[keep])


def match_by_confidence(conf: np.ndarray, n_ref: int, pairs: Pairs,
                        iou_min: float) -> tuple[np.ndarray, np.ndarray]:
    """``(pred_to_ref, ref_to_pred)``: matched index or -1 for each prediction / reference."""
    conf = np.asarray(conf, dtype=float)
    n_pred = len(conf)
    pred_to_ref = np.full(n_pred, -1, dtype=np.int64)
    ref_to_pred = np.full(n_ref, -1, dtype=np.int64)
    ok = pairs.iou >= iou_min
    ip, ir, iou = pairs.pred[ok], pairs.ref[ok], pairs.iou[ok]
    # Candidates of each prediction, best IoU first (ties: lower ref index first).
    order = np.lexsort((ir, -iou, ip))
    ip, ir = ip[order], ir[order]
    starts = np.searchsorted(ip, np.arange(n_pred + 1))
    for p in np.argsort(-conf, kind="stable"):
        for r in ir[starts[p]:starts[p + 1]]:
            if ref_to_pred[r] < 0:
                ref_to_pred[r] = p
                pred_to_ref[p] = r
                break
    return pred_to_ref, ref_to_pred
