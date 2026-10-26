"""Step 2: Google <-> Microsoft match graph and per-instance agreement classes.

Two polygons match when IoU >= ``iou_min`` or when at least ``contain_min`` of either polygon's
area lies inside the other (research-labels §3.1). Connected components of the bipartite match
graph give the classes:

==================  ===========================================================
``agree``           one Google, one Microsoft
``split``           one Microsoft, >= 2 Google (Google splits what Microsoft merges)
``merge``           one Google, >= 2 Microsoft (Google merges what Microsoft splits)
``complex``         >= 2 of each (an n:m conflict; treated like split/merge)
``lone_google_hi``  Google without a match, confidence >= its cell's t90
``lone_google_lo``  Google without a match, confidence < t90
``lone_ms``         Microsoft without a match
==================  ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree

CONFLICT_CLASSES = ("split", "merge", "complex")
CLASSES = ("agree", *CONFLICT_CLASSES, "lone_google_hi", "lone_google_lo", "lone_ms")


@dataclass(frozen=True)
class MatchResult:
    """Edges and per-instance classes.

    ``edges``: ``g, m, iou, contain_g, contain_m`` (indices into the input arrays).
    ``google`` / ``microsoft``: per instance ``component, cls, n_matches, max_iou`` and, for
    Microsoft, ``matches_hi`` (matches at least one Google with confidence >= t90).
    """

    edges: pd.DataFrame
    google: pd.DataFrame
    microsoft: pd.DataFrame


def candidate_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i in a, j in b) whose geometries intersect."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    tree = STRtree(b)
    ia, ib = tree.query(a, predicate="intersects")
    return ia.astype(np.int64), ib.astype(np.int64)


def overlap_stats(a: np.ndarray, b: np.ndarray, ia: np.ndarray,
                  ib: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU, share of a inside b and share of b inside a for each pair."""
    ga, gb = a[ia], b[ib]
    inter = shapely.area(shapely.intersection(ga, gb))
    area_a, area_b = shapely.area(ga), shapely.area(gb)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
        contain_a = np.where(area_a > 0, inter / area_a, 0.0)
        contain_b = np.where(area_b > 0, inter / area_b, 0.0)
    return iou, contain_a, contain_b


def _components(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Connected-component labels of an undirected graph (union-find), relabelled 0..k-1."""
    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(u.tolist(), v.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def match(google: np.ndarray, google_conf: np.ndarray, google_t90: np.ndarray,
          microsoft: np.ndarray, *, iou_min: float, contain_min: float) -> MatchResult:
    """Build the match graph and classify every instance (see module docstring)."""
    google = np.asarray(google, dtype=object)
    microsoft = np.asarray(microsoft, dtype=object)
    ng, nm = len(google), len(microsoft)
    gi, mi = candidate_pairs(google, microsoft)
    iou, contain_g, contain_m = overlap_stats(google, microsoft, gi, mi)
    is_edge = (iou >= iou_min) | (contain_g >= contain_min) | (contain_m >= contain_min)
    edges = pd.DataFrame({"g": gi[is_edge], "m": mi[is_edge], "iou": iou[is_edge],
                          "contain_g": contain_g[is_edge], "contain_m": contain_m[is_edge]})
    comp = _components(ng + nm, edges["g"].to_numpy(), edges["m"].to_numpy() + ng)
    comp_g, comp_m = comp[:ng], comp[ng:]
    n_comp = int(comp.max()) + 1 if len(comp) else 0
    size_g = np.bincount(comp_g, minlength=n_comp)
    size_m = np.bincount(comp_m, minlength=n_comp)

    def classify(c: np.ndarray) -> np.ndarray:
        g, m = size_g[c], size_m[c]
        out = np.full(len(c), "", dtype=object)
        out[(g == 1) & (m == 1)] = "agree"
        out[(g >= 2) & (m == 1)] = "split"
        out[(g == 1) & (m >= 2)] = "merge"
        out[(g >= 2) & (m >= 2)] = "complex"
        out[(g >= 1) & (m == 0)] = "lone_google"
        out[(g == 0) & (m >= 1)] = "lone_ms"
        return out

    cls_g = classify(comp_g)
    hi = np.asarray(google_conf, dtype=float) >= np.asarray(google_t90, dtype=float)
    lone = cls_g == "lone_google"
    cls_g[lone & hi] = "lone_google_hi"
    cls_g[lone & ~hi] = "lone_google_lo"
    cls_m = classify(comp_m)

    n_g = np.bincount(edges["g"], minlength=ng)
    n_m = np.bincount(edges["m"], minlength=nm)
    max_iou_g = np.zeros(ng)
    max_iou_m = np.zeros(nm)
    if len(edges):
        np.maximum.at(max_iou_g, edges["g"].to_numpy(), edges["iou"].to_numpy())
        np.maximum.at(max_iou_m, edges["m"].to_numpy(), edges["iou"].to_numpy())
    matches_hi = np.zeros(nm, dtype=bool)
    if len(edges):
        hi_edges = hi[edges["g"].to_numpy()]
        matches_hi[edges["m"].to_numpy()[hi_edges]] = True
    g_frame = pd.DataFrame({"component": comp_g, "cls": cls_g, "n_matches": n_g,
                            "max_iou": max_iou_g, "hi": hi})
    m_frame = pd.DataFrame({"component": comp_m, "cls": cls_m, "n_matches": n_m,
                            "max_iou": max_iou_m, "matches_hi": matches_hi})
    return MatchResult(edges=edges, google=g_frame, microsoft=m_frame)


def class_table(result: MatchResult) -> pd.DataFrame:
    """Counts per class and source (rows of ``match_classes.csv``)."""
    rows = []
    for source, frame in (("google", result.google), ("microsoft", result.microsoft)):
        counts = frame["cls"].value_counts()
        for cls in CLASSES:
            if cls in counts.index:
                rows.append({"source": source, "cls": cls, "instances": int(counts[cls])})
    return pd.DataFrame(rows)
