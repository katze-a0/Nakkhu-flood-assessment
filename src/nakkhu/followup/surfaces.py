"""Label-free derived predictors of prereg_v9_a3 ``definitions`` (plan 014 Steps 1-2).

Every function here reads DEM / routing rasters and geometry only: no flood label, change map
or audit answer. Rasters are numpy arrays on a :class:`~nakkhu.hydro.dem.GridSpec`; D8 paths
follow :func:`nakkhu.hydro.route.downstream_index` (ESRI codes, -1 = pit / flat / grid exit),
exactly as the sweep's ``compute_hand``.

- :func:`walk_downstream`: first cell of a mask on the D8 path of each start cell (the start
  counts) and the path length to it, vectorised over the starts.
- :func:`main_mask`: M = cells touched by the main lines with contributing area >= 1 km2.
- :func:`hand_above`: ``max(0, z_ref(c) - z_ref(first))`` (``hand_any_t``, ``hand_mr_d8`` and
  the team recipe all use it); NaN where the path never reaches the drainage.
- :func:`rem_channel_min` / :func:`hand_mr_rem`: the relative elevation model.
- :func:`horn_slope_deg`, :func:`zt_curvatures`, :func:`tpi`, :func:`twi`: P1-6 terrain.
- :func:`team_recipe`: the team's ArcGIS chain on one DEM, routing clipped to the basin.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree

from nakkhu.hydro import condition as cond
from nakkhu.hydro import route
from nakkhu.hydro.dem import GridSpec

SQRT2 = float(np.sqrt(2.0))


def walk_downstream(ds: np.ndarray, target: np.ndarray, start: np.ndarray, width: int,
                    res: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(first, length_m)`` for each flat ``start`` index: the first cell on its D8 path with
    ``target`` True (the start itself counts) and the path length to it (``res`` per straight
    step, ``res * sqrt(2)`` per diagonal step). ``first`` is -1 and the length NaN when the path
    ends (pit, flat, grid exit) first or the start is -1."""
    ds = np.asarray(ds, dtype=np.int64).ravel()
    target = np.asarray(target, dtype=bool).ravel()
    cur = np.asarray(start, dtype=np.int64).copy()
    first = np.full(cur.shape, -1, dtype=np.int64)
    length = np.zeros(cur.shape, dtype=np.float64)
    active = cur >= 0
    for _ in range(ds.size + 1):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        cells = cur[idx]
        hit = target[cells]
        first[idx[hit]] = cells[hit]
        active[idx[hit]] = False
        idx, cells = idx[~hit], cells[~hit]
        nxt = ds[cells]
        stop = nxt < 0
        active[idx[stop]] = False
        idx, cells, nxt = idx[~stop], cells[~stop], nxt[~stop]
        diagonal = ((nxt // width) != (cells // width)) & ((nxt % width) != (cells % width))
        length[idx] += np.where(diagonal, res * SQRT2, res)
        cur[idx] = nxt
    else:                                  # pragma: no cover - D8 of a conditioned DEM is acyclic
        raise RuntimeError("D8 path longer than the grid (cycle in the flow directions)")
    length[first < 0] = np.nan
    return first, length


def main_mask(lines: Any, grid: GridSpec, acc: np.ndarray, min_area_km2: float = 1.0
              ) -> np.ndarray:
    """M: cells touched (``all_touched``) by ``lines`` (a GeoDataFrame) whose D8 contributing
    area (``acc`` cells x cell area) is >= ``min_area_km2``."""
    touched = cond.burn_mask(lines, grid, all_touched=True)
    return touched & (np.asarray(acc) * grid.cell_area_m2 >= min_area_km2 * 1e6)


def stream_mask(acc: np.ndarray, grid: GridSpec, threshold_km2: float) -> np.ndarray:
    """Stream cells at ``threshold_km2`` exactly as the sweep (``hydro.hand.stream_mask``)."""
    from nakkhu.hydro.hand import stream_mask as sweep_streams

    return sweep_streams(np.asarray(acc), grid, threshold_km2)


def hand_above(z_ref: np.ndarray, cells: np.ndarray, first: np.ndarray) -> np.ndarray:
    """``max(0, z_ref(c) - z_ref(first(c)))`` per cell; NaN where ``cells`` or ``first`` is -1
    or either elevation is NaN (the sweep clamps negative HAND to 0 the same way)."""
    z = np.asarray(z_ref, dtype=np.float64).ravel()
    cells = np.asarray(cells, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    out = np.full(cells.shape, np.nan)
    ok = (cells >= 0) & (first >= 0)
    with np.errstate(invalid="ignore"):
        out[ok] = np.maximum(z[cells[ok]] - z[first[ok]], 0.0)
    return out


def cell_centres(grid: GridSpec, cells: np.ndarray) -> np.ndarray:
    """``(n, 2)`` map coordinates of the centres of flat cell indices."""
    cells = np.asarray(cells, dtype=np.int64)
    rows, cols = np.divmod(cells, grid.width)
    t = grid.transform
    return np.column_stack([t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e])


def point_cells(grid: GridSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Flat index of the cell holding each point (-1 outside the grid), as
    :func:`nakkhu.hazard.centroid_cells`."""
    rows, cols = grid.rowcol(np.asarray(x), np.asarray(y))
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    return np.where(inside, rows * grid.width + cols, -1)


def sample(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(cells), np.nan)
    ok = np.asarray(cells) >= 0
    out[ok] = np.asarray(values, dtype=np.float64).ravel()[np.asarray(cells)[ok]]
    return out


def nearest_distance(grid: GridSpec, mask: np.ndarray, x: np.ndarray, y: np.ndarray
                     ) -> np.ndarray:
    """Euclidean distance (m) from each point to the nearest centre of a ``mask`` cell."""
    cells = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    if cells.size == 0:
        return np.full(len(x), np.nan)
    tree = KDTree(cell_centres(grid, cells))
    dist, _ = tree.query(np.column_stack([x, y]))
    return dist


def rem_channel_min(grid: GridSpec, main: np.ndarray, z_raw: np.ndarray,
                    radius_m: float = 60.0) -> tuple[np.ndarray, np.ndarray, KDTree]:
    """``(m_cells, zmin, tree)``: the M cells, for each the minimum ``z_raw`` over the M cells
    whose centres lie within ``radius_m`` of its centre (inclusive), and a KD-tree of the M
    cell centres."""
    m_cells = np.flatnonzero(np.asarray(main, dtype=bool).ravel())
    centres = cell_centres(grid, m_cells)
    tree = KDTree(centres)
    z = np.asarray(z_raw, dtype=np.float64).ravel()[m_cells]
    groups = tree.query_ball_point(centres, r=radius_m + 1e-6)
    zmin = np.array([np.nanmin(z[g]) for g in groups], dtype=np.float64)
    return m_cells, zmin, tree


def hand_mr_rem(grid: GridSpec, main: np.ndarray, z_raw: np.ndarray, cells: np.ndarray,
                radius_m: float = 60.0) -> np.ndarray:
    """``max(0, z_raw(c) - z_ch(c))``: z_ch = minimum z_raw over the M cells within
    ``radius_m`` of the M cell nearest (cell centres) to c. NaN for cells -1."""
    cells = np.asarray(cells, dtype=np.int64)
    out = np.full(cells.shape, np.nan)
    ok = cells >= 0
    if not ok.any() or not np.asarray(main).any():
        return out
    _, zmin, tree = rem_channel_min(grid, main, z_raw, radius_m)
    _, nearest = tree.query(cell_centres(grid, cells[ok]))
    z = np.asarray(z_raw, dtype=np.float64).ravel()[cells[ok]]
    with np.errstate(invalid="ignore"):
        out[ok] = np.maximum(z - zmin[nearest], 0.0)
    return out


# --- terrain features (P1-6), on the raw DEM -------------------------------------------------


def _windows(z: np.ndarray) -> dict[int, np.ndarray]:
    """3x3 neighbours z1..z9 (z1 = NW, z2 = N, z3 = NE, z4 = W, z5 = centre, z6 = E, z7 = SW,
    z8 = S, z9 = SE), edges padded by replication."""
    p = np.pad(np.asarray(z, dtype=np.float64), 1, mode="edge")
    h, w = np.shape(z)
    return {1: p[:-2, :-2], 2: p[:-2, 1:-1], 3: p[:-2, 2:], 4: p[1:-1, :-2], 5: p[1:-1, 1:-1],
            6: p[1:-1, 2:], 7: p[2:, :-2], 8: p[2:, 1:-1], 9: p[2:, 2:]} if h and w else {}


def horn_slope_deg(z: np.ndarray, res: float) -> np.ndarray:
    """Slope (degrees) by Horn's (1981) 3x3 finite difference."""
    k = _windows(z)
    dzdx = ((k[3] + 2 * k[6] + k[9]) - (k[1] + 2 * k[4] + k[7])) / (8.0 * res)
    dzdy = ((k[7] + 2 * k[8] + k[9]) - (k[1] + 2 * k[2] + k[3])) / (8.0 * res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def zt_curvatures(z: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """``(plan, profile)`` curvature (1/m) of Zevenbergen and Thorne (1987), with the signs of
    Moore et al. (1991): profile = -2(DG^2 + EH^2 + FGH)/(G^2 + H^2), plan = 2(DH^2 + EG^2 -
    FGH)/(G^2 + H^2); 0 where the surface is flat (G = H = 0)."""
    k = _windows(z)
    L = float(res)
    D = ((k[4] + k[6]) / 2.0 - k[5]) / L ** 2
    E = ((k[2] + k[8]) / 2.0 - k[5]) / L ** 2
    F = (-k[1] + k[3] + k[7] - k[9]) / (4.0 * L ** 2)
    G = (-k[4] + k[6]) / (2.0 * L)
    H = (k[2] - k[8]) / (2.0 * L)
    gh = G ** 2 + H ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(gh > 0, -2.0 * (D * G ** 2 + E * H ** 2 + F * G * H) / gh, 0.0)
        plan = np.where(gh > 0, 2.0 * (D * H ** 2 + E * G ** 2 - F * G * H) / gh, 0.0)
    return plan, profile


def tpi(z: np.ndarray, res: float, radius_m: float) -> np.ndarray:
    """Topographic position index: z minus the mean z of the cells whose centres lie within
    ``radius_m`` (circular window, the cell itself included; NaN cells ignored)."""
    r = int(np.floor(radius_m / res))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = ((xx * res) ** 2 + (yy * res) ** 2 <= radius_m ** 2 + 1e-9).astype(np.float64)
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z)
    total = ndimage.convolve(np.where(ok, z, 0.0), kernel, mode="constant", cval=0.0)
    count = ndimage.convolve(ok.astype(np.float64), kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return z - total / count


def twi(acc: np.ndarray, grid: GridSpec, slope_deg: np.ndarray,
        min_slope_deg: float = 0.1) -> np.ndarray:
    """ln(a / tan(max(slope, min_slope))), a = contributing area per unit contour width =
    (accumulated cells x cell area) / cell size (m)."""
    a = np.asarray(acc, dtype=np.float64) * grid.cell_area_m2 / grid.res
    slope = np.radians(np.maximum(np.asarray(slope_deg, dtype=np.float64), min_slope_deg))
    return np.log(a / np.tan(slope))


def count_within(x: np.ndarray, y: np.ndarray, radius_m: float) -> np.ndarray:
    """Number of other points whose distance is <= ``radius_m`` (the point itself excluded)."""
    tree = KDTree(np.column_stack([x, y]))
    return np.array([len(g) - 1 for g in tree.query_ball_point(np.column_stack([x, y]),
                                                                r=radius_m)], dtype=np.int64)


# --- the team recipe (P1-10) ------------------------------------------------------------------


def team_recipe(z_raw: np.ndarray, grid: GridSpec, basin: np.ndarray, conditioning: str, *,
                burn: np.ndarray | None, burn_depth_m: float, max_dist_cells: int,
                stream_area_km2: float, digitized: np.ndarray, outlet: int | None = None,
                wbt: Any = None, workdir: str | Path | None = None) -> dict[str, np.ndarray]:
    """The team's ArcGIS chain on ``z_raw`` with routing clipped to ``basin`` (cells outside
    are no-data): ``fill`` (pysheds fill, no burn) or ``breach_burn`` (``burn`` lowered by
    ``burn_depth_m``, least-cost breach, burn undone in the reference). Returns the full-grid
    ``z_ref``, ``fdir``, ``acc``, the ``streams`` (contributing area >= ``stream_area_km2``
    inside the basin) and ``hand_acc`` / ``hand_dig`` (drainage = the streams / the
    ``digitized`` cells inside the basin), NaN outside the basin or where the path leaves it
    first.

    ``outlet`` (flat index of a basin cell) is the basin's outlet: its routing elevation is set
    1 m below every other routing elevation of the basin before conditioning, so the fill /
    breach drains the whole basin (and resolves its flats) to it, as ArcGIS lets the outlet cell
    drain off the clipped raster; like a burned cell, its HAND reference is restored to
    ``max(conditioned, original)``."""
    basin = np.asarray(basin, dtype=bool)
    z = np.where(basin, np.asarray(z_raw, dtype=np.float64), np.nan)
    if conditioning == "fill":
        mask = np.zeros(grid.shape, dtype=bool)
        source = z.copy()
    elif conditioning == "breach_burn":
        mask = np.asarray(burn, dtype=bool) & basin
        source = cond.burn(z, mask, burn_depth_m)
    else:
        raise ValueError(f"unknown team conditioning {conditioning!r}")
    restore = mask.copy()
    if outlet is not None and outlet >= 0:
        source.ravel()[outlet] = np.nanmin(source[basin]) - 1.0
        restore.ravel()[outlet] = True
    if conditioning == "fill":
        routing = cond.fill(source, grid)
    else:
        routing = cond.breach(source, grid, max_dist_cells=max_dist_cells, wbt=wbt,
                              workdir=workdir)
    routing = np.where(basin, routing, np.nan)
    z_ref = cond.reference_elevation(routing, restore if restore.any() else None, z)
    fdir = route.flowdir(routing, grid)
    acc = route.accumulation(fdir, grid)
    ds = route.downstream_index(fdir)
    streams = stream_mask(acc, grid, stream_area_km2) & basin
    dig = np.asarray(digitized, dtype=bool) & basin
    cells = np.flatnonzero(basin.ravel())
    out: dict[str, np.ndarray] = {"z_ref": z_ref, "fdir": fdir, "acc": acc, "streams": streams}
    for name, drain in (("hand_acc", streams), ("hand_dig", dig)):
        first, _ = walk_downstream(ds, drain, cells, grid.width)
        full = np.full(grid.shape, np.nan)
        full.ravel()[cells] = hand_above(z_ref, cells, first)
        out[name] = full
    return out


def tier_agreement(a: np.ndarray, b: np.ndarray, edges: Sequence[float]) -> dict[str, float]:
    """Weighted (linear) kappa on the tiers of ``edges`` and |a - b| median / q90 over the
    entries finite in both (the P1-10 label-free reproduction check)."""
    from nakkhu import hazard

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[ok] - b[ok])
    n_tiers = len(edges) + 1
    ta, tb = hazard.tier_index(a[ok], edges), hazard.tier_index(b[ok], edges)
    return {"n": int(ok.sum()), "n_missing_a": int((~np.isfinite(a)).sum()),
            "n_missing_b": int((~np.isfinite(b)).sum()),
            "kappa_linear": hazard.weighted_kappa(ta, tb, n_tiers) if ok.any() else np.nan,
            "share_same_tier": float((ta == tb).mean()) if ok.any() else np.nan,
            "median_abs_diff_m": float(np.median(d)) if d.size else np.nan,
            "q90_abs_diff_m": float(np.quantile(d, 0.9)) if d.size else np.nan,
            "mean_abs_diff_m": float(d.mean()) if d.size else np.nan,
            "spearman": hazard.spearman(a[ok], b[ok]) if ok.sum() >= 3 else np.nan}
