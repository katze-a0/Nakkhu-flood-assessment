"""Objective stream-initiation threshold by constant-drop analysis (Tarboton et al. 1991).

For each candidate contributing-area threshold the stream network (inside the domain) is split
into Strahler streams, i.e. maximal runs of cells with the same Strahler order. A stream's drop
is the elevation at its first cell minus the elevation of the junction cell it flows into;
streams that leave the network (the outlet, the grid edge) are incomplete and excluded. The
mean drop of first-order streams is compared with that of higher-order streams by a Welch
t-statistic, and the selected threshold is the smallest candidate with ``|t| < t_critical``
(TauDEM ``DropAnalysis`` rule).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrahlerStreams:
    order: np.ndarray  # per stream: Strahler order
    start: np.ndarray  # per stream: flat index of the first (most upstream) cell
    junction: np.ndarray  # per stream: flat index of the receiving cell, -1 if incomplete


def strahler_streams(ds: np.ndarray, stream: np.ndarray, acc: np.ndarray) -> StrahlerStreams:
    """Strahler streams of the network ``stream`` (flat bool) with receivers ``ds``."""
    cells = np.flatnonzero(stream)
    cells = cells[np.argsort(acc[cells], kind="stable")]  # donors before receivers
    n = stream.size
    max_up = np.zeros(n, dtype=np.int32)
    n_max = np.zeros(n, dtype=np.int32)
    carry = np.full(n, -1, dtype=np.int64)  # stream id arriving with the maximum order
    orders: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for cell in cells.tolist():
        if max_up[cell] == 0:
            order, sid = 1, -1
        elif n_max[cell] >= 2:
            order, sid = int(max_up[cell]) + 1, -1
        else:
            order, sid = int(max_up[cell]), int(carry[cell])
        if sid < 0:
            sid = len(orders)
            orders.append(order)
            starts.append(cell)
            ends.append(cell)
        else:
            ends[sid] = cell
        receiver = int(ds[cell])
        if receiver >= 0 and stream[receiver]:
            if order > max_up[receiver]:
                max_up[receiver], n_max[receiver], carry[receiver] = order, 1, sid
            elif order == max_up[receiver]:
                n_max[receiver] += 1
    end_arr = np.asarray(ends, dtype=np.int64)
    junction = ds[end_arr] if end_arr.size else np.zeros(0, dtype=np.int64)
    if junction.size:
        valid = junction >= 0
        valid[valid] = stream[junction[valid]]
        junction = np.where(valid, junction, -1)
    return StrahlerStreams(order=np.asarray(orders, dtype=np.int32),
                           start=np.asarray(starts, dtype=np.int64), junction=junction)


@dataclass(frozen=True)
class DropTest:
    threshold_km2: float
    n_stream_cells: int
    max_order: int
    n_first: int
    n_higher: int
    mean_drop_first_m: float
    mean_drop_higher_m: float
    t: float


def drop_test(streams: StrahlerStreams, z: np.ndarray, threshold_km2: float,
              n_stream_cells: int) -> DropTest:
    complete = streams.junction >= 0
    drops = z[streams.start[complete]] - z[streams.junction[complete]]
    orders = streams.order[complete]
    first, higher = drops[orders == 1], drops[orders >= 2]
    t = float("nan")
    if first.size >= 2 and higher.size >= 2:
        se = np.sqrt(first.var(ddof=1) / first.size + higher.var(ddof=1) / higher.size)
        t = float((first.mean() - higher.mean()) / se) if se > 0 else float("nan")
    return DropTest(
        threshold_km2=float(threshold_km2),
        n_stream_cells=int(n_stream_cells),
        max_order=int(streams.order.max()) if streams.order.size else 0,
        n_first=int(first.size),
        n_higher=int(higher.size),
        mean_drop_first_m=float(first.mean()) if first.size else float("nan"),
        mean_drop_higher_m=float(higher.mean()) if higher.size else float("nan"),
        t=t,
    )


def drop_analysis(ds: np.ndarray, contributing_km2: np.ndarray, z: np.ndarray,
                  domain: np.ndarray, candidates_km2: Sequence[float],
                  t_critical: float = 2.0) -> tuple[list[DropTest], float | None]:
    """Tests per candidate (ascending) and the selected threshold (None if none passes).

    ``contributing_km2``, ``z`` and ``domain`` are flat arrays on the same grid as ``ds``.
    """
    tests = []
    for threshold in sorted(candidates_km2):
        stream = domain & (contributing_km2 >= threshold)
        streams = strahler_streams(ds, stream, contributing_km2)
        tests.append(drop_test(streams, z, threshold, int(stream.sum())))
    selected = next((t.threshold_km2 for t in tests
                     if np.isfinite(t.t) and abs(t.t) < t_critical), None)
    return tests, selected
