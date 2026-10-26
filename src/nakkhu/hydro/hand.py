"""Height above nearest drainage and its units check (plan 008, Steps 5-6; task T06).

HAND(c) = z(c) - z(d), where d is the first stream cell on the D8 flow path from c. The primary
implementation is pysheds ``compute_hand``; WhiteboxTools ``elevation_above_stream`` and pyflwdir
``FlwdirRaster.hand`` are the H1 cross-checks. Production surfaces route on the conditioned
(burned) DEM and measure z on the reference elevations (burn undone), so a cell lower than
its drainage cell gets a negative raw value: :func:`check_hand` clamps those to 0 and reports
their share.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from nakkhu.hydro.dem import GridSpec, read_raster, write_raster
from nakkhu.hydro.route import DIRMAP, as_raster, downstream_index, pysheds_grid

PYFLWDIR_NODATA = -9999.0


class HandUnitsError(ValueError):
    """HAND fails the units check (plan 008 STOP condition)."""


def stream_mask(acc: np.ndarray, grid: GridSpec, threshold_km2: float) -> np.ndarray:
    """Cells whose contributing area (accumulation x cell area) reaches ``threshold_km2``."""
    return acc * grid.cell_area_m2 >= threshold_km2 * 1e6


def hand_pysheds(fdir: np.ndarray, elevation: np.ndarray, streams: np.ndarray,
                 grid: GridSpec) -> np.ndarray:
    """Raw HAND (may be negative); NaN where the flow path never reaches a stream."""
    pgrid, _, _ = pysheds_grid(grid)
    hand = pgrid.compute_hand(
        as_raster(fdir, grid, np.int64(0)),
        as_raster(elevation.astype(np.float64), grid, np.nan),
        as_raster(streams.astype(bool), grid, False),
        dirmap=DIRMAP,
        nodata_out=np.nan,
    )
    return np.asarray(hand, dtype=np.float64)


def hand_pyflwdir(routing_dem: np.ndarray, elevation: np.ndarray, streams: np.ndarray,
                  grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    """``(hand, d8)``: pyflwdir derives its own D8 from ``routing_dem`` (``from_dem``)."""
    import pyflwdir

    flw = pyflwdir.from_dem(
        data=np.nan_to_num(routing_dem, nan=PYFLWDIR_NODATA).astype(np.float64),
        nodata=PYFLWDIR_NODATA, transform=grid.transform, latlon=False, outlets="edge",
    )
    hand = flw.hand(drain=streams.astype(bool), elevtn=elevation.astype(np.float64))
    hand = hand.astype(np.float64)
    hand[(hand == PYFLWDIR_NODATA) | np.isnan(routing_dem)] = np.nan
    return hand, flw.to_array(ftype="d8")


def hand_pyflwdir_on_d8(d8: np.ndarray, elevation: np.ndarray, streams: np.ndarray,
                        grid: GridSpec) -> np.ndarray:
    """pyflwdir's HAND step on given D8 directions (ESRI codes; pits, flats, nodata -> pit).

    Isolates the HAND computation from pyflwdir's own flow-direction rule (H1)."""
    import pyflwdir

    codes = np.where(np.isin(d8, DIRMAP), d8, 0).astype(np.uint8)
    flw = pyflwdir.from_array(codes, ftype="d8", transform=grid.transform, latlon=False)
    hand = flw.hand(drain=streams.astype(bool), elevtn=elevation.astype(np.float64))
    hand = hand.astype(np.float64)
    hand[hand == PYFLWDIR_NODATA] = np.nan
    return hand


def whitebox_d8(routing_dem: np.ndarray, grid: GridSpec, wbt: Any,
                workdir: str | Path | None = None) -> np.ndarray:
    """WhiteboxTools ``d8_pointer`` in ESRI codes (0 = no receiver)."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        dem_path, out = Path(tmp) / "dem.tif", Path(tmp) / "d8.tif"
        write_raster(dem_path, np.nan_to_num(routing_dem, nan=-9999.0).astype(np.float64),
                     grid, nodata=-9999.0, plain=True)
        rc = wbt.d8_pointer(str(dem_path), str(out), esri_pntr=True)
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"WhiteboxTools d8_pointer failed (rc={rc})")
        with rasterio.open(out) as src:
            d8 = src.read(1).astype(np.int64)
            nodata = src.nodata
    if nodata is not None:
        d8[d8 == int(nodata)] = 0
    return d8


def hand_whitebox(routing_dem: np.ndarray, streams: np.ndarray, grid: GridSpec,
                  wbt: Any, workdir: str | Path | None = None) -> np.ndarray:
    """WhiteboxTools ``elevation_above_stream``: directions and z both from ``routing_dem``."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        dem_path, str_path, out = (Path(tmp) / n for n in ("dem.tif", "streams.tif", "hand.tif"))
        write_raster(dem_path, np.nan_to_num(routing_dem, nan=-9999.0).astype(np.float64),
                     grid, nodata=-9999.0, plain=True)
        write_raster(str_path, streams.astype(np.uint8), grid, nodata=None, plain=True)
        rc = wbt.elevation_above_stream(str(dem_path), str(str_path), str(out))
        if rc != 0 or not out.is_file():
            raise RuntimeError(f"WhiteboxTools elevation_above_stream failed (rc={rc})")
        hand = read_raster(out).astype(np.float64)
    hand[np.isnan(routing_dem)] = np.nan
    return hand


@dataclass
class UnitsReport:
    n_domain_cells: int
    n_nan_in_domain: int
    n_nan_exit_grid: int
    n_stream_cells_in_domain: int
    stream_max_abs_m: float
    negative_share: float
    negative_min_m: float
    min_m: float
    max_m: float
    mean_m: float
    relief_m: float
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["units_ok"] = self.ok
        out["problems"] = "; ".join(self.problems)
        return out


def exits_grid(fdir: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """For flat indices ``cells``: does the D8 path end on the grid rim (leaves the grid)?"""
    height, width = fdir.shape
    ds = downstream_index(fdir)
    out = np.zeros(cells.size, dtype=bool)
    for k, cell in enumerate(cells.tolist()):
        for _ in range(fdir.size):
            nxt = int(ds[cell])
            if nxt < 0:
                break
            cell = nxt
        r, c = divmod(cell, width)
        out[k] = r in (0, height - 1) or c in (0, width - 1)
    return out


def check_hand(raw: np.ndarray, domain: np.ndarray, streams: np.ndarray,
               reference: np.ndarray, fdir: np.ndarray | None = None
               ) -> tuple[np.ndarray, UnitsReport]:
    """Clamp negatives to 0 and check metres, >= 0, finite in ``domain``, 0 on streams.

    ``reference`` is the elevation surface HAND was measured on; HAND cannot exceed its relief
    over the domain, which catches the old 0-31633 values (non-metre units). With ``fdir``, a
    non-finite domain cell whose flow path leaves the grid without meeting a stream (the DEM's
    divide differs from the GLO-30 domain) is counted in ``n_nan_exit_grid`` and tolerated;
    any other non-finite cell (an interior pit) fails.
    """
    inside = raw[domain]
    finite = inside[np.isfinite(inside)]
    negative = finite < 0
    zdom = reference[domain]
    relief = float(np.nanmax(zdom) - np.nanmin(zdom))
    hand = np.where(raw < 0, 0.0, raw)
    clamped = hand[domain]
    on_stream = hand[domain & streams]
    nan_cells = np.flatnonzero(domain.ravel() & np.isnan(raw.ravel()))
    n_exit = int(exits_grid(fdir, nan_cells).sum()) if fdir is not None and nan_cells.size else 0
    report = UnitsReport(
        n_domain_cells=int(domain.sum()),
        n_nan_in_domain=int(nan_cells.size),
        n_nan_exit_grid=n_exit,
        n_stream_cells_in_domain=int((domain & streams).sum()),
        stream_max_abs_m=float(np.nanmax(np.abs(on_stream))) if on_stream.size else 0.0,
        negative_share=float(negative.mean()) if finite.size else 0.0,
        negative_min_m=float(finite[negative].min()) if negative.any() else 0.0,
        min_m=float(np.nanmin(clamped)),
        max_m=float(np.nanmax(clamped)),
        mean_m=float(np.nanmean(clamped)),
        relief_m=relief,
    )
    if report.n_nan_in_domain > report.n_nan_exit_grid:
        report.problems.append(f"{report.n_nan_in_domain - report.n_nan_exit_grid} non-finite "
                               "cells inside the domain (not draining off the grid)")
    if report.stream_max_abs_m > 1e-6:
        report.problems.append(f"stream cells not 0 (max |HAND| {report.stream_max_abs_m:g} m)")
    if report.max_m > relief + 1e-3:
        report.problems.append(f"max HAND {report.max_m:.2f} m exceeds domain relief "
                               f"{relief:.2f} m (units?)")
    if report.min_m < 0:
        report.problems.append("negative HAND after clamping")
    return hand, report
