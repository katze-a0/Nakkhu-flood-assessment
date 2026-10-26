"""Label-free DEM diagnostics of a (derived) HAND sweep: ``nakkhu hand dem-diagnostics``.

Reads only the sweep's prepared DEMs and routing rasters (``{data_root}/interim/hand/<cfghash>/``),
the building inventories (same filters and order as ``hand attribute``) and the OSM main lines.
No flood label, change map or audit answer is opened, and no skill metric is computed.

Outputs (run ``runs/<stage>-diag/<id>/``):

- ``dem_elevation_diff.csv``: DEM - reference elevation (EGM2008 m, sweep grid) at building
  centroid cells and over the routing domain / analysis extent: median, quartiles, IQR.
- ``dem_hand_diff.csv``: building HAND of each surface minus the reference DEM's HAND at the
  same conditioning and threshold label (``drop`` = each DEM's own drop-analysis threshold).
- ``dem_drain_main.csv``: share of buildings whose first stream cell on the D8 path at a
  threshold lies on the main channel (``drains_main_t`` of prereg_v9_a1 ``definitions``).
- ``dem_artefacts.csv`` / ``dem_artefact_zones.geojson``: native voids, void-filled cells and
  cells far from GLO-30 (possible artefacts), per DEM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from nakkhu import hazard
from nakkhu.config import Project
from nakkhu.hydro import condition as cond
from nakkhu.hydro import dem as demmod
from nakkhu.hydro import route
from nakkhu.hydro.dem import GridSpec
from nakkhu.runs import Run, atomic_write_text, create_run

DIAG_SUFFIX = "-diag"


class DiagnosticsError(RuntimeError):
    """The diagnostics cannot run (missing sweep, rasters or settings)."""


# --- pure helpers ---------------------------------------------------------------------------


def diff_stats(values: np.ndarray, reference: np.ndarray,
               limits_m: tuple[float, ...] = ()) -> dict[str, Any]:
    """Summary of ``values - reference`` over the cells finite in both."""
    d = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    d = d[np.isfinite(d)]
    out: dict[str, Any] = {"n": int(d.size)}
    if d.size == 0:
        return out | {"median_m": np.nan, "q25_m": np.nan, "q75_m": np.nan, "iqr_m": np.nan,
                      "mean_m": np.nan, "median_abs_m": np.nan}
    q25, med, q75 = np.percentile(d, [25, 50, 75])
    out.update({"median_m": float(med), "q25_m": float(q25), "q75_m": float(q75),
                "iqr_m": float(q75 - q25), "mean_m": float(d.mean()),
                "median_abs_m": float(np.median(np.abs(d)))})
    for limit in limits_m:
        out[f"share_abs_gt_{limit:g}m"] = float((np.abs(d) > limit).mean())
    return out


def first_stream_cell(ds: np.ndarray, streams: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Flat index of the first stream cell on the D8 path from each ``start`` cell (the start
    cell itself counts); -1 if the path ends (pit, flat, grid exit) before a stream or the
    start is outside the grid (-1)."""
    streams = np.asarray(streams, dtype=bool).ravel()
    cur = np.asarray(start, dtype=np.int64).copy()
    out = np.full(cur.shape, -1, dtype=np.int64)
    active = cur >= 0
    for _ in range(ds.size + 1):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        hit = streams[cur[idx]]
        out[idx[hit]] = cur[idx[hit]]
        active[idx[hit]] = False
        idx = idx[~hit]
        nxt = ds[cur[idx]]
        stop = nxt < 0
        active[idx[stop]] = False
        cur[idx[~stop]] = nxt[~stop]
    return out


def main_mask(lines: gpd.GeoDataFrame, grid: GridSpec, acc: np.ndarray,
              min_area_km2: float) -> np.ndarray:
    """Main mask M: cells touched by the main lines with contributing area >= ``min_area_km2``."""
    touched = cond.burn_mask(lines, grid, all_touched=True)
    return touched & (acc * grid.cell_area_m2 >= min_area_km2 * 1e6)


def drain_main_summary(ds: np.ndarray, acc: np.ndarray, main: np.ndarray, cells: np.ndarray,
                       grid: GridSpec, threshold_km2: float) -> dict[str, Any]:
    """``drains_main_t`` counts for buildings at centroid ``cells`` (flat indices)."""
    streams = hand_streams(acc, grid, threshold_km2)
    first = first_stream_cell(ds, streams, cells)
    reached = first >= 0
    on_main = np.zeros(cells.shape, dtype=bool)
    on_main[reached] = main.ravel()[first[reached]]
    n = int((cells >= 0).sum())
    return {"n_buildings": n, "n_reach_stream": int(reached.sum()),
            "n_drains_main": int(on_main.sum()),
            "n_drains_tributary": int((reached & ~on_main).sum()),
            "n_no_stream": int(n - reached.sum()),
            "share_drains_main": float(on_main.sum() / n) if n else np.nan,
            "share_drains_tributary": float((reached & ~on_main).sum() / n) if n else np.nan}


def hand_streams(acc: np.ndarray, grid: GridSpec, threshold_km2: float) -> np.ndarray:
    from nakkhu.hydro.hand import stream_mask

    return stream_mask(acc, grid, threshold_km2)


def native_voids(path: Path, target_geoms: dict[str, Any]) -> dict[str, Any]:
    """Void (nodata / non-finite) share of the native raster inside each target (EPSG:32645)."""
    out: dict[str, Any] = {}
    with rasterio.open(path) as src:
        data = src.read(1)
        bad = ~np.isfinite(data)
        if src.nodata is not None:
            bad |= data == src.nodata
        out["native_res_m_or_deg"] = float(abs(src.res[0]))
        out["native_crs"] = src.crs.to_string() if src.crs.to_epsg() else "custom (see manifest)"
        for name, geom in target_geoms.items():
            inside = ~geometry_mask([transform_geom("EPSG:32645", src.crs, mapping(geom))],
                                    out_shape=data.shape, transform=src.transform)
            n = int(inside.sum())
            out[f"native_cells_{name}"] = n
            out[f"native_void_cells_{name}"] = int((bad & inside).sum())
            out[f"native_void_share_{name}"] = float((bad & inside).sum() / n) if n else np.nan
    return out


def zones(mask: np.ndarray, grid: GridSpec, **props: Any) -> list[dict[str, Any]]:
    return [{**props, "area_m2": shape(g).area, "geometry": shape(g)}
            for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform)
            if v == 1]


# --- runner ---------------------------------------------------------------------------------


def run_diagnostics(project: Project, config: str, sweep_ref: str | None = None
                    ) -> tuple[Run, dict[str, Any]]:
    from nakkhu.hydro.sweep import (
        SweepError,
        latest_sweep,
        load_greenhouses,
        load_inventory,
        load_settings,
        sweep_run,
        write_csv,
    )

    settings, _ = load_settings(project, config)
    opts = settings.v.get("diagnostics")
    if not opts:
        raise DiagnosticsError(f"{config} has no `diagnostics` section")
    try:
        sweep = (latest_sweep(project, settings.stage) if sweep_ref is None
                 else sweep_run(project, sweep_ref, settings.stage))
    except SweepError as error:
        raise DiagnosticsError(str(error)) from error
    if sweep.status != "complete":
        raise DiagnosticsError(f"sweep run {sweep.ref} is {sweep.status}")
    summary_in = json.loads((sweep.path / "summary.json").read_text(encoding="utf-8"))
    interim = project.paths.resolve(summary_in["interim"])
    table = pd.read_csv(sweep.path / "config_grid.csv")
    done = table[table["status"] == "done"].copy()
    prep = pd.read_csv(sweep.path / "dem_prep.csv")
    dems = [d for d in settings.v["dems"] if d in set(done["dem"])]
    refs = [r for r in opts["references"] if r in dems]
    limits = tuple(float(x) for x in opts.get("artefact_abs_diff_m", []))
    grid = settings.grid()
    if grid.to_dict() != summary_in["grid"]:
        raise DiagnosticsError("sweep grid differs from the config grid")
    extent_geom = settings.read_polygon(settings.v["analysis_extent"])
    domain_mask = demmod.read_raster(interim / "grid" / "domain.tif").astype(bool)
    extent_mask = grid.mask(extent_geom)
    elev = {d: demmod.read_raster(interim / "grid" / f"dem_{d}.tif").astype(np.float64)
            for d in dems}

    run = create_run(project, settings.stage + DIAG_SUFFIX, settings.config)
    greenhouses = load_greenhouses(settings)
    cells: dict[str, np.ndarray] = {}
    inventory_info: dict[str, Any] = {}
    with run.step("inventories"):
        for name in settings.v["inventories"]:
            frame, info = load_inventory(settings, name, extent_geom, greenhouses)
            cells[name] = hazard.centroid_cells(frame.geometry, grid)
            inventory_info[name] = {"n_buildings": len(frame), **{
                k: v for k, v in info.items() if k != "cut_thresholds"}}

    with run.step("elevation"):
        rows = []
        for ref in refs:
            for d in dems:
                if d == ref:
                    continue
                for region, values, base in (
                        ("routing_domain", elev[d][domain_mask], elev[ref][domain_mask]),
                        ("analysis_extent", elev[d][extent_mask], elev[ref][extent_mask])):
                    rows.append({"reference": ref, "dem": d, "where": region,
                                 **diff_stats(values, base, limits)})
                for name, idx in cells.items():
                    ok = idx >= 0
                    rows.append({"reference": ref, "dem": d, "where": f"buildings:{name}",
                                 **diff_stats(elev[d].ravel()[idx[ok]],
                                              elev[ref].ravel()[idx[ok]], limits)})
        write_csv(run.path / "dem_elevation_diff.csv", rows)

    with run.step("hand"):
        hand_cache: dict[str, np.ndarray] = {}

        def hand(cfg_id: str) -> np.ndarray:
            if cfg_id not in hand_cache:
                hand_cache[cfg_id] = demmod.read_raster(
                    interim / cfg_id / "hand.tif").ravel().astype(np.float64)
            return hand_cache[cfg_id]

        rows = []
        meta = done.set_index("cfg_id")
        for row in done.itertuples():
            for ref in refs:
                if row.dem == ref:
                    continue
                ref_id = f"{ref}__{row.conditioning}__t{str(row.threshold).replace('.', 'p')}"
                if ref_id not in meta.index:
                    continue
                for name, idx in cells.items():
                    ok = idx >= 0
                    a, b = hand(row.cfg_id)[idx[ok]], hand(ref_id)[idx[ok]]
                    rows.append({"reference": ref, "dem": row.dem,
                                 "conditioning": row.conditioning, "threshold": row.threshold,
                                 "threshold_km2": row.threshold_km2,
                                 "reference_threshold_km2": meta.at[ref_id, "threshold_km2"],
                                 "routing_gate_pass": row.routing_gate_pass,
                                 "inventory": name, **diff_stats(a, b),
                                 "spearman": hazard.spearman(a, b),
                                 "n_hand_le_3m": int((a <= 3).sum()),
                                 "n_hand_le_3m_reference": int((b <= 3).sum())})
            hand_cache.clear()
        write_csv(run.path / "dem_hand_diff.csv", rows)

    with run.step("drain-main"):
        lines = gpd.read_file(settings.resolve(opts["main_lines"])).to_crs(grid.crs)
        rows = []
        for d in dems:
            for c in settings.conditionings():
                combo = interim / f"{d}__{c}"
                if not (combo / "fdir.tif").is_file():
                    continue
                fdir = demmod.read_raster(combo / "fdir.tif").astype(np.int64)
                acc = demmod.read_raster(combo / "acc.tif").astype(np.float64)
                ds = route.downstream_index(fdir)
                main = main_mask(lines, grid, acc, float(opts["main_min_area_km2"]))
                for t in opts["drain_thresholds_km2"]:
                    for name, idx in cells.items():
                        rows.append({"dem": d, "conditioning": c, "threshold_km2": float(t),
                                     "inventory": name, "main_cells": int(main.sum()),
                                     **drain_main_summary(ds, acc, main, idx, grid, float(t))})
        write_csv(run.path / "dem_drain_main.csv", rows)

    with run.step("artefacts"):
        rows, zone_rows = [], []
        targets = {"extent": extent_geom,
                   "routing_domain": settings.read_polygon(settings.v["grid"]["domain"])}
        prep_by = prep.set_index("dem")
        for d in dems:
            entry = settings.v["dems"][d]
            native = native_voids(settings.resolve(entry["path"]), targets)
            filled_tif = interim / "grid" / f"void_filled_{d}.tif"
            filled = (demmod.read_raster(filled_tif).astype(bool) if filled_tif.is_file()
                      else np.zeros(grid.shape, dtype=bool))
            row: dict[str, Any] = {"dem": d, **native,
                                   "void_filled_cells_domain": int((filled & domain_mask).sum()),
                                   "void_filled_cells_extent": int((filled & extent_mask).sum())}
            for name, idx in cells.items():
                ok = idx >= 0
                row[f"void_filled_buildings_{name}"] = int(filled.ravel()[idx[ok]].sum())
            if "vertical_shift_median_m" in prep_by.columns and d in prep_by.index:
                row["vertical_from"] = prep_by.at[d, "vertical_from"]
                row["vertical_shift_median_m"] = prep_by.at[d, "vertical_shift_median_m"]
            if d != "glo30" and "glo30" in elev:
                diff = np.abs(elev[d] - elev["glo30"])
                for limit in limits:
                    far = (diff > limit) & ~filled
                    row[f"cells_abs_diff_gt_{limit:g}m_domain"] = int((far & domain_mask).sum())
                    row[f"cells_abs_diff_gt_{limit:g}m_extent"] = int((far & extent_mask).sum())
                    for name, idx in cells.items():
                        ok = idx >= 0
                        row[f"buildings_abs_diff_gt_{limit:g}m_{name}"] = int(
                            far.ravel()[idx[ok]].sum())
                if limits:
                    big = (diff > max(limits)) & ~filled & domain_mask
                    zone_rows += zones(big, grid, dem=d, kind=f"abs_diff_vs_glo30_gt_"
                                                              f"{max(limits):g}m")
            if filled.any():
                zone_rows += zones(filled & domain_mask, grid, dem=d, kind="void_filled_glo30")
            rows.append(row)
        write_csv(run.path / "dem_artefacts.csv", rows)
        frame = (gpd.GeoDataFrame(zone_rows, geometry="geometry", crs=grid.crs)
                 if zone_rows else gpd.GeoDataFrame({"dem": [], "kind": [], "area_m2": []},
                                                    geometry=[], crs=grid.crs))
        atomic_write_text(run.path / "dem_artefact_zones.geojson",
                          frame.to_crs("EPSG:4326").to_json())

    summary = {"sweep_run": sweep.ref, "interim": project.paths.relative(interim),
               "dems": dems, "references": refs, "inventories": inventory_info,
               "label_free": True,
               "main_mask": "cells touched (all_touched) by "
                            f"{opts['main_lines']} with D8 contributing area >= "
                            f"{opts['main_min_area_km2']} km2 on the same DEM x conditioning "
                            "(prereg_v9_a1 definitions.main_mask)"}
    run.write_summary(summary)
    return run, summary
