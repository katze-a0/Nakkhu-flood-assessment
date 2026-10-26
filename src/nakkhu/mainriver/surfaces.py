"""``nakkhu mainriver surfaces``: the main-river HAND ensemble, label-free.

For every gate-passing DEM x conditioning of ``configs/hand/main_river_v1.yaml`` the full
hand_mr_d8 raster is computed from the source sweep's own routing rasters (``fdir.tif``,
``acc.tif``, ``dem_ref.tif``) over the routing domain, then every building of the configured
inventories is attributed with the three building statistics exactly as ``hand attribute`` does
(centroid cell; minimum over footprint cells; minimum over the footprint + ``buffer_m``).

Outputs (run ``hand-mr/<run_id>``; rasters under ``data/interim/hand_mr/<cfghash>/``):

- ``surface_grid.csv``: every DEM x conditioning, routing gate, ensemble, raster statistics and
  sha256;
- ``building_hand_mr_<inventory>.parquet``: building_uid, area_m2 and one column per member;
- ``count_stability.csv`` / ``tier_range_summary.csv``: tier counts per member and their min /
  max over the primary ensemble (and, as a separate group, the extended one);
- ``building_tiers_<inventory>.csv`` + ``building_tiers.json``: the per-building tier export of
  the primary member (``hand tier-export`` format, for the exposure map);
- ``surface_checks.json``: the primary (and FABDEM breach) centroid values against the promoted
  plan 014 prep columns (same definition: must be identical).

It reads no change map, no audit answer and no reference label.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np
import pandas as pd

from nakkhu import hazard
from nakkhu.config import Project, ResolvedConfig
from nakkhu.evaluate.guard import sha256_file
from nakkhu.mainriver import spec as S

RASTER = "hand_mr.tif"


def main_lines(project: Project, cfg: dict[str, Any], crs: str) -> Any:
    import geopandas as gpd

    d = cfg["drainage"]
    path = project.paths.resolve(d["main_lines"])
    if sha256_file(path) != d["main_lines_sha256"]:
        raise S.MainRiverError(f"{d['main_lines']} is not the pinned file")
    lines = gpd.read_file(path).to_crs(crs)
    return lines[lines.geometry.notna() & ~lines.geometry.is_empty]


def hand_mr_raster(fdir: np.ndarray, acc: np.ndarray, z_ref: np.ndarray, domain: np.ndarray,
                   lines: Any, grid: Any, min_area_km2: float) -> tuple[np.ndarray, dict]:
    """hand_mr_d8 on every grid cell with a finite z_ref (as the prep and the sweep HAND, which
    are not masked to the routing domain: buildings of the extent outside it keep a value);
    NaN where the D8 path ends (pit, flat, grid exit) before M. Statistics are reported over
    the routing ``domain``."""
    from nakkhu.followup import surfaces as F
    from nakkhu.hydro import route

    ds = route.downstream_index(np.asarray(fdir).astype(np.int64))
    main = F.main_mask(lines, grid, acc, min_area_km2)
    cells = np.flatnonzero(np.isfinite(np.asarray(z_ref, dtype=np.float64)).ravel())
    first, _ = F.walk_downstream(ds, main, cells, grid.width, grid.res)
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    out.ravel()[cells] = F.hand_above(z_ref, cells, first)
    dom = np.flatnonzero(np.asarray(domain, dtype=bool).ravel())
    vals = out.ravel()[dom]
    info = {"main_cells": int(main.sum()), "n_domain_cells": int(dom.size),
            "n_grid_cells_finite": int(np.isfinite(out).sum()),
            "n_nan_in_domain": int(np.isnan(vals).sum()),
            "min_m": float(np.nanmin(vals)), "max_m": float(np.nanmax(vals)),
            "mean_m": float(np.nanmean(vals)),
            "share_zero": float(np.mean(vals[np.isfinite(vals)] == 0.0))}
    return out.astype(np.float32), info


def _settings(project: Project, cfg: dict[str, Any]) -> Any:
    from nakkhu.hydro.sweep import load_settings

    settings, _ = load_settings(project, cfg["base_config"])
    if [float(e) for e in settings.edges] != [float(e) for e in cfg["tiers_m"]]:
        raise S.MainRiverError("tiers_m differ from the base sweep config")
    if [str(s) for s in settings.v["statistics"]] != S.statistics(cfg):
        raise S.MainRiverError("statistics differ from the base sweep config")
    return settings


def run_surfaces(project: Project, config: str = S.ENSEMBLE_CONFIG) -> tuple[Any, dict]:
    from nakkhu.hydro import dem as demmod
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, write_csv
    from nakkhu.hydro.tier_export import tier_counts, tier_table
    from nakkhu.runs import atomic_write_json, create_run, is_complete

    cfg = S.load_ensemble(project, config)
    settings = _settings(project, cfg)
    grid = settings.grid()
    surfs = S.surfaces(project, cfg)
    for ref in {s.sweep_run for s in surfs}:
        if not is_complete(S.sweep_dir(project, ref)):
            raise S.MainRiverError(f"sweep run {ref} is not complete")
        if S.sweep_grid(project, ref) != grid.to_dict():
            raise S.MainRiverError(f"sweep run {ref} is on another grid")
    canonical = S.sweep_interim(project, str(cfg["sources"]["canonical"]))
    values = {"command": "mainriver-surfaces", "version": cfg["version"],
              "config": config, "config_sha256": sha256_file(project.paths.resolve(config)),
              "base_config_sha256": sha256_file(project.paths.resolve(cfg["base_config"]))}
    run = create_run(project, str(cfg["stage"]), ResolvedConfig(values=values, source=config))
    interim = project.paths.data_root / "interim" / "hand_mr" / run.run_id.rsplit("-", 1)[-1]
    lines = main_lines(project, cfg, settings.crs)
    stats = S.statistics(cfg)
    edges = [float(e) for e in cfg["tiers_m"]]
    labels = hazard.tier_labels(edges)
    pcol = S.primary_column(cfg)
    summary: dict[str, Any] = {"version": cfg["version"], "interim": project.paths.relative(
        interim), "grid": grid.to_dict(), "primary": pcol, "surfaces": {}, "inventories": {}}
    grid_rows: list[dict[str, Any]] = []
    rasters: dict[str, np.ndarray] = {}
    with run.step("rasters"):
        (interim / "grid").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical / "grid" / "cell_index.tif", interim / "grid" / "cell_index.tif")
        shutil.copyfile(canonical / "grid" / "domain.tif", interim / "grid" / "domain.tif")
        domain = demmod.read_raster(canonical / "grid" / "domain.tif").astype(bool)
        for s in surfs:
            row: dict[str, Any] = {"cfg_id": s.cfg_id, "dem": s.dem, "conditioning": s.conditioning,
                                   "threshold": S.SUFFIX, "drainage": cfg["drainage"]["definition"],
                                   "source": s.source, "sweep_run": s.sweep_run,
                                   "ensemble": s.ensemble, "routing_gate_pass": s.routing_gate_pass,
                                   "routing_gate_reason": s.gate_reason,
                                   "in_sensitivity_range": s.in_sensitivity_range,
                                   "is_primary": s.is_primary, "diagnostic": False,
                                   "status": "done" if s.computed else "not computed (gate)"}
            if s.computed:
                src = S.sweep_interim(project, s.sweep_run) / f"{s.dem}__{s.conditioning}"
                r = {k: demmod.read_raster(src / f"{k}.tif") for k in ("fdir", "acc", "dem_ref")}
                hand, info = hand_mr_raster(r["fdir"], r["acc"].astype(np.float64),
                                            r["dem_ref"].astype(np.float64), domain, lines, grid,
                                            float(cfg["drainage"]["main_min_area_km2"]))
                path = demmod.write_raster(interim / s.cfg_id / RASTER, hand, grid,
                                           nodata=np.nan)
                rasters[s.cfg_id] = hand.ravel()
                row.update(info)
                row["inputs_sha256"] = json.dumps({k: sha256_file(src / f"{k}.tif")
                                                   for k in ("fdir", "acc", "dem_ref")})
                row["raster"] = project.paths.relative(path)
                row["raster_sha256"] = sha256_file(path)
            grid_rows.append(row)
            summary["surfaces"][s.cfg_id] = {k: row.get(k) for k in (
                "raster", "raster_sha256", "routing_gate_pass", "ensemble", "status")}
        summary["cell_index_sha256"] = sha256_file(interim / "grid" / "cell_index.tif")
    meta = pd.DataFrame(grid_rows).set_index("cfg_id")
    write_csv(run.path / "surface_grid.csv", grid_rows)

    stability: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    tiers_meta: dict[str, Any] = {"columns": ["building_id", "x", "y", "hand_m", "tier"],
                                  "crs": settings.crs, "hand_column": pcol, "edges": edges,
                                  "inventories": {}}
    gh = load_greenhouses(settings)
    extent = settings.read_polygon(settings.v["analysis_extent"])
    for name in cfg["inventories"]:
        with run.step(f"attribute-{name.lower()}"):
            frame, info = load_inventory(settings, name, extent, gh)
            cent = hazard.centroid_cells(frame.geometry, grid)
            index_tif = interim / "grid" / "cell_index.tif"
            foot = hazard.cell_groups(frame.geometry, index_tif)
            buf = hazard.cell_groups(frame.geometry.buffer(float(settings.v["buffer_m"])),
                                     index_tif)
            cols: dict[str, np.ndarray] = {}
            for cid, vals in rasters.items():
                for stat in stats:
                    if stat == "centroid":
                        v = hazard.sample_centroids(vals, cent)
                    elif stat == "footprint_min":
                        v = foot.minimum(vals)
                    elif stat == "buffer_min":
                        v = buf.minimum(vals)
                    else:
                        raise S.MainRiverError(f"unknown statistic {stat!r}")
                    cols[f"{cid}__{stat}"] = np.asarray(v).astype(np.float32)
            table = pd.DataFrame({"building_uid": frame["building_uid"],
                                  "area_m2": frame.geometry.area.astype(np.float32), **cols})
            table.to_parquet(run.path / f"building_hand_mr_{name}.parquet", index=False,
                             compression="zstd")
            ref_vals = cols[pcol]
            ref_t = hazard.tier_index(ref_vals, edges)
            for key, col in cols.items():
                cid, stat = key.rsplit("__", 1)
                t = hazard.tier_index(col, edges)
                counts = hazard.tier_counts(t, len(labels))
                stability.append({
                    "inventory": name, "inventory_role": info["role"], "cfg_id": cid,
                    "dem": meta.at[cid, "dem"], "conditioning": meta.at[cid, "conditioning"],
                    "threshold": S.SUFFIX, "threshold_km2": np.nan, "statistic": stat,
                    "ensemble": meta.at[cid, "ensemble"], "is_primary": key == pcol,
                    "diagnostic": False,
                    "routing_gate_pass": bool(meta.at[cid, "routing_gate_pass"]),
                    "in_sensitivity_range": bool(meta.at[cid, "in_sensitivity_range"]),
                    "n_buildings": len(col), "n_missing": int((t < 0).sum()),
                    **{f"n_{lab}": int(n) for lab, n in zip(labels, counts, strict=True)},
                    "spearman_vs_primary": hazard.spearman(col.astype(np.float64),
                                                           ref_vals.astype(np.float64)),
                    "kappa_linear_vs_primary": hazard.weighted_kappa(t, ref_t, len(labels)),
                    "share_same_tier_vs_primary": float(
                        (t[(t >= 0) & (ref_t >= 0)] == ref_t[(t >= 0) & (ref_t >= 0)]).mean())})
            checks[name] = prep_check(project, cfg, name, table)
            tt = tier_table(frame, table, pcol, edges)
            tt.to_csv(run.path / f"building_tiers_{name}.csv", index=False,
                      lineterminator="\n")
            tiers_meta["inventories"][name] = {
                "file": f"building_tiers_{name}.csv", "counts": tier_counts(tt, labels),
                "n": len(tt), "n_missing": int(tt["tier"].isna().sum()), "filters": info}
            summary["inventories"][name] = {"n_buildings": len(frame), **info}
    st = pd.DataFrame(stability)
    write_csv(run.path / "count_stability.csv", stability)
    write_csv(run.path / "tier_range_summary.csv", tier_ranges(st, labels))
    atomic_write_json(run.path / "surface_checks.json", checks)
    atomic_write_json(run.path / "building_tiers.json", tiers_meta)
    bad = [f"{inv}:{k}" for inv, c in checks.items() for k, v in c.items()
           if not v.get("identical")]
    summary["prep_check_identical"] = not bad
    summary["prep_check_failures"] = bad
    run.write_summary(summary)
    return run, summary


def tier_ranges(stability: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """``hand.sweep.range_summary`` over the primary ensemble (``in_sensitivity_range``), plus
    one descriptive group over primary + extended gate-passing members."""
    from nakkhu.hydro.sweep import range_summary

    base = range_summary(stability, labels)
    cols = [f"n_{lab}" for lab in labels]
    frame = stability.assign(n_le_3=stability[cols[0]] + stability[cols[1]])
    rows = []
    for inventory, inv in frame.groupby("inventory", sort=False):
        primary = inv[inv["is_primary"]].iloc[0]
        g = inv[inv["routing_gate_pass"].astype(bool)]
        for group, sub in (("extended: all gate-passing DEMs x statistics", g),
                           ("extended: centroid", g[g["statistic"] == "centroid"])):
            row: dict[str, Any] = {"inventory": inventory, "group": group,
                                   "n_members": len(sub),
                                   "kappa_linear_vs_primary_min": sub[
                                       "kappa_linear_vs_primary"].min()}
            for c in [*cols, "n_le_3"]:
                row[f"{c}_primary"] = int(primary[c])
                row[f"{c}_min"] = int(sub[c].min())
                row[f"{c}_max"] = int(sub[c].max())
            rows.append(row)
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def prep_check(project: Project, cfg: dict[str, Any], inventory: str,
               table: pd.DataFrame) -> dict[str, Any]:
    """Centroid values vs the promoted plan 014 prep columns of the same definition."""
    out: dict[str, Any] = {}
    for cid, c in (cfg.get("prep_check") or {}).items():
        col = f"{cid}__{S.SUFFIX}__centroid"
        path = project.paths.resolve(str(c["file"]).format(inventory=inventory))
        if not path.is_file() or col not in table.columns:
            out[cid] = {"identical": False, "reason": f"{path.name} or {col} missing"}
            continue
        prep = pd.read_parquet(path, columns=["building_uid", c["column"]])
        if not np.array_equal(prep["building_uid"].astype(str).to_numpy(),
                              table["building_uid"].astype(str).to_numpy()):
            out[cid] = {"identical": False, "reason": "building order differs"}
            continue
        a = prep[c["column"]].to_numpy(dtype=np.float64).astype(np.float32)
        b = table[col].to_numpy(dtype=np.float32)
        same_nan = np.isnan(a) == np.isnan(b)
        ok = ~np.isnan(a) & ~np.isnan(b)
        out[cid] = {"file": project.paths.relative(path), "file_sha256": sha256_file(path),
                    "column": c["column"], "n": len(a), "nan_mismatch": int((~same_nan).sum()),
                    "n_nan": int(np.isnan(b).sum()),
                    "max_abs_diff_m": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None,
                    "identical": bool(same_nan.all() and np.array_equal(a[ok], b[ok]))}
    return out
