"""``nakkhu followup prep`` (plan 014 Step 2): every label-free table the follow-up tests read.

Reads the pinned v9 building HAND tables, the v9 predictors (``flood.validate.predictors``: the
same building order, centroids, distance to the channel and blocks as journal 47), the sweep
runs' interim routing rasters, the team's HAND zips and the UNOSAT zip (integrity, CRS and
extent only). No change-map class, observability flag, audit item or answer is opened, and no
skill metric is computed. Outputs (run ``runs/followup-prep/<id>/``, promoted to
``artifacts/followup/prep_*``):

- ``prep_<group>_<inventory>.parquet``: per building (v9 order), groups ``core`` (uid, x, y,
  block, hand_v9, distance, dist_cell, area), one per DEM (``hand_any`` / ``dist_stream`` /
  ``drains_main`` per threshold, ``hand_mr_d8``, ``hand_mr_rem``, ``flowdist_main``),
  ``features`` (the P1-6 table) and ``team`` (hand_team, coverage, team masks, the team
  recipe on our DEMs).
- ``prep_drain_shares.csv``: label-free ``drains_main_t`` shares per DEM x threshold.
- ``prep_surface_checks.csv``: recomputed hand_any vs the sweep's stored surfaces.
- ``prep_team_reproduction.csv`` / ``prep_team_coverage.csv``: the P1-10 label-free check and
  the team-basin coverage.
- ``prep_unosat.json``: UNOSAT zip integrity, layers, CRS, extent, metadata keywords.
- ``prep_checks.json``: HAND-table pins, join check, distance reproduction, clarifications.
- ``prep_manifest.json``: sha256 of every prep file and of the inputs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from nakkhu.config import Project, ResolvedConfig, load_yaml
from nakkhu.followup import spec as S
from nakkhu.followup import surfaces as F
from nakkhu.hydro.dem import GridSpec, read_raster

MAIN_MIN_AREA_KM2 = 1.0        # definitions.main_mask
REM_RADIUS_M = 60.0            # definitions.hand_mr_rem
TPI_RADII_M = (150.0, 500.0)
DENSITY_RADIUS_M = 100.0
MASK_CUTS = {"lt4": 4.0, "lt6": 6.0, "lt10": 10.0}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


# --- DEM sources ------------------------------------------------------------------------------


class DemSource:
    """One DEM's breach_burn routing from a complete sweep run (its own resolved config)."""

    def __init__(self, project: Project, name: str, entry: dict[str, Any]):
        from nakkhu.hydro.sweep import Settings
        from nakkhu.runs import is_complete

        self.name = name
        self.dem = str(entry.get("dem", name))
        self.ref = str(entry["sweep"])
        self.run = project.paths.runs / self.ref
        if not is_complete(self.run):
            raise S.FollowupError(f"sweep run {self.ref} is not complete")
        summary = json.loads((self.run / "summary.json").read_text(encoding="utf-8"))
        self.interim = project.paths.resolve(summary["interim"])
        values = load_yaml(self.run / "config.resolved.yaml")
        self.settings = Settings(project, ResolvedConfig(values=values, source=self.ref))
        self.grid = self.settings.grid()
        if self.grid.to_dict() != summary["grid"]:
            raise S.FollowupError(f"{self.ref}: grid differs from the run summary")
        combo = f"{self.dem}__breach_burn"
        self.combo = combo
        self.drop_km2 = (summary.get("drop_thresholds_km2") or {}).get(combo)
        gate = pd.read_csv(self.run / "burn_check.csv")
        row = gate.loc[(gate["dem"] == self.dem) & (gate["conditioning"] == "breach_burn")]
        self.gate_pass = bool(row["gate_pass"].iloc[0]) if len(row) else None
        self.gate_reason = str(row["gate_reason"].iloc[0]) if len(row) else "no burn check"
        self.thresholds_kind = str(entry.get("thresholds", "p1_4"))

    def path(self, *parts: str) -> Path:
        return self.interim.joinpath(*parts)

    def load(self) -> dict[str, np.ndarray]:
        from nakkhu.hydro import route

        fdir = read_raster(self.path(self.combo, "fdir.tif")).astype(np.int64)
        return {"z_ref": read_raster(self.path(self.combo, "dem_ref.tif")).astype(np.float64),
                "fdir": fdir, "ds": route.downstream_index(fdir),
                "acc": read_raster(self.path(self.combo, "acc.tif")).astype(np.float64),
                "z_raw": read_raster(self.path("grid", f"dem_{self.dem}.tif")).astype(
                    np.float64),
                "burn": read_raster(self.path("grid", "burn_mask.tif")).astype(bool)}

    def stored_surface(self, threshold_km2: float) -> Path | None:
        """The sweep's own hand.tif at this threshold (a fixed one, or the drop one)."""
        fixed = [float(t) for t in self.settings.v["thresholds_km2"]]
        if any(abs(threshold_km2 - t) < 1e-12 for t in fixed):
            label = S.tlabel(threshold_km2)
        elif self.drop_km2 is not None and abs(threshold_km2 - float(self.drop_km2)) < 1e-12:
            label = "tdrop"
        else:
            return None
        path = self.path(f"{self.combo}__{label}", "hand.tif")
        return path if path.is_file() else None

    def meta(self) -> dict[str, Any]:
        return {"dem": self.dem, "sweep_run": self.ref, "grid": self.grid.to_dict(),
                "interim": str(self.interim), "drop_threshold_km2": self.drop_km2,
                "routing_gate_pass": self.gate_pass, "routing_gate_reason": self.gate_reason,
                "thresholds": self.thresholds_kind}


def thresholds_of(source: DemSource, spec: dict[str, Any]) -> list[float]:
    if source.thresholds_kind == "all16":
        out = S.candidates_km2(spec)
    elif source.thresholds_kind == "p1_4":
        out = S.p1_4_thresholds(spec)
    else:
        raise S.FollowupError(f"{source.name}: unknown thresholds {source.thresholds_kind!r}")
    if source.drop_km2 is not None and not any(abs(t - source.drop_km2) < 1e-12 for t in out):
        out = [*out, float(source.drop_km2)]
    return sorted(out)


def main_lines(project: Project, spec: dict[str, Any], crs: str) -> gpd.GeoDataFrame:
    path = project.paths.resolve(spec["common"]["main_lines"]["sha256_of"])
    lines = gpd.read_file(path).to_crs(crs)
    return lines[lines.geometry.notna() & ~lines.geometry.is_empty]


def dem_columns(source: DemSource, spec: dict[str, Any], lines: gpd.GeoDataFrame,
                points: dict[str, tuple[np.ndarray, np.ndarray]], *, features: bool = False
                ) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]],
                           list[dict[str, Any]], dict[str, Any]]:
    """Per inventory the DEM's columns; drain-share rows; stored-surface check rows; info."""
    r = source.load()
    grid, name = source.grid, source.name
    main = F.main_mask(lines, grid, r["acc"], MAIN_MIN_AREA_KM2)
    cols: dict[str, dict[str, np.ndarray]] = {inv: {} for inv in points}
    drains: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    info: dict[str, Any] = {"main_cells": int(main.sum()), "burn_cells": int(r["burn"].sum()),
                            "burn_cells_below_threshold": {}}
    cells = {inv: F.point_cells(grid, x, y) for inv, (x, y) in points.items()}
    first_main = {}
    for inv, c in cells.items():
        first, length = F.walk_downstream(r["ds"], main, c, grid.width, grid.res)
        first_main[inv] = first
        cols[inv][f"hand_mr_d8__{name}"] = F.hand_above(r["z_ref"], c, first)
        cols[inv][f"hand_mr_rem__{name}"] = F.hand_mr_rem(grid, main, r["z_raw"], c,
                                                          REM_RADIUS_M)
        cols[inv][f"flowdist_main__{name}"] = length
        cols[inv][f"elev__{name}"] = F.sample(r["z_raw"], c)
    first_drop: dict[str, np.ndarray] = {}
    for t in thresholds_of(source, spec):
        lab = S.tlabel(t)
        streams = F.stream_mask(r["acc"], grid, t)
        info["burn_cells_below_threshold"][lab] = int((r["burn"] & ~streams).sum())
        stored = source.stored_surface(t)
        stored_vals = read_raster(stored).astype(np.float64).ravel() if stored else None
        for inv, c in cells.items():
            x, y = points[inv]
            first, _ = F.walk_downstream(r["ds"], streams, c, grid.width, grid.res)
            hand = F.hand_above(r["z_ref"], c, first)
            on_main = np.where(first >= 0, main.ravel()[np.maximum(first, 0)], False)
            drains_main = np.where(first >= 0, on_main.astype(float), np.nan)
            cols[inv][f"hand_any__{name}__{lab}"] = hand
            cols[inv][f"dist_stream__{name}__{lab}"] = F.nearest_distance(grid, streams, x, y)
            cols[inv][f"drains_main__{name}__{lab}"] = drains_main
            if source.drop_km2 is not None and abs(t - source.drop_km2) < 1e-12:
                first_drop[inv] = first
            n = int((c >= 0).sum())
            drains.append({"inventory": inv, "dem": name, "threshold_km2": t,
                           "is_drop_threshold": source.drop_km2 is not None
                           and abs(t - source.drop_km2) < 1e-12,
                           "n_buildings": n, "n_reach_stream": int((first >= 0).sum()),
                           "n_drains_main": int(on_main.sum()),
                           "share_drains_main": float(on_main.sum() / n) if n else np.nan,
                           "routing_gate_pass": source.gate_pass})
            if stored is not None and stored_vals is not None:
                ref = F.sample(stored_vals, c).astype(np.float32)
                mine = hand.astype(np.float32)
                same_nan = np.isnan(ref) == np.isnan(mine)
                ok = ~np.isnan(ref) & ~np.isnan(mine)
                checks.append({"inventory": inv, "dem": name, "threshold_km2": t,
                               "stored": str(stored.relative_to(source.interim)),
                               "n": int(len(c)), "nan_mismatch": int((~same_nan).sum()),
                               "max_abs_diff_m": float(np.max(np.abs(ref[ok] - mine[ok])))
                               if ok.any() else np.nan,
                               "identical": bool(same_nan.all() and np.array_equal(
                                   ref[ok], mine[ok]))})
    if features:
        slope = F.horn_slope_deg(r["z_raw"], grid.res)
        plan, profile = F.zt_curvatures(r["z_raw"], grid.res)
        tpis = {rad: F.tpi(r["z_raw"], grid.res, rad) for rad in TPI_RADII_M}
        twi = F.twi(r["acc"], grid, slope)
        area_km2 = r["acc"] * grid.cell_area_m2 / 1e6
        for inv, c in cells.items():
            cols[inv]["slope_deg"] = F.sample(slope, c)
            cols[inv]["curv_plan"] = F.sample(plan, c)
            cols[inv]["curv_profile"] = F.sample(profile, c)
            for rad, values in tpis.items():
                cols[inv][f"tpi_{rad:g}"] = F.sample(values, c)
            cols[inv]["twi"] = F.sample(twi, c)
            cols[inv]["log10_area_cell_km2"] = np.log10(F.sample(area_km2, c))
            cols[inv]["log10_area_first_drop_km2"] = np.log10(F.sample(area_km2,
                                                                       first_drop[inv]))
            cols[inv]["log10_area_main_km2"] = np.log10(F.sample(area_km2, first_main[inv]))
    return cols, drains, checks, info


# --- base tables (v9 order, centroids, distance, blocks) ------------------------------------


def base_tables(project: Project, cfg: dict[str, Any], v9: DemSource
                ) -> tuple[dict[str, pd.DataFrame], dict[str, gpd.GeoDataFrame],
                           dict[str, Any]]:
    """v9 predictors per inventory (``flood.validate.predictors``) after checking that it
    reads the pinned HAND table, plus an independent recomputation of the centroids and the
    distance to the channel, the join check (primary hand.tif re-sampled at the live
    centroids) and the comparison with the promoted ``hand/building_tiers_*.csv``."""
    import shapely

    from nakkhu.flood import validate as V
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings
    from nakkhu.splits import read_lines

    settings, _ = load_settings(project, cfg["sweep_config"])
    extent = settings.read_polygon(settings.v["analysis_extent"])
    greenhouses = load_greenhouses(settings)
    lines = shapely.union_all(read_lines([project.paths.resolve(V.CHANNEL_LINES)]))
    tiers_meta = json.loads(project.paths.resolve(cfg["building_tiers"]).read_text(
        encoding="utf-8"))
    primary = read_raster(v9.path(f"{v9.combo}__tdrop", "hand.tif")).astype(np.float32).ravel()
    out, frames, checks = {}, {}, {}
    for inv in cfg["inventories"]:
        pin = cfg["hand_tables"][inv]
        pinned = project.paths.runs / pin["run"] / f"building_hand_{inv}.parquet"
        got = sha256(pinned)
        live = V.hand_table_path(project, inv)
        c: dict[str, Any] = {"hand_table": f"runs/{pin['run']}/{pinned.name}",
                             "hand_table_sha256": got, "pinned_sha256": pin["sha256"],
                             "validate_reads": str(live.relative_to(project.paths.root))
                             if live else None}
        if got != pin["sha256"]:
            raise S.FollowupError(f"{inv}: pinned HAND table sha256 {got[:12]} != "
                                  f"{pin['sha256'][:12]}")
        if live is None or live.resolve() != pinned.resolve():
            raise S.FollowupError(f"{inv}: flood.validate would read {live}, not the pinned "
                                  f"{pinned}")
        pred = V.predictors(project, inv, cfg["sweep_config"])
        frame, _ = load_inventory(settings, inv, extent, greenhouses)
        cent = frame.geometry.centroid
        x, y = cent.x.to_numpy(), cent.y.to_numpy()
        dist = shapely.distance(shapely.points(x, y), lines)
        c["n_buildings"] = len(pred)
        c["uid_order_equal"] = bool((frame["building_uid"].to_numpy()
                                     == pred["building_uid"].to_numpy()).all())
        c["centroids_identical"] = bool(np.array_equal(x, pred["x"].to_numpy())
                                        and np.array_equal(y, pred["y"].to_numpy()))
        c["distance_identical"] = bool(np.array_equal(dist, pred["dist_channel_m"].to_numpy()))
        c["distance_sha256"] = array_sha(pred["dist_channel_m"].to_numpy(dtype=np.float64))
        c["distance_recomputed_sha256"] = array_sha(dist.astype(np.float64))
        cells = F.point_cells(v9.grid, x, y)
        resampled = F.sample(primary, cells).astype(np.float32)
        table = pred[S.V9_COLUMN].to_numpy(dtype=np.float32)
        ok = np.isfinite(resampled) & np.isfinite(table)
        c["join_check"] = {"surface": f"{v9.combo}__tdrop/hand.tif",
                           "max_abs_diff_m": float(np.max(np.abs(resampled[ok] - table[ok])))
                           if ok.any() else None,
                           "nan_mismatch": int((np.isnan(resampled) != np.isnan(table)).sum()),
                           "n": int(len(table))}
        meta = tiers_meta["inventories"].get(inv) or {}
        tiers_csv = project.paths.artifacts / "hand" / meta.get("file", f"building_tiers_{inv}.csv")
        if tiers_csv.is_file():
            t = pd.read_csv(tiers_csv)
            same = len(t) == len(pred) and (t["building_id"].to_numpy()
                                            == pred["building_uid"].to_numpy()).all()
            c["promoted_building_tiers"] = {
                "file": f"artifacts/hand/{tiers_csv.name}", "rows_and_ids_equal": bool(same),
                "max_abs_dx_m": float(np.max(np.abs(t["x"] - x))) if same else None,
                "max_abs_dy_m": float(np.max(np.abs(t["y"] - y))) if same else None,
                "max_abs_dhand_m": float(np.nanmax(np.abs(t["hand_m"] - table))) if same
                else None}
        for key in ("uid_order_equal", "centroids_identical", "distance_identical"):
            if not c[key]:
                raise S.FollowupError(f"{inv}: {key} failed (validate.predictors vs the "
                                      "independent recomputation)")
        if c["join_check"]["nan_mismatch"] or (c["join_check"]["max_abs_diff_m"] or 0) > 0:
            raise S.FollowupError(f"{inv}: HAND-table join check failed {c['join_check']}")
        dcell = shapely.distance(shapely.points(F.cell_centres(v9.grid, cells)), lines)
        core = pd.DataFrame({"building_uid": pred["building_uid"].to_numpy(),
                             "x": x, "y": y, "block_id": pred["block_id"].to_numpy(),
                             "cell_v9": cells, "hand_v9": table.astype(np.float64),
                             "distance": pred["dist_channel_m"].to_numpy(dtype=np.float64),
                             "dist_cell": np.where(cells >= 0, dcell, np.nan),
                             "area_m2": pred["area_m2"].to_numpy(dtype=np.float64)})
        out[inv], frames[inv], checks[inv] = core, frame, c
    return out, frames, checks


# --- team HAND (P1-9 / P1-10) -----------------------------------------------------------------


def team_zip(project: Project, cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    path = project.paths.resolve(cfg["team"]["results_zip"])
    want = (spec.get("team_hand_data") or {}).get("files") or {}
    if path.name not in want:
        raise S.FollowupError(f"{path.name} is not declared in the addendum team_hand_data")
    got = sha256(path)
    if got != want[path.name]:
        raise S.FollowupError(f"{path.name} sha256 {got[:12]} != declared {want[path.name][:12]}")
    for name, digest in want.items():            # every declared team zip must match
        other = path.parent / name
        if not other.is_file() or sha256(other) != digest:
            raise S.FollowupError(f"team file {name} missing or altered")
    return path


def read_team_raster(zip_path: Path, member: str) -> tuple[np.ndarray, GridSpec]:
    import rasterio

    with rasterio.open(S.vsizip(zip_path, member)) as ds:
        values = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            values[values == ds.nodata] = np.nan
        grid = GridSpec(crs=ds.crs.to_string(), transform=ds.transform, width=ds.width,
                        height=ds.height)
    values[~np.isfinite(values)] = np.nan
    return values, grid


def read_team_vector(zip_path: Path, member: str, crs: str) -> gpd.GeoDataFrame:
    import shapely

    frame = gpd.read_file(S.vsizip(zip_path, member)).to_crs(crs)
    frame["geometry"] = shapely.force_2d(frame.geometry.values)
    return frame


def team_columns(project: Project, cfg: dict[str, Any], spec: dict[str, Any],
                 sources: dict[str, DemSource], cores: dict[str, pd.DataFrame],
                 burn_lines: gpd.GeoDataFrame
                 ) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]],
                            list[dict[str, Any]], dict[str, Any]]:
    from nakkhu.hydro import condition as cond

    tcfg = cfg["team"]
    zpath = team_zip(project, cfg, spec)
    base = tcfg["members_dir"]
    crs = sources["glo30"].grid.crs
    hand_team, tgrid = read_team_raster(zpath, f"{base}/{tcfg['hand']}")
    dem125, dgrid = read_team_raster(zpath, f"{base}/{tcfg['dem']}")
    if dgrid != tgrid:
        raise S.FollowupError("team Dem12.5 and Hand Value grids differ")
    basin = read_team_vector(zpath, f"{base}/{tcfg['basin']}", crs)
    digitized = read_team_vector(zpath, f"{base}/{tcfg['digitized']}", crs)
    outlet = read_team_vector(zpath, f"{base}/{tcfg['outlet']}", crs).geometry.iloc[0]
    poly = basin.geometry.union_all()
    clipped = burn_lines.copy()
    clipped["geometry"] = clipped.geometry.intersection(poly)
    clipped = clipped[~clipped.geometry.is_empty]
    edges = S.tier_edges(spec)
    out: dict[str, dict[str, np.ndarray]] = {inv: {} for inv in cores}
    pts = {inv: (c["x"].to_numpy(), c["y"].to_numpy()) for inv, c in cores.items()}
    tcells = {inv: F.point_cells(tgrid, x, y) for inv, (x, y) in pts.items()}
    for inv in cores:
        x, y = pts[inv]
        out[inv]["hand_team"] = F.sample(hand_team, tcells[inv])
        out[inv]["team_basin"] = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=crs).within(
            poly).to_numpy().astype(float)
        for key, member in tcfg["masks"].items():
            mask, mgrid = read_team_raster(zpath, f"{base}/{member}")
            out[inv][f"team_mask_{key}"] = F.sample(mask, F.point_cells(mgrid, x, y))
    team_fdir, _ = read_team_raster(zpath, f"{base}/{tcfg['flow_direction']}")
    team_streams, _ = read_team_raster(zpath, f"{base}/{tcfg['streams']}")
    wbt = cond.whitebox_tools()
    v = sources["glo30"].settings.v
    depth, max_dist = float(v["burn"]["depth_m"]), int(v["breach"]["max_dist_cells"])
    area = float(tcfg["stream_area_km2"])
    recipe_rows: list[dict[str, Any]] = []
    repro: list[dict[str, Any]] = []
    for dem in tcfg["dems"]:
        if dem == "dem12.5":
            grid, z = tgrid, dem125
            bmask = grid.mask(poly) & np.isfinite(z)
        else:
            src = sources[dem]
            grid = src.grid
            z = read_raster(src.path("grid", f"dem_{src.dem}.tif")).astype(np.float64)
            bmask = grid.mask(poly) & np.isfinite(z)
        burn = cond.burn_mask(clipped, grid, all_touched=True) if len(clipped) else \
            np.zeros(grid.shape, dtype=bool)
        dig = cond.burn_mask(digitized, grid, all_touched=True)
        cells = {inv: F.point_cells(grid, *pts[inv]) for inv in cores}
        outlet_cell = nearest_basin_cell(grid, bmask, outlet)
        for conditioning in tcfg["conditionings"]:
            res = F.team_recipe(z, grid, bmask, conditioning, burn=burn, burn_depth_m=depth,
                                max_dist_cells=max_dist, stream_area_km2=area, digitized=dig,
                                outlet=outlet_cell, wbt=wbt,
                                workdir=project.paths.data_root / "interim")
            tag = f"{dem}__{conditioning}"
            for inv in cores:
                for kind in ("acc", "dig"):
                    out[inv][f"hand_team_{kind}__{tag}"] = F.sample(res[f"hand_{kind}"],
                                                                    cells[inv])
            ox, oy = F.cell_centres(grid, np.array([outlet_cell]))[0]
            outlet_km2 = float(res["acc"].ravel()[outlet_cell] * grid.cell_area_m2 / 1e6)
            row = {"dem": dem, "conditioning": conditioning, "res_m": grid.res,
                   "basin_cells": int(bmask.sum()),
                   "basin_km2": float(bmask.sum() * grid.cell_area_m2 / 1e6),
                   "stream_cells": int(res["streams"].sum()),
                   "digitized_cells": int((dig & bmask).sum()),
                   "burn_cells": int((burn & bmask).sum()) if conditioning == "breach_burn"
                   else 0,
                   "outlet_cell_offset_m": float(np.hypot(ox - outlet.x, oy - outlet.y)),
                   "outlet_area_km2": outlet_km2,
                   "n_flat_or_pit_cells": int(np.isin(res["fdir"][bmask], (-1, -2)).sum()),
                   "hand_acc_finite_share": float(np.isfinite(res["hand_acc"][bmask]).mean()),
                   "hand_dig_finite_share": float(np.isfinite(res["hand_dig"][bmask]).mean())}
            if dem == "dem12.5":
                both = bmask & np.isfinite(team_fdir)
                row["share_fdir_equal_team"] = float((res["fdir"][both] == team_fdir[both])
                                                     .mean())
                ts = np.nan_to_num(team_streams, nan=0.0) > 0
                row["team_stream_cells"] = int(ts.sum())
                row["share_team_streams_in_ours"] = float(res["streams"][ts].mean()) \
                    if ts.any() else np.nan
                row["share_our_streams_in_team"] = float(ts[res["streams"]].mean()) \
                    if res["streams"].any() else np.nan
                grid_agree = F.tier_agreement(res["hand_acc"][bmask], hand_team[bmask], edges)
                repro.append({"scope": "all basin cells", "inventory": "", "member":
                              f"hand_team_acc__{tag}", **grid_agree})
                for inv in cores:
                    covered = np.isfinite(out[inv]["hand_team"])
                    agree = F.tier_agreement(out[inv][f"hand_team_acc__{tag}"][covered],
                                             out[inv]["hand_team"][covered], edges)
                    repro.append({"scope": "covered building cells", "inventory": inv,
                                  "member": f"hand_team_acc__{tag}", **agree})
            recipe_rows.append(row)
    info = {"team_zip": zpath.name, "team_grid": tgrid.to_dict(),
            "basin_area_km2": float(poly.area / 1e6),
            "digitized_length_km": float(digitized.geometry.length.sum() / 1e3),
            "burn_lines_in_basin": len(clipped), "stream_area_km2": area,
            "breach_max_dist_cells": max_dist,
            "primary_reproduction_member": "hand_team_acc__dem12.5__fill"}
    return out, recipe_rows, repro, info


def nearest_basin_cell(grid: GridSpec, basin: np.ndarray, point: Any) -> int:
    """Flat index of the basin cell whose centre is nearest ``point``."""
    cells = np.flatnonzero(np.asarray(basin, dtype=bool).ravel())
    xy = F.cell_centres(grid, cells)
    return int(cells[np.argmin(np.hypot(xy[:, 0] - point.x, xy[:, 1] - point.y))])


def team_coverage(cores: dict[str, pd.DataFrame], team: dict[str, dict[str, np.ndarray]]
                  ) -> list[dict[str, Any]]:
    rows = []
    for inv, core in cores.items():
        valid = np.isfinite(team[inv]["hand_team"])
        inside = team[inv]["team_basin"] > 0
        blocks = core["block_id"].astype(str).to_numpy()
        rows.append({"inventory": inv, "n_buildings": len(core),
                     "n_valid_hand_team": int(valid.sum()),
                     "share_valid_hand_team": float(valid.mean()),
                     "n_centroid_in_team_basin": int(inside.sum()),
                     "n_blocks": int(len(set(blocks))),
                     "n_blocks_with_covered_building": int(len(set(blocks[valid]))),
                     "note": "all buildings of the analysis extent (no observability or "
                             "label used); P1-9 reports coverage on the R1 rows"})
    return rows


# --- UNOSAT integrity (no building intersection) ---------------------------------------------


METADATA_KEYWORDS = ("DEM", "HAND", "elevation", "slope", "height", "terrain", "mask")


def unosat_integrity(project: Project, cfg: dict[str, Any], extent: Any,
                     cores: dict[str, pd.DataFrame], crs: str) -> dict[str, Any]:
    import re

    from nakkhu.manifest import load_manifest

    ucfg = cfg["unosat"]
    entries = {e["id"]: e for e in load_manifest(project.paths.resolve(ucfg["manifest"]))}
    entry = entries.get(ucfg["id"])
    if entry is None:
        raise S.FollowupError(f"{ucfg['manifest']} has no entry {ucfg['id']}")
    path = project.paths.resolve(entry["local_path"])
    got = sha256(path)
    out: dict[str, Any] = {"manifest": ucfg["manifest"], "id": ucfg["id"],
                           "file": entry["local_path"], "sha256": got,
                           "sha256_matches_manifest": got == entry["sha256"],
                           "bytes": path.stat().st_size, "licence": entry["licence"],
                           "layers": {}}
    if got != entry["sha256"]:
        raise S.FollowupError(f"UNOSAT zip sha256 {got[:12]} != manifest {entry['sha256'][:12]}")
    with zipfile.ZipFile(path) as zf:
        out["members"] = len(zf.namelist())
        bad = zf.testzip()
        out["zip_crc_ok"] = bad is None
        for key, layer in ucfg["layers"].items():
            member = f"{ucfg['members_dir']}/{layer}.shp"
            frame = gpd.read_file(S.vsizip(path, member))
            info: dict[str, Any] = {"member": member, "crs": str(frame.crs),
                                    "features": len(frame),
                                    "geometry_types": sorted(frame.geom_type.unique().tolist()),
                                    "valid_share": float(frame.geometry.is_valid.mean()),
                                    "bounds_lonlat": [float(v) for v in
                                                      frame.to_crs("EPSG:4326").total_bounds]}
            if key in ("flood_extent", "analysis_extent"):
                info["area_km2_whole_layer"] = float(frame.to_crs(crs).area.sum() / 1e6)
            if key == "analysis_extent":
                ae = frame.to_crs(crs).geometry.union_all()
                info["share_of_nakkhu_extent_inside"] = float(extent.intersection(ae).area
                                                              / extent.area)
                for inv, core in cores.items():
                    pts = gpd.GeoSeries(gpd.points_from_xy(core["x"], core["y"]), crs=crs)
                    info[f"n_{inv}_centroids_inside"] = int(pts.within(ae).sum())
            xml = f"{member}.xml"
            if xml in zf.namelist():
                text = zf.read(xml).decode("utf-8", errors="replace")
                plain = re.sub(r"<[^>]+>", " ", text)
                plain = re.sub(r"&lt;[^&]*&gt;", " ", plain)
                hits = []
                for word in METADATA_KEYWORDS:
                    for m in re.finditer(rf"\b{re.escape(word)}\b", plain, flags=re.I):
                        hits.append(" ".join(plain[max(0, m.start() - 100):m.end() + 100]
                                             .split()))
                info["metadata_keyword_snippets"] = sorted(set(hits))[:20]
            out["layers"][key] = info
    out["note"] = ("integrity, CRS and extent only; no building was intersected with the flood "
                   "extent or the affected-building points (that is P1-8)")
    return out


# --- driver ---------------------------------------------------------------------------------


GROUP_FILES = "prep_{group}_{inventory}.parquet"


def run_prep(project: Project, config: str = S.PREP_CONFIG) -> tuple[Any, dict[str, Any]]:
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, create_run

    cfg = S.load_prep_config(project, config)
    spec = S.load_spec(project, cfg["addendum"])
    resolved = ResolvedConfig(values={"command": "followup-prep", "config": config,
                                      "config_sha256": sha256(project.paths.resolve(config)),
                                      "addendum": cfg["addendum"]}, source=config)
    sources = {name: DemSource(project, name, entry) for name, entry in cfg["dems"].items()}
    v9 = sources[cfg.get("feature_dem", "glo30")]
    run = create_run(project, S.PREP_STAGE, resolved)
    files: list[str] = []
    summary: dict[str, Any] = {"addendum": cfg["addendum"], "dems": {}}

    def write_table(frame: pd.DataFrame, name: str) -> None:
        frame.to_parquet(run.path / name, index=False, compression="zstd")
        files.append(name)

    def write_csv(rows: list[dict[str, Any]] | pd.DataFrame, name: str) -> None:
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(run.path / name, index=False, lineterminator="\n")
        files.append(name)

    with run.step("base"):
        cores, frames, checks = base_tables(project, cfg, v9)
    points = {inv: (c["x"].to_numpy(), c["y"].to_numpy()) for inv, c in cores.items()}
    lines = main_lines(project, spec, v9.grid.crs)
    summary["main_lines"] = len(lines)
    drains: list[dict[str, Any]] = []
    surface_checks: list[dict[str, Any]] = []
    groups: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, src in sources.items():
        with run.step(f"dem-{name}"):
            cols, d, sc, info = dem_columns(src, spec, lines, points,
                                            features=name == cfg.get("feature_dem", "glo30"))
            drains += d
            surface_checks += sc
            groups[name] = cols
            summary["dems"][name] = {**src.meta(), **info}
    feature_dem = cfg.get("feature_dem", "glo30")
    with run.step("features"):
        feature_names = ("slope_deg", "curv_plan", "curv_profile", "tpi_150", "tpi_500", "twi",
                         "log10_area_cell_km2", "log10_area_first_drop_km2",
                         "log10_area_main_km2")
        for inv, core in cores.items():
            g = groups[feature_dem][inv]
            feats = {"building_uid": core["building_uid"].to_numpy()}
            for k in feature_names:
                feats[k] = g.pop(k)
            feats["elev_glo30"] = g[f"elev__{feature_dem}"]
            feats["elev_fabdem"] = groups["fabdem"][inv]["elev__fabdem"]
            feats["dsm_excess"] = feats["elev_glo30"] - feats["elev_fabdem"]
            feats["area_m2"] = core["area_m2"].to_numpy()
            feats["density_100m"] = F.count_within(core["x"].to_numpy(), core["y"].to_numpy(),
                                                   DENSITY_RADIUS_M).astype(float)
            write_table(pd.DataFrame(feats), GROUP_FILES.format(group="features", inventory=inv))
    for inv, core in cores.items():
        write_table(core, GROUP_FILES.format(group="core", inventory=inv))
        for name in sources:
            frame = pd.DataFrame({"building_uid": core["building_uid"].to_numpy(),
                                  **groups[name][inv]})
            write_table(frame, GROUP_FILES.format(group=name, inventory=inv))
    with run.step("team"):
        from nakkhu.hydro import condition as cond

        burn_lines = cond.read_lines([project.paths.resolve(p)
                                      for p in v9.settings.v["burn"]["lines"]], v9.grid.crs)
        team, recipe, repro, tinfo = team_columns(project, cfg, spec, sources, cores,
                                                  burn_lines)
        for inv, core in cores.items():
            write_table(pd.DataFrame({"building_uid": core["building_uid"].to_numpy(),
                                      **team[inv]}),
                        GROUP_FILES.format(group="team", inventory=inv))
        write_csv(recipe, "prep_team_recipe.csv")
        write_csv(repro, "prep_team_reproduction.csv")
        write_csv(team_coverage(cores, team), "prep_team_coverage.csv")
        summary["team"] = tinfo
    with run.step("unosat"):
        from nakkhu.hydro.sweep import load_settings

        settings, _ = load_settings(project, cfg["sweep_config"])
        extent = settings.read_polygon(settings.v["analysis_extent"])
        un = unosat_integrity(project, cfg, extent, cores, v9.grid.crs)
        atomic_write_json(run.path / "prep_unosat.json", un)
        files.append("prep_unosat.json")
    write_csv(drains, "prep_drain_shares.csv")
    write_csv(surface_checks, "prep_surface_checks.csv")
    glo = summary["dems"][feature_dem]["burn_cells_below_threshold"]
    clarifications = {
        "hand_any_t": ("streams = contributing area >= t only, exactly as the sweep code "
                       "(hydro.hand.stream_mask); the burn conditions the routing DEM only. "
                       "The addendum's wording 'plus the burned lines' is not implemented, so "
                       "hand_any at the 0.2 km2 drop threshold equals hand_v9 (coordinator "
                       "decision 2026-10-17). Burned cells below the threshold on GLO-30 per "
                       "threshold: " + json.dumps(glo)),
        "density_100m": "count of other centroids of the same inventory within 100 m (self "
                        "excluded)",
        "team_outlet": "definitions.hand_team_acc: the outlet = the basin cell (centre) nearest "
                       "the team's Nakkhu_Outlet point; its routing elevation is set 1 m below "
                       "the basin minimum before fill / breach so the clipped basin drains (and "
                       "its flats resolve) to it, as ArcGIS lets the outlet drain off the "
                       "raster; its HAND reference is restored like a burned cell. Without it "
                       "pysheds left the int16 outlet flats unresolved (8 % of Dem12.5 basin "
                       "cells never reached a stream).",
        "hma8m": "not prepared (coordinator 2026-10-17); P1-4 lists it as not ready"}
    all_identical = all(r["identical"] for r in surface_checks)
    checks_doc = {"inventories": checks, "stored_surfaces_identical": all_identical,
                  "clarifications": clarifications}
    atomic_write_json(run.path / "prep_checks.json", checks_doc)
    files.append("prep_checks.json")
    manifest = {"run": run.ref, "addendum": cfg["addendum"],
                "config": config, "config_sha256": resolved.values["config_sha256"],
                "inventories": list(cfg["inventories"]),
                "groups": ["core", "features", "team", *sources],
                "dems": {k: v for k, v in summary["dems"].items()},
                "files": {name: sha256(run.path / name) for name in files},
                "inputs": {"hand_tables": {inv: c["hand_table_sha256"]
                                           for inv, c in checks.items()},
                           "unosat_zip": un["sha256"], "team_zip": tinfo["team_zip"]}}
    atomic_write_json(run.path / "prep_manifest.json", manifest)
    run.write_summary({"files": len(files), "stored_surfaces_identical": all_identical,
                       "main_lines": summary["main_lines"]})
    for name in [*files, "prep_manifest.json"]:
        promote(run.path / name, f"{S.ARTIFACT_DIR}/{name}", project.paths, replace=True)
    return run, {**summary, "checks": checks_doc, "manifest": manifest}


# --- loading the promoted prep (runner) ---------------------------------------------------------


def load_prep(project: Project) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """The promoted prep tables per inventory (all groups joined in v9 order) and the prep
    manifest; refuses if any promoted file differs from the manifest's sha256."""
    art = project.paths.artifacts
    path = art / S.PREP_MANIFEST
    if not path.is_file():
        raise S.FollowupError(f"artifacts/{S.PREP_MANIFEST} missing (run `nakkhu followup prep`)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        f = art / S.ARTIFACT_DIR / name
        if not f.is_file() or sha256(f) != digest:
            raise S.FollowupError(f"artifacts/{S.ARTIFACT_DIR}/{name} missing or altered "
                                  "since the prep run")
    tables = {}
    for inv in manifest["inventories"]:
        parts = [pd.read_parquet(art / S.ARTIFACT_DIR / GROUP_FILES.format(group=g,
                                                                           inventory=inv))
                 for g in manifest["groups"]]
        uid = parts[0]["building_uid"].to_numpy()
        for p in parts[1:]:
            if not (p["building_uid"].to_numpy() == uid).all():
                raise S.FollowupError(f"{inv}: prep groups are not in the same building order")
        merged = parts[0]
        for p in parts[1:]:
            p = p.drop(columns="building_uid")
            dup = [c for c in p.columns if c in merged.columns]
            for c in dup:                      # e.g. area_m2 in core and features
                a = merged[c].to_numpy(dtype=float)
                b = p[c].to_numpy(dtype=float)
                if not np.array_equal(a, b, equal_nan=True):
                    raise S.FollowupError(f"{inv}: column {c} differs between prep groups")
            merged = pd.concat([merged, p.drop(columns=dup)], axis=1)
        tables[inv] = merged
    return tables, manifest
