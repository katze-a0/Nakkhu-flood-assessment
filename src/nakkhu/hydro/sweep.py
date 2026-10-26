"""Plan 008 runs: the HAND sweep (``nakkhu hand sweep`` / ``hand run``) and building attribution
(``nakkhu hand attribute``).

Sweep steps (journalled in ``runs/hand/<utc>-<cfghash>/``):

1. ``prepare``: warp every available DEM onto the common grid; units guard against GLO-30.
2. ``routing``: per DEM x conditioning, burn (if any), condition, D8 directions, accumulation.
3. ``domain``: rebuild the upper-Bagmati routing domain from burned GLO-30; compare with the
   unburned delineations.
4. ``burn_check``: share of the OSM Nakkhu channel band inside the delineated Nakkhu catchment.
5. ``drop_analysis``: Tarboton threshold per DEM x conditioning.
6. ``surfaces``: HAND per DEM x conditioning x threshold, units check (STOP if it fails).
7. ``agreement`` (sweep only): pysheds vs WhiteboxTools vs pyflwdir on a seeded draw, and
   GLO-30 surfaces vs ASF GLO-30 HAND.

Rasters go to ``{data_root}/interim/hand/<cfghash>/`` (gitignored); tables to the run directory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

from nakkhu import hazard
from nakkhu.config import Project, ResolvedConfig, load_config, load_yaml
from nakkhu.hydro import condition as cond
from nakkhu.hydro import dem as demmod
from nakkhu.hydro import drop_analysis, route
from nakkhu.hydro import hand as handmod
from nakkhu.hydro.dem import GridSpec
from nakkhu.manifest import load_manifest
from nakkhu.runs import Run, atomic_write_json, atomic_write_text, create_run, list_runs

SWEEP_CONFIG = "configs/hand/sweep.yaml"
STAGE = "hand"
ATTRIBUTE_STAGE = "hand-attribute"
DROP = "drop"


class SweepError(RuntimeError):
    """The sweep cannot proceed (missing reference DEM, failed STOP condition, ...)."""


@dataclass(frozen=True)
class Surface:
    dem: str
    conditioning: str
    threshold: str  # "0.05", ..., or "drop"

    @property
    def combo(self) -> str:
        return f"{self.dem}__{self.conditioning}"

    @property
    def cfg_id(self) -> str:
        return f"{self.combo}__t{self.threshold.replace('.', 'p')}"


def threshold_label(value: float) -> str:
    return f"{value:g}"


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> Path:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))
    return path


# --- settings -------------------------------------------------------------------------------


class Settings:
    """The sweep config plus project paths and the data manifest."""

    def __init__(self, project: Project, config: ResolvedConfig):
        self.project = project
        self.config = config
        self.v = config.values
        self.crs = self.v["grid"]["crs"]
        manifest_path = (project.config.values.get("paths") or {}).get(
            "manifest", "configs/data/manifest.yaml")
        self.manifest = {e["id"]: e for e in load_manifest(project.paths.resolve(manifest_path))}

    def resolve(self, value: str) -> Path:
        return self.project.paths.resolve(value)

    @property
    def interim(self) -> Path:
        return self.project.paths.data_root / "interim" / "hand" / self.config.cfghash

    def read_polygon(self, key_or_path: str) -> Any:
        frame = gpd.read_file(self.resolve(key_or_path)).to_crs(self.crs)
        return unary_union(frame.geometry)

    def grid(self) -> GridSpec:
        g = self.v["grid"]
        domain = self.read_polygon(g["domain"])
        return demmod.grid_for_bounds(domain.bounds, float(g["res_m"]), float(g["buffer_m"]),
                                      g["crs"])

    def dem_status(self, name: str) -> tuple[str, str]:
        entry = self.v["dems"][name]
        mid = entry.get("manifest_id")
        if not mid:
            return "skipped", "not registered in the data manifest (access on request)"
        record = self.manifest.get(mid)
        if record is None:
            return "skipped", f"manifest has no entry {mid}"
        if record.get("status") == "pending":
            return "skipped", f"pending: needs user account (manifest {mid})"
        if not self.resolve(entry["path"]).is_file():
            return "skipped", f"file missing: {entry['path']}"
        unharmonised = self.datum_problem(entry)
        if unharmonised:
            return "skipped", unharmonised
        return "available", ""

    def datum_problem(self, entry: dict[str, Any]) -> str:
        """Why a DEM may not join this sweep's vertical datum ("" if it may). A DEM without a
        ``vertical`` conversion key is used as-is, so its documented ``vertical_datum`` must be
        the ``vertical_target``; otherwise it is skipped. This keeps the canonical sweep
        (``configs/hand/sweep.yaml``, no ``vertical`` keys) from mixing EGM96 DEMs into its
        EGM2008 grid once they are fetched; ``sweep_dems.yaml`` converts them."""
        if "vertical" in entry:
            return ""
        text = str(entry.get("vertical_datum") or "").strip()
        if text and not text.lower().startswith(self.vertical_target.lower()):
            return (f"vertical datum {text!r} is not the sweep target {self.vertical_target} "
                    "and the entry has no `vertical` conversion")
        return ""

    def available_dems(self) -> list[str]:
        return [n for n in self.v["dems"] if self.dem_status(n)[0] == "available"]

    def conditionings(self) -> dict[str, dict[str, Any]]:
        return self.v["conditioning"]

    def fixed_thresholds(self) -> list[str]:
        return [threshold_label(t) for t in self.v["thresholds_km2"]]

    def all_thresholds(self) -> list[str]:
        return [*self.fixed_thresholds(), DROP]

    @property
    def edges(self) -> list[float]:
        return [float(e) for e in self.v["tiers_m"]]

    def primary(self) -> Surface:
        p = self.v["primary"]
        return Surface(p["dem"], p["conditioning"], str(p["threshold"]))

    @property
    def stage(self) -> str:
        return str(self.v.get("stage", STAGE))

    @property
    def attribute_stage(self) -> str:
        return str(self.v.get("attribute_stage", ATTRIBUTE_STAGE))

    @property
    def vertical_target(self) -> str:
        return str(self.v.get("vertical_target", "egm2008"))

    def geoid_paths(self) -> dict[str, Path]:
        """Geoid grids (``geoids: {datum: {manifest_id, path}}``) that are fetched."""
        out = {}
        for name, entry in (self.v.get("geoids") or {}).items():
            record = self.manifest.get(entry.get("manifest_id"))
            path = self.resolve(entry["path"])
            if record is not None and record.get("status") != "pending" and path.is_file():
                out[name] = path
        return out


EXTENSION_SECTIONS = ("dems", "conditioning", "inventories")


def extend_config(base: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    """A sweep config derived from ``base`` (which stays untouched, e.g. hash-pinned).

    Every top-level key of ``extension`` (except ``extends`` and ``only``) replaces the base
    value, except the mappings :data:`EXTENSION_SECTIONS`, whose entries are added to (or
    replace, entry by entry) the base entries. Then ``only: {section: [keys]}`` keeps just those
    keys of ``section``, in that order. The ``extends`` path stays in the values, so it enters
    the config hash."""
    merged = {k: v for k, v in base.items()}
    for key, value in extension.items():
        if key == "only":
            continue
        if key in EXTENSION_SECTIONS:
            merged[key] = {**(base.get(key) or {}), **(value or {})}
        else:
            merged[key] = value
    for section, keys in (extension.get("only") or {}).items():
        if section not in EXTENSION_SECTIONS:
            raise SweepError(f"`only` supports {EXTENSION_SECTIONS}, not {section!r}")
        missing = [k for k in keys if k not in merged[section]]
        if missing:
            raise SweepError(f"`only.{section}`: unknown keys {missing}")
        merged[section] = {k: merged[section][k] for k in keys}
    merged["only"] = extension.get("only") or {}
    return merged


def _extended_values(project: Project, path: str, chain: tuple[str, ...] = ()) -> dict[str, Any]:
    """Values of ``path`` with its ``extends`` chain applied (base first)."""
    if path in chain:
        raise SweepError(f"circular `extends`: {' -> '.join((*chain, path))}")
    values = load_yaml(project.paths.resolve(path))
    if "extends" not in values:
        return values
    return extend_config(_extended_values(project, values["extends"], (*chain, path)), values)


def load_settings(project: Project, path: str = SWEEP_CONFIG) -> tuple[Settings, Surface | None]:
    """Sweep settings; a single-configuration file (with ``base``) also returns its surface, and
    a file with ``extends`` is a derived sweep (:func:`extend_config`)."""
    values = load_yaml(project.paths.resolve(path))
    if "extends" in values:
        merged = _extended_values(project, path)
        return Settings(project, ResolvedConfig(values=merged, source=project.paths.relative(
            project.paths.resolve(path)))), None
    if "base" not in values:
        return Settings(project, load_config(project.paths.resolve(path))), None
    base = load_config(project.paths.resolve(values["base"]))
    merged = dict(base.values)
    single = {k: values[k] for k in ("dem", "conditioning", "threshold")}
    merged["run"] = single
    resolved = ResolvedConfig(values=merged, source=project.paths.relative(
        project.paths.resolve(path)))
    surface = Surface(single["dem"], single["conditioning"], str(single["threshold"]))
    return Settings(project, resolved), surface


# --- sweep steps ----------------------------------------------------------------------------


@dataclass
class Combo:
    dem: str
    conditioning: str
    routing_dem: np.ndarray  # conditioned (burned) elevations used for D8
    reference: np.ndarray  # conditioned elevations, burn undone (HAND z)
    fdir: np.ndarray
    acc: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.dem}__{self.conditioning}"


class SweepRunner:
    def __init__(self, settings: Settings, run: Run, *, dems: Iterable[str] | None = None,
                 conditionings: Iterable[str] | None = None,
                 thresholds: Iterable[str] | None = None, agreement: bool = True):
        self.s = settings
        self.run = run
        self.grid = settings.grid()
        self.dems = list(dems) if dems is not None else list(settings.v["dems"])
        self.conds = (list(conditionings) if conditionings is not None
                      else list(settings.conditionings()))
        self.thresholds = (list(thresholds) if thresholds is not None
                           else settings.all_thresholds())
        self.do_agreement = agreement
        self.summary: dict[str, Any] = {"grid": self.grid.to_dict(),
                                        "interim": settings.project.paths.relative(
                                            settings.interim)}
        self.timings: dict[str, float] = {}
        self.wbt = cond.whitebox_tools()
        self.summary["whitebox_version"] = cond.whitebox_version(self.wbt)

    # -- helpers --
    def out(self, name: str) -> Path:
        return self.run.path / name

    def raster_path(self, *parts: str) -> Path:
        return self.s.interim.joinpath(*parts)

    def save(self, array: np.ndarray, *parts: str, nodata: Any = None,
             tags: dict[str, Any] | None = None) -> Path:
        return demmod.write_raster(self.raster_path(*parts), array, self.grid, nodata=nodata,
                                   tags=tags)

    # -- step 1 --
    def prepare(self) -> None:
        s, grid = self.s, self.grid
        self.research_domain = s.read_polygon(s.v["grid"]["domain"])
        self.research_domain_mask = grid.mask(self.research_domain)
        ref_name = s.v["reference_dem"]
        if s.dem_status(ref_name)[0] != "available":
            raise SweepError(f"reference DEM {ref_name} is not available")
        self.prepared: dict[str, np.ndarray] = {}
        rows = []
        order = [ref_name] + [d for d in self.dems if d != ref_name]
        for name in order:
            entry = s.v["dems"][name]
            status, reason = s.dem_status(name)
            row: dict[str, Any] = {"dem": name, "manifest_id": entry.get("manifest_id"),
                                   "status": status, "reason": reason,
                                   "surface": entry.get("surface"),
                                   "vertical_datum": entry.get("vertical_datum"),
                                   "apply_scale": bool(entry.get("apply_scale", False))}
            if status == "available":
                resampling = entry.get("resampling", s.v["grid"]["resampling"])
                array = demmod.prepare_dem(s.resolve(entry["path"]), grid,
                                           resampling=resampling,
                                           apply_scale=bool(entry.get("apply_scale", False)))
                row["resampling"] = resampling
                row.update(self.harmonise_vertical(name, entry, array))
                row["nan_in_grid_before_fill"] = int(np.isnan(array).sum())
                donor = entry.get("void_fill")
                if donor:
                    if donor not in self.prepared:
                        raise SweepError(f"{name}: void_fill donor {donor} is not prepared "
                                         "before it (use the reference DEM)")
                    array, filled = demmod.fill_voids(array, self.prepared[donor])
                    self.save(filled.astype(np.uint8), "grid", f"void_filled_{name}.tif",
                              nodata=None)
                    row.update({"void_fill": donor, "void_filled_grid": int(filled.sum()),
                                "void_filled_domain": int(filled[
                                    self.research_domain_mask].sum())})
                median = demmod.check_elevation_units(
                    array, self.prepared.get(ref_name, array), self.research_domain_mask,
                    float(s.v["max_median_abs_diff_m"]))
                dom = array[self.research_domain_mask]
                row.update({
                    "height": grid.height, "width": grid.width,
                    "transform": json.dumps(list(grid.transform)[:6]),
                    "nan_in_grid": int(np.isnan(array).sum()),
                    "nan_in_domain": int(np.isnan(dom).sum()),
                    "min_domain_m": float(np.nanmin(dom)), "max_domain_m": float(np.nanmax(dom)),
                    "median_abs_diff_vs_reference_m": median,
                })
                if row["nan_in_domain"]:
                    raise SweepError(f"{name}: {row['nan_in_domain']} nodata cells in the domain")
                self.prepared[name] = array
                self.save(array, "grid", f"dem_{name}.tif", nodata=np.nan,
                          tags={"vertical_datum": entry.get("vertical_datum"),
                                "source": entry["path"]})
            rows.append(row)
        write_csv(self.out("dem_prep.csv"), rows)
        lines = cond.read_lines([s.resolve(p) for p in s.v["burn"]["lines"]], grid.crs)
        self.burn = cond.burn_mask(lines, grid, all_touched=bool(s.v["burn"]["all_touched"]))
        self.save(self.burn.astype(np.uint8), "grid", "burn_mask.tif", nodata=None)
        index = np.arange(grid.height * grid.width, dtype=np.int32).reshape(grid.shape)
        self.save(index, "grid", "cell_index.tif", nodata=None)
        self.summary["dems"] = {r["dem"]: {"status": r["status"], "reason": r["reason"]}
                                for r in rows}
        self.summary["burn_cells"] = int(self.burn.sum())
        self.summary["burn_lines"] = len(lines)

    def harmonise_vertical(self, name: str, entry: dict[str, Any],
                           array: np.ndarray) -> dict[str, Any]:
        """Convert ``array`` in place from the DEM's ``vertical`` datum to the sweep's
        ``vertical_target`` (default EGM2008, the GLO-30/FABDEM datum); a DEM without
        ``vertical`` is taken to be in the target datum already."""
        source = str(entry.get("vertical", self.s.vertical_target))
        target = self.s.vertical_target
        info: dict[str, Any] = {"vertical_from": source, "vertical_to": target}
        if source == target:
            return info
        geoids = self.s.geoid_paths()
        needed = [d for d in (source, target) if d != "ellipsoid" and d not in geoids]
        if needed:
            raise SweepError(f"{name}: geoid grid(s) {needed} not fetched (configs/data "
                             "sources geoid_*)")
        shift = demmod.geoid_shift(self.grid, source, target, geoids)
        assert shift is not None
        array += shift
        dom = shift[self.research_domain_mask]
        info.update({"vertical_shift_median_m": float(np.median(dom)),
                     "vertical_shift_min_m": float(dom.min()),
                     "vertical_shift_max_m": float(dom.max())})
        return info

    # -- step 2 --
    def route_combo(self, dem: str, conditioning: str) -> Combo:
        spec = self.s.conditionings()[conditioning]
        depth = float(self.s.v["burn"]["depth_m"])
        raw = self.prepared[dem]
        mask = self.burn if spec.get("burn") else None
        source = cond.burn(raw, mask, depth) if mask is not None else raw
        routing = cond.condition(source, self.grid, spec["method"],
                                 max_dist_cells=int(self.s.v["breach"]["max_dist_cells"]),
                                 wbt=self.wbt, workdir=self.s.interim)
        reference = cond.reference_elevation(routing, mask, raw)
        fdir = route.flowdir(routing, self.grid)
        acc = route.accumulation(fdir, self.grid)
        combo = Combo(dem, conditioning, routing, reference, fdir, acc)
        name = combo.name
        self.save(routing, name, "dem_cond.tif", nodata=np.nan)
        self.save(reference, name, "dem_ref.tif", nodata=np.nan)
        self.save(fdir.astype(np.int16), name, "fdir.tif", nodata=0)
        self.save(acc.astype(np.float32), name, "acc.tif", nodata=None)
        return combo

    def check_only(self, conditioning: str) -> bool:
        return bool(self.s.conditionings()[conditioning].get("check_only", False))

    def combos(self) -> list[tuple[str, str]]:
        """DEM x conditioning pairs that produce HAND surfaces."""
        available = set(self.prepared)
        return [(d, c) for d in self.dems if d in available for c in self.conds
                if not self.check_only(c)]

    def routing_pairs(self) -> list[tuple[str, str]]:
        """Surface pairs, plus check-only levels per DEM and what the domain step needs."""
        ref = self.s.v["reference_dem"]
        pairs = self.combos()
        extra = [(d, c) for d in self.dems if d in self.prepared
                 for c in self.s.conditionings() if self.check_only(c)]
        extra += [(ref, self.s.primary().conditioning), (ref, "breach_noburn")]
        for pair in extra:
            if pair[1] in self.s.conditionings() and pair not in pairs:
                pairs.append(pair)
        return pairs

    # -- step 3 --
    def delineate(self, combo: Combo, line: Any, point: Any, opts: dict[str, Any]) -> tuple[
            Any, int | None]:
        area = opts.get("area_km2")
        outlet = route.snap_outlet(combo.acc, self.grid, line, point,
                                   line_buffer_m=float(opts["line_buffer_m"]),
                                   max_dist_m=float(opts["max_dist_m"]),
                                   area_km2=tuple(area) if area else None)
        if outlet is None:
            return None, None
        ds = route.downstream_index(combo.fdir)
        basin = route.catchment(ds, combo.acc, outlet).reshape(self.grid.shape)
        return route.mask_to_polygon(basin, self.grid), outlet

    def osm_lines(self) -> gpd.GeoDataFrame:
        return gpd.read_file(self.s.resolve(self.s.v["burn"]["lines"][0]))

    def confluence(self) -> Any:
        lon, lat = self.s.v["confluence_lonlat"]
        return gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(self.grid.crs).iloc[0]

    def domain(self, burned: Combo, unburned: Combo | None) -> None:
        opts = self.s.v["domain_outlet"]
        lines = self.osm_lines()
        bag = lines[lines["name"].fillna("").str.contains(opts["line_name_contains"])]
        line = unary_union(bag.to_crs(self.grid.crs).geometry)
        point = self.confluence()
        poly, outlet = self.delineate(burned, line, point, opts)
        if poly is None or outlet is None:
            raise SweepError("no domain outlet found on the burned GLO-30")
        mask = self.grid.mask(poly)
        self.domain_mask = mask
        self.domain_poly = poly
        self.save(mask.astype(np.uint8), "grid", "domain.tif", nodata=None)
        rows = []

        def describe(name: str, geom: Any, out_idx: int | None, source: Combo | None) -> None:
            row: dict[str, Any] = {"domain": name, "area_km2": geom.area / 1e6,
                                   "iou_vs_burned": geom.intersection(poly).area
                                   / geom.union(poly).area}
            if out_idx is not None and source is not None:
                r, c = divmod(out_idx, self.grid.width)
                x, y = self.grid.cell_centre(r, c)
                row.update({"outlet_x": x, "outlet_y": y,
                            "outlet_upstream_km2": source.acc.ravel()[out_idx]
                            * self.grid.cell_area_m2 / 1e6,
                            "outlet_dist_to_confluence_m": Point(x, y).distance(point)})
            rows.append(row)

        describe(f"{burned.name} (adopted)", poly, outlet, burned)
        if unburned is not None:
            upoly, uout = self.delineate(unburned, line, point, opts)
            if upoly is not None:
                describe(unburned.name, upoly, uout, unburned)
        describe("research (unburned GLO-30, pyflwdir, EPSG:4326 grid)", self.research_domain,
                 None, None)
        write_csv(self.out("routing_domain_comparison.csv"), rows)
        r, c = divmod(outlet, self.grid.width)
        x, y = self.grid.cell_centre(r, c)
        frame = gpd.GeoDataFrame(
            [{"name": "upper_bagmati_routing_domain_glo30_osm_burned", "dem": burned.dem,
              "conditioning": burned.conditioning,
              "burn_depth_m": float(self.s.v["burn"]["depth_m"]),
              "area_km2": round(poly.area / 1e6, 2), "outlet_x_epsg32645": x,
              "outlet_y_epsg32645": y, "geometry": poly}], crs=self.grid.crs).to_crs("EPSG:4326")
        atomic_write_text(self.out("routing_domain_burned.geojson"), frame.to_json())
        self.summary["domain"] = rows

    # -- step 4 --
    def burn_check(self, combo: Combo) -> dict[str, Any]:
        opts, check = self.s.v["nakkhu_outlet"], self.s.v["burn_check"]
        lines = self.osm_lines()
        nak = lines[lines["name"].isin(opts["line_names"])]
        line = unary_union(nak.to_crs(self.grid.crs).geometry)
        lat0, lat1 = check["lat_band"]
        band_names = lines[lines["name"].isin(check["line_names"])]
        band = unary_union(band_names.geometry).intersection(box(-180, lat0, 180, lat1))
        band_utm = gpd.GeoSeries([band], crs="EPSG:4326").to_crs(self.grid.crs).iloc[0]
        poly, outlet = self.delineate(combo, line, self.confluence(), opts)
        burned = bool(self.s.conditionings()[combo.conditioning].get("burn"))
        row: dict[str, Any] = {"dem": combo.dem, "conditioning": combo.conditioning,
                               "burned": burned, "channel_band_length_m": band_utm.length}
        if poly is None:
            row.update({"outlet_found": False, "catchment_km2": float("nan"),
                        "share_inside": 0.0})
        else:
            row.update({"outlet_found": True, "catchment_km2": poly.area / 1e6,
                        "share_inside": band_utm.intersection(poly).length / band_utm.length})
        gate = self.s.v["routing_gate"]
        ref_area = self.s.read_polygon(gate["reference_catchment"]).area / 1e6
        rel = (row["catchment_km2"] - ref_area) / ref_area if poly is not None else float("nan")
        reasons = []
        if row["share_inside"] < float(gate["min_channel_share"]):
            reasons.append(f"channel share {row['share_inside']:.4f} < {gate['min_channel_share']}")
        if not np.isfinite(rel):
            reasons.append("no Nakkhu outlet")
        elif abs(rel) > float(gate["max_catchment_rel_diff"]):
            reasons.append(f"catchment {row['catchment_km2']:.2f} km2 differs from the reference "
                           f"{ref_area:.2f} km2 by {rel:+.1%}")
        row.update({"reference_catchment_km2": ref_area, "catchment_rel_diff": rel,
                    "gate_pass": not reasons, "gate_reason": "; ".join(reasons)})
        return row

    # -- step 5 --
    def drop(self, combo: Combo) -> tuple[list[dict[str, Any]], float | None]:
        opts = self.s.v["drop_analysis"]
        ds = route.downstream_index(combo.fdir)
        area = combo.acc.ravel() * self.grid.cell_area_m2 / 1e6
        tests, selected = drop_analysis.drop_analysis(
            ds, area, combo.reference.ravel().astype(np.float64), self.domain_mask.ravel(),
            opts["candidates_km2"], float(opts["t_critical"]))
        rows = [{"dem": combo.dem, "conditioning": combo.conditioning, **vars(t),
                 "selected": selected is not None and t.threshold_km2 == selected}
                for t in tests]
        return rows, selected

    # -- step 6 --
    def surface(self, combo: Combo, threshold: str, threshold_km2: float) -> dict[str, Any]:
        start = time.perf_counter()
        surface = Surface(combo.dem, combo.conditioning, threshold)
        streams = handmod.stream_mask(combo.acc, self.grid, threshold_km2)
        raw = handmod.hand_pysheds(combo.fdir, combo.reference, streams, self.grid)
        hand, report = handmod.check_hand(raw, self.domain_mask, streams, combo.reference,
                                          combo.fdir)
        if not report.ok:
            raise handmod.HandUnitsError(f"{surface.cfg_id}: " + "; ".join(report.problems))
        self.save(streams.astype(np.uint8), surface.cfg_id, "streams.tif", nodata=None)
        path = self.save(hand.astype(np.float32), surface.cfg_id, "hand.tif", nodata=np.nan,
                         tags={"units": "m", "cfg_id": surface.cfg_id})
        return {"status": "done", "reason": "", "threshold_km2": threshold_km2,
                **report.to_dict(), "runtime_s": round(time.perf_counter() - start, 2),
                "hand_path": self.s.project.paths.relative(path)}

    def grid_row(self, surface: Surface, **values: Any) -> dict[str, Any]:
        spec = self.s.conditionings().get(surface.conditioning, {})
        primary = self.s.primary()
        diagnostic = bool(spec.get("diagnostic", False))
        gate = getattr(self, "gate", {}).get(surface.combo)
        gate_pass = bool(gate["gate_pass"]) if gate else None
        done = values.get("status") == "done"
        return {"cfg_id": surface.cfg_id, "dem": surface.dem,
                "conditioning": surface.conditioning, "diagnostic": diagnostic,
                "threshold": surface.threshold, "is_primary": surface == primary,
                "routing_gate_pass": gate_pass,
                "routing_gate_reason": gate["gate_reason"] if gate else "",
                "in_sensitivity_range": bool(done and not diagnostic and gate_pass), **values}

    # -- step 7 --
    def agreement_sample(self, done: list[Surface]) -> list[Surface]:
        opts = self.s.v["agreement"]
        rng = np.random.default_rng(int(opts["seed"]))
        diagnostic = {c for c, v in self.s.conditionings().items() if v.get("diagnostic")}
        eligible = [x for x in done if x.conditioning not in diagnostic]
        dems = sorted({x.dem for x in eligible})
        thresholds = [t for t in self.s.all_thresholds() if any(x.threshold == t for x in eligible)]
        if not dems or not thresholds:
            return []
        perm = [thresholds[i] for i in rng.permutation(len(thresholds))]
        picks: list[Surface] = []
        n = min(int(opts["n_configs"]), len(eligible))
        i = 0
        while len(picks) < n and i < 10 * n * len(dems):
            dem, thr = dems[i % len(dems)], perm[i % len(perm)]
            options = [x for x in eligible if x.dem == dem and x.threshold == thr
                       and x not in picks]
            if options:
                picks.append(options[int(rng.integers(len(options)))])
            i += 1
        return picks

    def agreement(self, picks: list[Surface], combos: dict[str, Combo],
                  thresholds_km2: dict[str, float]) -> None:
        tol = float(self.s.v["agreement"]["tolerance_m"])
        rows = []
        domain = self.domain_mask
        for x in picks:
            combo = combos[x.combo]
            streams = handmod.stream_mask(combo.acc, self.grid, thresholds_km2[x.cfg_id])
            z = combo.routing_dem
            surfaces = {
                "pysheds": handmod.hand_pysheds(combo.fdir, z, streams, self.grid),
            }
            surfaces["pyflwdir"], d8_flw = handmod.hand_pyflwdir(z, z, streams, self.grid)
            surfaces["whitebox"] = handmod.hand_whitebox(z, streams, self.grid, self.wbt,
                                                         self.s.interim)
            d8_wbt = handmod.whitebox_d8(z, self.grid, self.wbt, self.s.interim)
            surfaces["pyflwdir_hand_on_pysheds_d8"] = handmod.hand_pyflwdir_on_d8(
                combo.fdir, z, streams, self.grid)
            surfaces["pyflwdir_hand_on_whitebox_d8"] = handmod.hand_pyflwdir_on_d8(
                d8_wbt, z, streams, self.grid)
            d8_equal = float((d8_flw[domain].astype(np.int64) == combo.fdir[domain]).mean())
            d8_wbt_equal = float((d8_wbt[domain] == combo.fdir[domain]).mean())
            d8_share = {"pyflwdir": d8_equal, "whitebox": d8_wbt_equal,
                        "pyflwdir_hand_on_pysheds_d8": 1.0,
                        "pyflwdir_hand_on_whitebox_d8": d8_wbt_equal}
            for a, b in (("pysheds", "pyflwdir"), ("pysheds", "whitebox"),
                         ("pyflwdir", "whitebox"), ("pysheds", "pyflwdir_hand_on_pysheds_d8"),
                         ("pysheds", "pyflwdir_hand_on_whitebox_d8")):
                metrics = hazard.compare_surfaces(surfaces[a][domain], surfaces[b][domain], tol,
                                                  self.s.edges)
                rows.append({"cfg_id": x.cfg_id, "dem": x.dem, "conditioning": x.conditioning,
                             "threshold": x.threshold,
                             "threshold_km2": thresholds_km2[x.cfg_id],
                             "elevation": "routing DEM (same input to all three)",
                             "pair": f"{a}_vs_{b}", **metrics,
                             "share_d8_equal": d8_share[b] if a == "pysheds" else float("nan")})
        rows += self.primary_isolation(combos, thresholds_km2, tol)
        write_csv(self.out("implementation_agreement.csv"), rows)

    def primary_isolation(self, combos: dict[str, Combo], thresholds_km2: dict[str, float],
                          tol: float) -> list[dict[str, Any]]:
        """H1 isolation on the primary surface itself (production reference elevations)."""
        x = self.s.primary()
        if x.cfg_id not in thresholds_km2:
            return []
        combo, domain = combos[x.combo], self.domain_mask
        streams = handmod.stream_mask(combo.acc, self.grid, thresholds_km2[x.cfg_id])
        primary = demmod.read_raster(self.raster_path(x.cfg_id, "hand.tif")).astype(np.float64)
        d8_wbt = handmod.whitebox_d8(combo.routing_dem, self.grid, self.wbt, self.s.interim)
        own, d8_flw = handmod.hand_pyflwdir(combo.routing_dem, combo.reference, streams,
                                            self.grid)
        candidates = {
            "pyflwdir_hand_on_pysheds_d8": (handmod.hand_pyflwdir_on_d8(
                combo.fdir, combo.reference, streams, self.grid), 1.0),
            "pyflwdir_hand_on_whitebox_d8": (handmod.hand_pyflwdir_on_d8(
                d8_wbt, combo.reference, streams, self.grid),
                float((d8_wbt[domain] == combo.fdir[domain]).mean())),
            "pyflwdir": (own, float((d8_flw[domain].astype(np.int64)
                                     == combo.fdir[domain]).mean())),
        }
        rows = []
        for name, (surface, d8_equal) in candidates.items():
            clamped = np.where(surface < 0, 0.0, surface)  # as the production surface
            metrics = hazard.compare_surfaces(primary[domain], clamped[domain], tol,
                                              self.s.edges)
            rows.append({"cfg_id": x.cfg_id, "dem": x.dem, "conditioning": x.conditioning,
                         "threshold": x.threshold, "threshold_km2": thresholds_km2[x.cfg_id],
                         "elevation": "reference (production primary, negatives clamped)",
                         "pair": f"primary_vs_{name}", **metrics, "share_d8_equal": d8_equal})
        return rows

    def asf(self, done: list[Surface]) -> None:
        s = self.s
        entry = s.v["reference_hand"]
        record = s.manifest.get(entry["manifest_id"], {})
        path = s.resolve(entry["path"])
        if record.get("status") == "pending" or not path.is_file():
            write_csv(self.out("asf_reference_agreement.csv"),
                      [{"status": "skipped", "reason": "ASF HAND not available"}])
            return
        asf = demmod.prepare_dem(path, self.grid, resampling="bilinear").astype(np.float64)
        extent = self.grid.mask(s.read_polygon(s.v["analysis_extent"]))
        tol = float(s.v["agreement"]["tolerance_m"])
        rows = []
        for x in done:
            if x.dem != "glo30":
                continue
            hand = demmod.read_raster(self.raster_path(x.cfg_id, "hand.tif")).astype(np.float64)
            for region, mask in (("routing_domain", self.domain_mask),
                                 ("analysis_extent", extent)):
                metrics = hazard.compare_surfaces(hand[mask], asf[mask], tol, s.edges)
                rows.append({"cfg_id": x.cfg_id, "conditioning": x.conditioning,
                             "threshold": x.threshold, "is_primary": x == s.primary(),
                             "region": region, "asf_resampling": "bilinear", **metrics})
        write_csv(self.out("asf_reference_agreement.csv"), rows)

    # -- driver --
    def execute(self) -> dict[str, Any]:
        s = self.s
        with self.run.step("prepare"):
            self.prepare()
        combos: dict[str, Combo] = {}
        with self.run.step("routing"):
            ref = s.v["reference_dem"]
            for dem, c in self.routing_pairs():
                combos[f"{dem}__{c}"] = self.route_combo(dem, c)
        with self.run.step("domain"):
            primary = s.primary()
            self.domain(combos[f"{ref}__{primary.conditioning}"],
                        combos.get(f"{ref}__breach_noburn"))
        with self.run.step("burn-check"):
            rows = [self.burn_check(c) for c in combos.values()]
            write_csv(self.out("burn_check.csv"), rows)
            self.summary["burn_check"] = rows
            self.gate = {f"{r['dem']}__{r['conditioning']}": r for r in rows}
        selected: dict[str, float | None] = {}
        with self.run.step("drop-analysis"):
            drop_rows = []
            for name, combo in combos.items():
                if DROP in self.thresholds and (combo.dem, combo.conditioning) in self.combos():
                    r, sel = self.drop(combo)
                    drop_rows += r
                    selected[name] = sel
            write_csv(self.out("drop_analysis.csv"), drop_rows)
            self.summary["drop_thresholds_km2"] = selected
        grid_rows: list[dict[str, Any]] = []
        done: list[Surface] = []
        thresholds_km2: dict[str, float] = {}
        with self.run.step("surfaces"):
            for dem in self.dems:
                for c in (c for c in self.conds if not self.check_only(c)):
                    for thr in self.thresholds:
                        surface = Surface(dem, c, thr)
                        status, reason = s.dem_status(dem)
                        if status != "available":
                            grid_rows.append(self.grid_row(surface, status="skipped",
                                                           reason=reason))
                            continue
                        km2 = selected.get(surface.combo) if thr == DROP else float(thr)
                        if km2 is None:
                            grid_rows.append(self.grid_row(
                                surface, status="skipped",
                                reason="no drop-analysis threshold with |t| < t_critical"))
                            continue
                        try:
                            values = self.surface(combos[surface.combo], thr, float(km2))
                        except handmod.HandUnitsError as error:
                            # Plan 008 STOP condition; a derived sweep may set
                            # `on_units_failure: skip` to record the surface as failed instead.
                            if s.v.get("on_units_failure", "stop") != "skip" or surface == \
                                    s.primary():
                                raise
                            grid_rows.append(self.grid_row(
                                surface, status="failed", reason=f"units check: {error}",
                                threshold_km2=float(km2)))
                            continue
                        grid_rows.append(self.grid_row(surface, **values))
                        done.append(surface)
                        thresholds_km2[surface.cfg_id] = float(km2)
            write_csv(self.out("config_grid.csv"), grid_rows)
        prim = next((r for r in grid_rows if r["is_primary"]), None)
        if prim is not None and prim.get("status") == "done" and not prim["routing_gate_pass"]:
            raise SweepError(f"primary {prim['cfg_id']} fails the routing gate: "
                             f"{prim['routing_gate_reason']}")
        if self.do_agreement:
            with self.run.step("agreement"):
                picks = self.agreement_sample(done)
                self.agreement(picks, combos, thresholds_km2)
                self.asf(done)
                self.summary["agreement_sample"] = [p.cfg_id for p in picks]
        self.summary.update({
            "surfaces_done": len(done),
            "surfaces_skipped": sum(r["status"] == "skipped" for r in grid_rows),
            "units_ok_all": all(r.get("units_ok", True) for r in grid_rows),
            "surfaces_in_sensitivity_range": sum(r["in_sensitivity_range"] for r in grid_rows),
            "surfaces_failing_routing_gate": sorted(
                r["cfg_id"] for r in grid_rows
                if r.get("status") == "done" and r["routing_gate_pass"] is False),
            "primary": s.primary().cfg_id,
            "sweep_config_sha256": s.config.sha256,
        })
        self.run.write_summary(self.summary)
        return self.summary


def run_sweep(project: Project, config: str = SWEEP_CONFIG) -> tuple[Run, dict[str, Any]]:
    settings, surface = load_settings(project, config)
    run = create_run(project, settings.stage, settings.config)
    if surface is None:
        runner = SweepRunner(settings, run,
                             agreement=bool(settings.v.get("agreement_step", True)))
    else:
        runner = SweepRunner(settings, run, dems=[surface.dem], conditionings=[
            surface.conditioning], thresholds=[surface.threshold], agreement=False)
    return run, runner.execute()


# --- building attribution (Step 7-8) --------------------------------------------------------


def latest_sweep(project: Project, stage: str = STAGE) -> Run:
    """Most recent complete full sweep (one with the agreement step), else any complete run."""
    runs = [r for r in list_runs(project.paths, stage)
            if r.status == "complete" and (r.path / "config_grid.csv").is_file()]
    full = [r for r in runs if (r.path / "implementation_agreement.csv").is_file()]
    candidates = full or runs
    if not candidates:
        raise SweepError("no complete `hand` sweep run; run `nakkhu hand sweep` first")
    return candidates[-1]


def sweep_run(project: Project, ref: str, default_stage: str = STAGE) -> Run:
    """The run ``<stage>/<run_id>`` (or a bare ``<run_id>`` of ``default_stage``)."""
    stage, _, run_id = ref.rstrip("/").rpartition("/")
    stage = stage.split("/")[-1] if stage else default_stage
    return Run(stage, run_id, project.paths.runs / stage / run_id)


def load_greenhouses(settings: Settings) -> gpd.GeoDataFrame | None:
    """OSM greenhouse areas (grid CRS) over the extent and training domain, or None if off."""
    from nakkhu.reference import greenhouse

    opts = settings.v.get("greenhouse_filter") or {}
    if not opts.get("enabled"):
        return None
    areas = [settings.read_polygon(settings.v["analysis_extent"])]
    training = (settings.project.config.values.get("aoi") or {}).get("training_domain")
    if training:
        areas.append(settings.read_polygon(training))
    lonlat = gpd.GeoSeries(areas, crs=settings.crs).to_crs("EPSG:4326").total_bounds
    pad = float(opts.get("bbox_pad_deg", 0.005))
    bbox = (lonlat[0] - pad, lonlat[1] - pad, lonlat[2] + pad, lonlat[3] + pad)
    frame = greenhouse.load_greenhouses(
        settings.resolve(opts["pbf"]), bbox,
        settings.project.paths.data_root / "interim" / "reference")
    return frame.to_crs(settings.crs)


def load_inventory(settings: Settings, name: str, extent: Any,
                   greenhouses: gpd.GeoDataFrame | None = None
                   ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Buildings with centroid in the extent, after the optional confidence cut and the
    greenhouse filter, with the counts at each stage."""
    entry = settings.v["inventories"][name]
    path = settings.resolve(entry["path"])
    if not entry.get("manifest_id") and not path.is_file():
        raise SweepError(f"inventory {name} is pending (no file at {entry['path']})")
    frame = gpd.read_parquet(path).to_crs(settings.crs)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    inside = frame.geometry.centroid.within(extent)
    frame = frame[inside].reset_index(drop=True)
    info: dict[str, Any] = {"n_in_extent": len(frame), "role": entry.get("role", "primary")}
    cut = entry.get("confidence_cut")
    if cut:
        keep, thresholds = hazard.confidence_cut(frame, settings.resolve(cut["thresholds"]),
                                                 cut["column"])
        frame = frame[keep].reset_index(drop=True)
        info.update({"confidence_cut": cut["column"], "n_after_cut": len(frame),
                     "cut_thresholds": sorted(float(v) for v in np.unique(thresholds))})
    if greenhouses is not None:
        from nakkhu.reference.greenhouse import greenhouse_overlap

        flags = greenhouse_overlap(frame.geometry, greenhouses.geometry,
                                   float(settings.v["greenhouse_filter"]["iou_min"]))
        info.update({"n_greenhouse_removed": int(flags["is_greenhouse"].sum()),
                     "n_greenhouse_by_iou": int((flags["max_iou"]
                                                 >= float(settings.v["greenhouse_filter"]
                                                          ["iou_min"])).sum()),
                     "n_greenhouse_by_centroid": int(flags["centroid_inside"].sum())})
        frame = frame[~flags["is_greenhouse"].to_numpy()].reset_index(drop=True)
    info["n_final"] = len(frame)
    frame.insert(0, "building_uid", [f"{name}:{i}" for i in range(len(frame))])
    return frame, info


def attribute(project: Project, inventories: list[str], sweep_ref: str | None = None,
              config: str = SWEEP_CONFIG) -> tuple[Run, dict[str, Any]]:
    settings, _ = load_settings(project, config)
    sweep = (latest_sweep(project, settings.stage) if sweep_ref is None
             else sweep_run(project, sweep_ref, settings.stage))
    if sweep.status != "complete":
        raise SweepError(f"sweep run {sweep.ref} is {sweep.status}")
    sweep_summary = json.loads((sweep.path / "summary.json").read_text(encoding="utf-8"))
    interim = project.paths.resolve(sweep_summary["interim"])
    grid_table = pd.read_csv(sweep.path / "config_grid.csv")
    done = grid_table[grid_table["status"] == "done"]
    primary = settings.primary()
    if primary.cfg_id not in set(done["cfg_id"]):
        raise SweepError(f"primary surface {primary.cfg_id} is not in sweep {sweep.ref}")
    if "in_sensitivity_range" not in done.columns:
        raise SweepError(f"sweep {sweep.ref} predates the routing gate; re-run `hand sweep`")
    expected = settings.v["primary"].get("threshold_km2")
    got = float(done.set_index("cfg_id").at[primary.cfg_id, "threshold_km2"])
    if expected is not None and abs(got - float(expected)) > 1e-9:
        raise SweepError(f"primary threshold {got} km2 != configured {expected} km2")
    grid = settings.grid()
    extent = settings.read_polygon(settings.v["analysis_extent"])
    stats = list(settings.v["statistics"])
    edges = settings.edges
    labels = hazard.tier_labels(edges)
    n_tiers = len(labels)
    run = create_run(project, settings.attribute_stage, settings.config)
    stability: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    per_inventory: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {"sweep_run": sweep.ref, "inventories": per_inventory}
    surfaces = {row.cfg_id: demmod.read_raster(interim / row.cfg_id / "hand.tif").ravel()
                for row in done.itertuples()}
    greenhouses = load_greenhouses(settings)
    if greenhouses is not None:
        atomic_write_text(run.path / "greenhouses_osm.geojson",
                          greenhouses.to_crs("EPSG:4326").to_json())
        summary["greenhouses"] = {"n_areas": len(greenhouses),
                                  "by_tag": greenhouses["tag"].value_counts().to_dict()}
    for name in inventories:
        with run.step(f"attribute-{name.lower()}"):     # step names are lowercase (L2_seed0)
            frame, info = load_inventory(settings, name, extent, greenhouses)
            centroid = hazard.centroid_cells(frame.geometry, grid)
            index_tif = interim / "grid" / "cell_index.tif"
            footprint = hazard.cell_groups(frame.geometry, index_tif)
            buffered = hazard.cell_groups(
                frame.geometry.buffer(float(settings.v["buffer_m"])), index_tif)
            columns: dict[str, np.ndarray] = {}
            for cfg_id, values in surfaces.items():
                for stat in stats:
                    if stat == "centroid":
                        col = hazard.sample_centroids(values, centroid)
                    elif stat == "footprint_min":
                        col = footprint.minimum(values)
                    elif stat == "buffer_min":
                        col = buffered.minimum(values)
                    else:
                        raise SweepError(f"unknown statistic {stat!r}")
                    columns[f"{cfg_id}__{stat}"] = col.astype(np.float32)
            keep = [c for c in ("source_id", "confidence") if c in frame.columns]
            table = pd.DataFrame({"building_uid": frame["building_uid"],
                                  **{c: frame[c] for c in keep},
                                  "area_m2": frame.geometry.area.astype(np.float32),
                                  **columns})
            table.to_parquet(run.path / f"building_hand_{name}.parquet", index=False,
                             compression="zstd")
            ref_key = f"{primary.cfg_id}__{primary_stat(settings)}"
            ref_vals = columns[ref_key]
            ref_tiers = hazard.tier_index(ref_vals, edges)
            meta = done.set_index("cfg_id")
            for key, col in columns.items():
                cfg_id, stat = key.rsplit("__", 1)
                tiers = hazard.tier_index(col, edges)
                counts = hazard.tier_counts(tiers, n_tiers)
                row = {"inventory": name, "inventory_role": info["role"],
                       "cfg_id": cfg_id, "dem": meta.at[cfg_id, "dem"],
                       "conditioning": meta.at[cfg_id, "conditioning"],
                       "threshold": meta.at[cfg_id, "threshold"],
                       "threshold_km2": meta.at[cfg_id, "threshold_km2"], "statistic": stat,
                       "is_primary": key == ref_key,
                       "diagnostic": bool(meta.at[cfg_id, "diagnostic"]),
                       "routing_gate_pass": bool(meta.at[cfg_id, "routing_gate_pass"]),
                       "in_sensitivity_range": bool(meta.at[cfg_id, "in_sensitivity_range"]),
                       "n_buildings": len(col),
                       "n_missing": int((tiers < 0).sum()),
                       **{f"n_{lab}": int(n) for lab, n in zip(labels, counts, strict=True)},
                       "spearman_vs_primary": hazard.spearman(col.astype(np.float64),
                                                              ref_vals.astype(np.float64)),
                       "kappa_linear_vs_primary": hazard.weighted_kappa(tiers, ref_tiers,
                                                                        n_tiers),
                       "share_same_tier_vs_primary": float(
                           (tiers[(tiers >= 0) & (ref_tiers >= 0)]
                            == ref_tiers[(tiers >= 0) & (ref_tiers >= 0)]).mean())}
                stability.append(row)
                matrix = hazard.transition_matrix(ref_tiers, tiers, n_tiers)
                for i, j in zip(*np.nonzero(matrix), strict=True):
                    transitions.append({"inventory": name, "cfg_id": cfg_id, "statistic": stat,
                                        "primary_tier": labels[i], "tier": labels[j],
                                        "n": int(matrix[i, j])})
            per_inventory[name] = {"n_buildings": len(frame), **info}
    write_csv(run.path / "count_stability.csv", stability)
    write_csv(run.path / "inventory_filters.csv",
              [{"inventory": k, **{c: v for c, v in i.items() if c != "cut_thresholds"},
                "cut_thresholds": ";".join(map(str, i.get("cut_thresholds", [])))}
               for k, i in per_inventory.items()])
    write_csv(run.path / "tier_range_summary.csv", range_summary(pd.DataFrame(stability), labels))
    write_csv(run.path / "tier_transitions.csv", transitions)
    summary["tiers"] = labels
    summary["primary"] = f"{primary.cfg_id}__{primary_stat(settings)}"
    atomic_write_json(run.path / "summary.json", summary)
    return run, summary


def range_summary(stability: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Min/max tier counts over the sensitivity members (``in_sensitivity_range``: done,
    non-diagnostic, routing gate passed); gated and diagnostic surfaces are excluded."""
    cols = [f"n_{lab}" for lab in labels]
    frame = stability.assign(n_le_3=stability[cols[0]] + stability[cols[1]])
    cols = [*cols, "n_le_3"]
    rows = []
    for inventory, inv in frame.groupby("inventory", sort=False):
        primary = inv[inv["is_primary"]].iloc[0]
        members = inv[inv["in_sensitivity_range"]]
        groups: list[tuple[str, pd.DataFrame]] = [("all surfaces x statistics", members)]
        groups += [(f"statistic={k}", g) for k, g in members.groupby("statistic", sort=False)]
        centroid = members[members["statistic"] == "centroid"]
        for factor in ("dem", "conditioning", "threshold"):
            groups += [(f"centroid, {factor}={k}", g)
                       for k, g in centroid.groupby(factor, sort=False)]
        for group, g in groups:
            row: dict[str, Any] = {"inventory": inventory, "group": group,
                                   "n_members": len(g),
                                   "kappa_linear_vs_primary_min": g[
                                       "kappa_linear_vs_primary"].min()}
            for c in cols:
                row[f"{c}_primary"] = int(primary[c])
                row[f"{c}_min"] = int(g[c].min())
                row[f"{c}_max"] = int(g[c].max())
            rows.append(row)
        excluded = inv[~inv["in_sensitivity_range"]]
        rows.append({"inventory": inventory, "group": "excluded (diagnostic or routing gate)",
                     "n_members": len(excluded),
                     "excluded_cfg_ids": ";".join(sorted(excluded["cfg_id"].unique()))})
    return pd.DataFrame(rows)


def primary_stat(settings: Settings) -> str:
    return str(settings.v["primary"]["statistic"])
