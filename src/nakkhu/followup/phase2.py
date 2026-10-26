"""Plan 014 Phase 2, label-free site preparation (``nakkhu followup p2 ...``).

Everything here is label-free: it reads imagery, footprints, OSM lines and the frozen routing
rasters, never a change-map class, an audit answer or a flood reference. The settings come from
``configs/followup/phase2/sites.yaml`` and the frozen rules of ``prereg_v9_a3`` (main mask,
hand_mr_d8, the P1-6 features). Steps, each a run under ``runs/followup-p2/``:

- ``sites``: per site the pre/post imagery overlap (ARD data masks), the site lines (listed OSM
  ways clipped to the overlap), the extended plan-005 lattice blocks whose centre lies within
  ``corridor_m`` of the site lines, minus blocks touching the Nakkhu extent + 320 m, shared
  blocks to the site with the nearer lines, the extent (union of the blocks), the main lines
  (site lines + Bagmati ways clipped to the extent) and the Microsoft building table (v9
  inventory rule with the site extent). Promoted to ``artifacts/followup/phase2/``.
- ``osm``: OSM building footprints over the routing clip from the pinned PBF (ground mask).
- ``mosaic``: 0.5 m pre (strips composited in the listed order) and post mosaics on a site grid
  aligned with the plan 003 lattice.
- ``register``: the plan-004 footprint registration (Google + Microsoft, and OSM) on the site's
  320 m lattice tiles of the pre mosaic, with the v9 thresholds.
- ``coreg``: the v9 change map's Step 1 only (observability masks + co-registration + held-back
  verification) for a site; no building feature, score or class is computed.
- ``predictors``: hand_v9, hand_mr_d8 and the 28 P1-6 features for the site's buildings, with the
  site's main lines and main mask; distance to the site's main lines.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from nakkhu.config import Project, ResolvedConfig, load_yaml
from nakkhu.followup import spec as S

CONFIG = "configs/followup/phase2/sites.yaml"
STAGE = "followup-p2"
ART = f"{S.ARTIFACT_DIR}/phase2"            # under artifacts/
SWEEP = "configs/hand/sweep.yaml"
CHANGE = "configs/flood/change.yaml"
CRS = "EPSG:32645"
#: change.yaml sections a site run must take unchanged (the v9 rule)
RULE_SECTIONS = ("coregistration", "mask", "normalise", "score", "sentinel2")
#: buildings keys a site run takes unchanged (paths of the site inputs differ)
BUILDING_RULE_KEYS = ("inventories", "primary_inventory", "audit_inventories",
                      "ground_mask_from", "ring_m", "min_ring_px", "observable_buffer_m")


class Phase2Error(S.FollowupError):
    """A Phase-2 site input is missing or inconsistent."""


# --- config and small helpers -------------------------------------------------------------------


def load_sites(project: Project, path: str = CONFIG) -> dict[str, Any]:
    cfg = load_yaml(project.paths.resolve(path))
    if not isinstance(cfg, dict) or "sites" not in cfg:
        raise Phase2Error(f"{path}: no `sites`")
    return cfg


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def art_path(project: Project, name: str) -> Path:
    return project.paths.artifacts / ART / name


def rel_art(name: str) -> str:
    return f"artifacts/{ART}/{name}"


def manifest_entry(project: Project, sid: str) -> dict[str, Any]:
    from nakkhu.manifest import load_manifest

    path = project.paths.resolve((project.config.values.get("paths") or {}).get(
        "manifest", "configs/data/manifest.yaml"))
    for entry in load_manifest(path):
        if entry["id"] == sid:
            if entry.get("status") == "pending" or not entry.get("local_path"):
                raise Phase2Error(f"{sid} is pending in {path.name}")
            return entry
    raise Phase2Error(f"{sid} is not in {path.name} (fetch it first)")


def scene_files(project: Project, scene: str) -> list[Path]:
    base = project.paths.resolve(manifest_entry(project, scene)["local_path"])
    files = sorted((base / "visual").glob("*.tif"))
    if not files:
        raise Phase2Error(f"no visual GeoTIFFs under {base / 'visual'}")
    return files


def datamask_polygon(project: Project, cfg: dict[str, Any], scene: str) -> Any:
    sid = cfg["datamasks"][scene]
    base = project.paths.resolve(manifest_entry(project, sid)["local_path"])
    parts = []
    for gpkg in sorted((base / "data-mask").glob("*.gpkg")):
        frame = gpd.read_file(gpkg)
        if frame.crs is None:
            raise Phase2Error(f"{gpkg.name}: no CRS")
        parts.extend(frame.to_crs(CRS).geometry.values)
    if not parts:
        raise Phase2Error(f"{sid}: no data-mask polygons")
    return shapely.make_valid(shapely.union_all(np.asarray(parts, dtype=object)))


def river_ways(project: Project, path: str) -> gpd.GeoDataFrame:
    """OSM ways of an Overpass JSON with their ids and names (EPSG:32645)."""
    from shapely.geometry import LineString

    doc = json.loads(project.paths.resolve(path).read_text(encoding="utf-8"))
    rows = []
    for e in doc.get("elements") or []:
        if e.get("type") != "way" or len(e.get("geometry") or []) < 2:
            continue
        tags = e.get("tags") or {}
        rows.append({"way_id": int(e["id"]), "name": tags.get("name:en") or tags.get("name"),
                     "geometry": LineString([(p["lon"], p["lat"]) for p in e["geometry"]])})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS)


# --- lattice ------------------------------------------------------------------------------------


def lattice_blocks(bounds: tuple[float, float, float, float], lat: dict[str, Any]
                   ) -> gpd.GeoDataFrame:
    """Every lattice block (960 m) intersecting ``bounds``; ids as ``splits.block_ids``."""
    west, north = float(lat["west"]), float(lat["north"])
    size = float(lat["tile_m"]) * int(lat["block_tiles"])
    x0, y0, x1, y1 = bounds
    c0, c1 = int(np.floor((x0 - west) / size)), int(np.floor((x1 - west) / size))
    r0, r1 = int(np.floor((north - y1) / size)), int(np.floor((north - y0) / size))
    rows = []
    # blocks west or north of the origin lie outside the WV02 F5AB8000 footprint that defines
    # the plan 003 origin, hence outside every site's imagery: they are not enumerated
    for r in range(max(r0, 0), r1 + 1):
        for c in range(max(c0, 0), c1 + 1):
            bx, by = west + c * size, north - r * size
            rows.append({"block_id": f"b_{c:02d}_{r:02d}", "block_col": c, "block_row": r,
                         "geometry": shapely.box(bx, by - size, bx + size, by)})
    return gpd.GeoDataFrame(rows, crs=CRS)


def block_of(x: np.ndarray, y: np.ndarray, lat: dict[str, Any]) -> np.ndarray:
    size = float(lat["tile_m"]) * int(lat["block_tiles"])
    c = np.floor((np.asarray(x) - float(lat["west"])) / size).astype(int)
    r = np.floor((float(lat["north"]) - np.asarray(y)) / size).astype(int)
    return np.array([f"b_{a:02d}_{b:02d}" for a, b in zip(c, r, strict=True)], dtype=object)


def corridor_blocks(site_lines: dict[str, Any], cfg: dict[str, Any], exclusion: Any
                    ) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, Any]]:
    """Blocks per site: centre within ``corridor_m`` of the site lines, not touching
    ``exclusion``; a block qualifying for several sites goes to the site whose lines are
    nearer to its centre (ties to the earlier site in the config)."""
    lat, radius = cfg["lattice"], float(cfg["corridor_m"])
    order = list(site_lines)
    cand: dict[str, gpd.GeoDataFrame] = {}
    info: dict[str, Any] = {}
    for s in order:
        lines = site_lines[s]
        if lines is None or lines.is_empty:
            raise Phase2Error(f"site {s}: no site line inside the imagery overlap")
        blocks = lattice_blocks(lines.buffer(radius + 1000.0).bounds, lat)
        centres = shapely.centroid(blocks.geometry.values)
        d = shapely.distance(centres, lines)
        blocks = blocks.assign(dist_m=d)
        within = blocks[blocks["dist_m"] <= radius]
        touch = shapely.intersects(within.geometry.values, exclusion)
        info[s] = {"blocks_within_corridor": int(len(within)),
                   "dropped_nakkhu_buffer": int(touch.sum())}
        cand[s] = within.loc[~touch].reset_index(drop=True)
    owner: dict[str, tuple[float, int, str]] = {}
    for i, s in enumerate(order):
        for bid, dist in zip(cand[s]["block_id"], cand[s]["dist_m"], strict=True):
            key = (float(dist), i, s)
            if bid not in owner or key < owner[bid]:
                owner[bid] = key
    out = {}
    for s in order:
        keep = np.array([owner[b][2] == s for b in cand[s]["block_id"]], dtype=bool)
        info[s]["shared_lost"] = int((~keep).sum())
        out[s] = cand[s].loc[keep].reset_index(drop=True)
        info[s]["n_blocks"] = int(len(out[s]))
    return out, info


# --- inventories over a site --------------------------------------------------------------------


def site_settings(project: Project, cfg: dict[str, Any], extent_path: str) -> Any:
    """The v9 sweep settings with the site extent and the routing-clip inventory paths."""
    from nakkhu.hydro.sweep import Settings, load_settings

    base, _ = load_settings(project, SWEEP)
    values = copy.deepcopy(base.v)
    values["analysis_extent"] = extent_path
    for name, path in cfg["inventories"].items():
        values["inventories"][name] = {**values["inventories"].get(name, {}), "path": path}
    values["phase2_site_inventories"] = True
    return Settings(project, ResolvedConfig(values=values, source=f"{SWEEP} (phase2 site)"))


def load_site_inventory(project: Project, cfg: dict[str, Any], site: str, name: str,
                        extent_path: str, greenhouses: Any | None = None
                        ) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """v9 ``load_inventory`` (centroid in the extent, greenhouse filter of the pinned PBF) over
    the site extent; ``building_uid`` = ``<site>/<inventory>:<i>``."""
    from nakkhu.hydro.sweep import load_greenhouses, load_inventory

    settings = site_settings(project, cfg, extent_path)
    extent = settings.read_polygon(extent_path)
    gh = load_greenhouses(settings) if greenhouses is None else greenhouses
    frame, info = load_inventory(settings, name, extent, gh)
    frame["building_uid"] = [f"{site}/{u}" for u in frame["building_uid"]]
    return frame, info


def _new_run(project: Project, command: str, config: str, **values: Any) -> Any:
    from nakkhu.runs import create_run

    return create_run(project, STAGE, ResolvedConfig(values={
        "command": command, "config": config,
        "config_sha256": sha256(project.paths.resolve(config)), **values}, source=config))


def _finish(project: Project, run: Any, summary: dict[str, Any], name: str,
            files: list[str] = ()) -> None:
    """Write the summary into the run and promote it with ``files`` (all from the run)."""
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json

    summary["run"] = run.ref
    atomic_write_json(run.path / name, summary)
    run.write_summary(summary)
    for f in [*files, name]:
        promote(run.path / f, f"{ART}/{f}", project.paths, replace=True)


# --- 1. sites -----------------------------------------------------------------------------------


def build_sites(project: Project, config: str = CONFIG) -> tuple[Any, dict[str, Any]]:
    from nakkhu.acquire import geo
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

    cfg = load_sites(project, config)
    run = create_run(project, STAGE, ResolvedConfig(values={"command": "p2-sites",
                                                            "config": config,
                                                            "config_sha256": sha256(
                                                                project.paths.resolve(config))},
                                                    source=config))
    ways = river_ways(project, cfg["rivers"])
    post = datamask_polygon(project, cfg, cfg["post"]["scene"])
    nakkhu = geo.read_geometry(project.paths.resolve(cfg["nakkhu_extent"]))
    exclusion = nakkhu.buffer(float(cfg["nakkhu_buffer_m"]))
    summary: dict[str, Any] = {"sites": {}, "files": {}}
    site_lines, overlaps = {}, {}
    with run.step("geometry"):
        for s, sc in cfg["sites"].items():
            pre = shapely.union_all(np.asarray([datamask_polygon(project, cfg, p)
                                                for p in sc["pre"]], dtype=object))
            overlap = shapely.make_valid(shapely.intersection(pre, post))
            ids = [int(i) for i in sc["way_ids"]]
            missing = sorted(set(ids) - set(ways["way_id"]))
            if missing:
                raise Phase2Error(f"site {s}: way ids {missing} not in {cfg['rivers']}")
            mine = ways[ways["way_id"].isin(ids)]
            clipped = shapely.intersection(shapely.union_all(mine.geometry.values), overlap)
            site_lines[s], overlaps[s] = clipped, overlap
            summary["sites"][s] = {"name": sc["name"], "role": sc["role"],
                                   "pre": list(sc["pre"]), "way_ids": ids,
                                   "overlap_km2": float(overlap.area / 1e6),
                                   "site_line_km_in_overlap": float(clipped.length / 1e3),
                                   "site_line_km_total": float(mine.length.sum() / 1e3)}
        blocks, binfo = corridor_blocks(site_lines, cfg, exclusion)
    bagmati = ways[ways["way_id"].isin([int(i) for i in cfg["bagmati_way_ids"]])]
    with run.step("buildings"):
        for s in cfg["sites"]:
            summary["sites"][s].update(binfo[s])
            b = blocks[s]
            extent = shapely.union_all(b.geometry.values)
            bag = shapely.intersection(shapely.union_all(bagmati.geometry.values), extent)
            main = shapely.union_all(np.asarray([site_lines[s], bag], dtype=object))
            lines_frame = gpd.GeoDataFrame(
                {"part": ["site", "bagmati"]}, geometry=[site_lines[s], bag], crs=CRS)
            lines_frame = lines_frame[~lines_frame.geometry.is_empty]
            names = {"lines": f"site_{s}_lines.geojson", "extent": f"site_{s}_extent.geojson",
                     "blocks": f"site_{s}_blocks.geojson",
                     "overlap": f"site_{s}_overlap.geojson"}
            atomic_write_text(run.path / names["lines"],
                              lines_frame.to_crs("EPSG:4326").to_json(drop_id=True))
            atomic_write_text(run.path / names["extent"], gpd.GeoDataFrame(
                {"site": [s]}, geometry=[extent], crs=CRS).to_crs("EPSG:4326").to_json(
                drop_id=True))
            atomic_write_text(run.path / names["blocks"],
                              b.to_crs("EPSG:4326").to_json(drop_id=True))
            atomic_write_text(run.path / names["overlap"], gpd.GeoDataFrame(
                {"site": [s]}, geometry=[overlaps[s]], crs=CRS).to_crs("EPSG:4326").to_json(
                drop_id=True))
            # the inventory reads the extent from the run until it is promoted
            frame, info = load_site_inventory(project, cfg, s, "microsoft",
                                              project.paths.relative(run.path / names["extent"]))
            cent = frame.geometry.centroid
            x, y = cent.x.to_numpy(), cent.y.to_numpy()
            table = gpd.GeoDataFrame(
                {"building_uid": frame["building_uid"].to_numpy(), "x": x, "y": y,
                 "block_id": block_of(x, y, cfg["lattice"]),
                 "area_m2": frame.geometry.area.to_numpy(dtype=float),
                 "in_overlap": shapely.within(shapely.points(x, y), overlaps[s])},
                geometry=frame.geometry.values, crs=CRS)
            names["buildings"] = f"site_{s}_buildings.parquet"
            table.to_parquet(run.path / names["buildings"], index=False, compression="zstd")
            summary["sites"][s].update({
                "extent_km2": float(extent.area / 1e6), "main_line_km": float(main.length / 1e3),
                "bagmati_km_in_extent": float(bag.length / 1e3),
                "n_buildings": int(len(table)),
                "n_buildings_in_overlap": int(table["in_overlap"].sum()),
                "n_blocks_with_building": int(table["block_id"].nunique()),
                "inventory": info})
            for name in names.values():
                summary["files"][name] = sha256(run.path / name)
    atomic_write_json(run.path / "sites_summary.json", summary)
    run.write_summary(summary)
    for name in [*summary["files"], "sites_summary.json"]:
        promote(run.path / name, f"{ART}/{name}", project.paths, replace=True)
    return run, summary


def site_geometry(project: Project, site: str) -> dict[str, Any]:
    """The promoted site files (extent, lines, blocks, buildings)."""
    from nakkhu.acquire import geo

    out: dict[str, Any] = {}
    for key, name in (("extent", f"site_{site}_extent.geojson"),
                      ("lines", f"site_{site}_lines.geojson"),
                      ("blocks", f"site_{site}_blocks.geojson")):
        p = art_path(project, name)
        if not p.is_file():
            raise Phase2Error(f"{rel_art(name)} missing (run `nakkhu followup p2 sites`)")
        out[f"{key}_path"] = rel_art(name)
    out["extent"] = geo.read_geometry(project.paths.resolve(out["extent_path"]))
    lines = gpd.read_file(project.paths.resolve(out["lines_path"])).to_crs(CRS)
    out["lines"] = lines
    out["main"] = shapely.union_all(lines.geometry.values)
    b = art_path(project, f"site_{site}_buildings.parquet")
    out["buildings"] = gpd.read_parquet(b) if b.is_file() else None
    return out


# --- 2. OSM footprints over the sites (pinned PBF) ----------------------------------------------


def osm_footprints(project: Project, config: str = CONFIG) -> dict[str, Any]:
    from nakkhu.acquire import geo
    from nakkhu.acquire.fetchers import _polygonal, osm_buildings

    cfg = load_sites(project, config)
    run = _new_run(project, "p2-osm", config)
    pbf = project.paths.resolve(cfg["osm_pbf"])
    if not pbf.is_file():
        raise Phase2Error(f"{pbf} missing (the pinned PBF)")
    clip = geo.clip_geometry(project, "routing")
    frame, stamp = osm_buildings(pbf, geo.lonlat_bounds(clip))
    frame = frame.to_crs(CRS)
    frame["geometry"] = shapely.make_valid(frame.geometry.values)
    frame["geometry"] = frame.geometry.apply(_polygonal)
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
    frame = frame[frame.geometry.intersects(clip)].reset_index(drop=True)
    frame["release"] = stamp or pbf.name
    dest = project.paths.resolve(cfg["inventories"]["osm"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[["source", "source_id", "confidence", "release", "building", "osm_version",
                   "osm_timestamp", "geometry"]]
    with run.step("osm"):
        frame.to_parquet(dest, index=False, compression="zstd")
    out = {"file": project.paths.relative(dest), "features": len(frame),
           "pbf": pbf.name, "pbf_sha256": sha256(pbf), "sha256": sha256(dest),
           "replication_timestamp": stamp}
    _finish(project, run, out, "site_osm_footprints.json")
    return out


# --- 3. site mosaics ------------------------------------------------------------------------------


def site_grid(project: Project, site: str, cfg: dict[str, Any]) -> Any:
    from nakkhu import tiling

    tcfg = load_yaml(project.paths.resolve(tiling.TILING_CONFIG))
    g = tcfg["grid"]
    geom = site_geometry(project, site)["extent"].buffer(float(cfg["grid_margin_m"]))
    grid = tiling.compute_grid(geom, g["crs"], float(g["pixel_size_m"]), int(g["tile_px"]))
    tile_m = grid.tile_px * grid.pixel_size
    for v, o in ((grid.west, float(cfg["lattice"]["west"])),
                 (grid.north, float(cfg["lattice"]["north"]))):
        if abs(((v - o) / tile_m) - round((v - o) / tile_m)) > 1e-9:
            raise Phase2Error("site grid is not on the plan 003 lattice")
    return grid


def mosaic_paths(project: Project, site: str) -> dict[str, Path]:
    base = project.paths.data_root / "interim" / "followup" / "phase2" / "mosaic"
    return {"pre": base / f"site_{site}_pre_0p5m.tif", "post": base / f"site_{site}_post_0p5m.tif"}


def build_mosaics(project: Project, site: str, config: str = CONFIG,
                  workers: int = 2) -> dict[str, Any]:
    from nakkhu import tiling
    from nakkhu.runs import atomic_write_json

    cfg = load_sites(project, config)
    run = _new_run(project, "p2-mosaic", config, site=site)
    sc = cfg["sites"][site]
    grid = site_grid(project, site, cfg)
    tcfg = load_yaml(project.paths.resolve(tiling.TILING_CONFIG))
    mcfg = {**tcfg["mosaic"], "workers": int(workers)}
    paths = mosaic_paths(project, site)
    out: dict[str, Any] = {"site": site, "grid": grid.section("phase2 site extent"),
                           "mosaics": {}}
    for kind, scenes in (("pre", list(sc["pre"])), ("post", [cfg["post"]["scene"]])):
        files = [f for scene in scenes for f in scene_files(project, scene)]
        with run.step(f"mosaic-{kind}"):
            rec = tiling.write_mosaic(project, files, grid, paths[kind], mcfg,
                                      label=f"site {site} {kind}")
        rec["scenes"] = scenes
        rec["sha256"] = sha256(paths[kind])
        rec.pop("histograms", None)
        out["mosaics"][kind] = rec
        atomic_write_json(paths[kind].with_suffix(".json"), rec)
    _finish(project, run, out, f"site_{site}_mosaics.json")
    return out


# --- 4. footprint registration on the site tiles ------------------------------------------------


def _register_one(args: tuple[Any, ...]) -> tuple[float, float, float, str]:
    import rasterio
    from rasterio.windows import Window

    from nakkhu.labels.register import register_tile

    mosaic, col0, row0, size, wkb, reg = args
    with rasterio.open(mosaic) as src:
        win = Window(col0, row0, size, size)
        rgb = np.moveaxis(src.read([1, 2, 3], window=win, boundless=True, fill_value=0), 0, -1)
        valid = src.read_masks(1, window=win, boundless=True) > 0
    polys = shapely.from_wkb(np.asarray(wkb, dtype=object)) if len(wkb) else np.array([])
    r = register_tile(rgb, valid, polys, max_shift_px=float("inf"), min_peak_ratio=0.0,
                      sigma_px=float(reg["sigma_px"]), exclusion_px=int(reg["exclusion_px"]))
    return r.raw_dx, r.raw_dy, r.peak_ratio, r.reason


def site_tiles(project: Project, site: str, cfg: dict[str, Any], grid: Any) -> pd.DataFrame:
    """320 m lattice tiles of the site grid that intersect the site extent; ``tile_id`` on the
    plan 003 lattice (t_<col>_<row> from the grid origin), ``affine`` / ``bounds`` as the plan
    003 index."""
    lat = cfg["lattice"]
    tile_m = float(lat["tile_m"])
    extent = site_geometry(project, site)["extent"]
    rows = []
    for r in range(grid.height_px // grid.tile_px):
        for c in range(grid.width_px // grid.tile_px):
            x0 = grid.west + c * tile_m
            y0 = grid.north - r * tile_m
            b = shapely.box(x0, y0 - tile_m, x0 + tile_m, y0)
            if not b.intersects(extent):
                continue
            gc = int(round((x0 - float(lat["west"])) / tile_m))
            gr = int(round((float(lat["north"]) - y0) / tile_m))
            rows.append({"tile_id": f"t_{gc:04d}_{gr:04d}", "col": gc, "row": gr,
                         "x0_px": c * grid.tile_px, "y0_px": r * grid.tile_px,
                         "affine": np.array([grid.pixel_size, 0.0, x0, 0.0, -grid.pixel_size,
                                             y0]),
                         "bounds": np.array([x0, y0 - tile_m, x0 + tile_m, y0]),
                         "crs": CRS, "domain": f"phase2_{site}", "low_valid": False})
    return pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)


def _footprint_geoms(project: Project, path: str, clip: Any) -> np.ndarray:
    frame = gpd.read_parquet(project.paths.resolve(path)).to_crs(CRS)
    geoms = frame.geometry.values
    geoms = geoms[shapely.intersects(geoms, clip)]
    parts = shapely.get_parts(np.asarray(geoms, dtype=object))
    return parts[~shapely.is_empty(parts)]


def register_site(project: Project, site: str, config: str = CONFIG, workers: int = 2
                  ) -> dict[str, Any]:
    from nakkhu.labels.build import _pool_map, decide_registration
    from nakkhu.labels.tiles import to_px

    cfg = load_sites(project, config)
    run = _new_run(project, "p2-register", config, site=site)
    rcfg = cfg["registration"]
    reg = load_yaml(project.paths.resolve(rcfg["settings"]))["registration"]
    grid = site_grid(project, site, cfg)
    mosaic = mosaic_paths(project, site)["pre"]
    if not mosaic.is_file():
        raise Phase2Error(f"{mosaic} missing (run `nakkhu followup p2 mosaic --site {site}`)")
    index = site_tiles(project, site, cfg, grid)
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    clip = shapely.box(*shapely.total_bounds(boxes))
    out: dict[str, Any] = {"site": site, "tiles": int(len(index))}
    for kind, sources, ref in (
            ("footprints", ("google_v3_uncut", "microsoft"), rcfg["footprints_reference"]),
            ("osm", ("osm",), rcfg["osm_reference"])):
        geoms = np.concatenate([_footprint_geoms(project, cfg["inventories"][s], clip)
                                for s in sources])
        ti, gi = shapely.STRtree(geoms).query(boxes, predicate="intersects")
        order = np.argsort(ti, kind="stable")
        ti, gi = ti[order], gi[order]
        cuts = np.searchsorted(ti, np.arange(len(index) + 1))
        items = []
        for i in range(len(index)):
            px = to_px(geoms[gi[cuts[i]:cuts[i + 1]]], index.iloc[i]["affine"])
            items.append((str(mosaic), int(index.iloc[i]["x0_px"]), int(index.iloc[i]["y0_px"]),
                          int(rcfg["tile_px"]), list(shapely.to_wkb(px)), reg))
        raw = pd.DataFrame(_pool_map(_register_one, items, workers),
                           columns=["raw_dx", "raw_dy", "peak_ratio", "reason"])
        raw.insert(0, "tile_id", index["tile_id"].to_numpy())
        refs = pd.read_csv(project.paths.resolve(ref))["min_peak_ratio"].unique()
        if len(refs) != 1:
            raise Phase2Error(f"{ref}: expected one min_peak_ratio, got {refs}")
        shifts = decide_registration(raw, reg, float(refs[0]))
        if kind == "osm":
            shifts.insert(1, "n_osm", [len(it[4]) for it in items])
        shifts["domain"] = index["domain"].to_numpy()
        shifts["low_valid"] = False
        name = f"site_{site}_{'osm_' if kind == 'osm' else ''}registration_shifts.csv"
        dest = run.path / name
        with run.step(f"register-{kind}"):
            shifts.to_csv(dest, index=False, lineterminator="\n")
        out[kind] = {"file": rel_art(name), "sha256": sha256(dest),
                     "threshold": float(refs[0]), "threshold_from": ref,
                     "n_ok": int(shifts["reg_ok"].sum()),
                     "reasons": shifts["reason"].value_counts().to_dict()}
    tname = f"site_{site}_tiles.parquet"
    index.to_parquet(run.path / tname, index=False)
    out["tile_index"] = {"file": rel_art(tname), "sha256": sha256(run.path / tname)}
    names = [tname, *(v["file"].rsplit("/", 1)[1] for k, v in out.items()
                      if k in ("footprints", "osm"))]
    _finish(project, run, out, f"site_{site}_registration.json", names)
    return out


# --- 5. co-registration gate (change-map Step 1 only) -------------------------------------------


def site_change_values(project: Project, site: str, config: str = CONFIG) -> dict[str, Any]:
    """The v9 change config with the site's region, scenes and input paths; every rule section
    and every building rule key is the v9 value (checked)."""
    v9 = load_yaml(project.paths.resolve(CHANGE))
    values = copy.deepcopy(v9)
    paths = mosaic_paths(project, site)
    cfg = load_sites(project, config)
    values["region"] = f"phase2_site_{site}"
    values["scenes"] = {
        "pre": {"id": f"phase2_{site}_pre", "mosaic": project.paths.relative(paths["pre"])},
        "post": [{"id": f"phase2_{site}_post", "strip": str(cfg["post"]["strip"]),
                  "mosaic": project.paths.relative(paths["post"])}]}
    values["buildings"]["inventories"] = ["microsoft"]
    values["buildings"]["phase2"] = {
        "site": site, "sites_config": config,
        "tile_index": rel_art(f"site_{site}_tiles.parquet"),
        "registration": {"footprints": f"{ART}/site_{site}_registration_shifts.csv",
                         "osm": f"{ART}/site_{site}_osm_registration_shifts.csv"}}
    for key in RULE_SECTIONS:
        if values[key] != v9[key]:
            raise Phase2Error(f"site change config changes the v9 rule section {key}")
    return values


class SiteChangeMap:
    """:class:`nakkhu.flood.pipeline.ChangeMap` over one Phase-2 site (built lazily so that
    importing this module does not pull in the change-map code)."""

    def __new__(cls, project: Project, site: str, config: str = CONFIG):
        from nakkhu.flood.pipeline import ChangeMap

        class _Site(ChangeMap):
            def __init__(self) -> None:
                self.project = project
                values = site_change_values(project, site, config)
                self.cfg = ResolvedConfig(values=values,
                                          source=f"{CHANGE} (phase2 site {site})")
                self.v = values
                self.timings = {}
                self.site = site
                self.sites_cfg = load_sites(project, config)
                aoi = project.config.values.setdefault("aoi", {})
                aoi[values["region"]] = site_geometry(project, site)["extent_path"]

            def load_buildings(self) -> None:
                import shapely as shp

                from nakkhu.audit.frames import apply_shifts
                from nakkhu.flood.buildings import SHIFT_KIND
                from nakkhu.hydro.sweep import load_greenhouses

                bcfg = self.v["buildings"]
                p2 = bcfg["phase2"]
                index = pd.read_parquet(project.paths.resolve(p2["tile_index"]))
                shifts = {k: pd.read_csv(project.paths.artifacts / rel).set_index("tile_id")[
                    ["dx", "dy", "reg_ok"]] for k, rel in p2["registration"].items()}
                extent_path = site_geometry(project, site)["extent_path"]
                gh = load_greenhouses(site_settings(project, self.sites_cfg, extent_path))
                self.inv_source, self.pending = {}, []
                self.inv, self.inv_info = {}, {}
                for name in dict.fromkeys([*bcfg["inventories"], *bcfg["ground_mask_from"]]):
                    path = project.paths.resolve(self.sites_cfg["inventories"][name])
                    if not path.is_file():
                        raise Phase2Error(f"site inventory {name} missing ({path})")
                    self.inv_source[name] = path
                    frame, meta = load_site_inventory(project, self.sites_cfg, site, name,
                                                      extent_path, gh)
                    geoms = np.asarray(frame.geometry.values, dtype=object)
                    kind = SHIFT_KIND[name]
                    moved = geoms if kind is None else apply_shifts(geoms, index, shifts[kind])
                    self.inv[name] = gpd.GeoDataFrame(
                        {"building_id": frame["building_uid"].to_numpy(),
                         **{c: frame[c].to_numpy() for c in ("source_id", "confidence")
                            if c in frame.columns}},
                        geometry=gpd.GeoSeries(moved, crs=frame.crs))
                    self.inv_info[name] = {k: v for k, v in meta.items()
                                           if k != "cut_thresholds"}
                    self.inv_info[name]["registration"] = kind or "none"
                self.inventories = list(bcfg["inventories"])
                built = [g for n in bcfg["ground_mask_from"] for g in self.inv[n].geometry.values]
                self.built = np.asarray(built, dtype=object)
                self.built_tree = shp.STRtree(self.built)

        return _Site()


def coreg_gate(project: Project, site: str, config: str = CONFIG) -> tuple[Any, dict[str, Any]]:
    """Steps setup + masks + co-registration of the v9 change map for one site, label-free."""
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, create_run

    cm = SiteChangeMap(project, site, config)
    run = create_run(project, STAGE, cm.cfg)
    summary: dict[str, Any] = {"site": site, "config_sha256": cm.cfg.sha256,
                               "label_free": "setup, observability masks and co-registration "
                                             "only; no building feature, score or class"}
    with run.step("setup"):
        cm.open_grid()
        cm.timed("buildings", cm.load_buildings)
        summary["blocks"] = len(cm.blocks)
        summary["inventories"] = cm.inv_info
    with run.step("masks"):
        stats = cm.timed("masks", lambda: cm.build_masks(run))
        atomic_write_json(run.path / "observable_mask_stats.json", stats)
    with run.step("coregister"):
        coreg = cm.timed("coregister", cm.coregister)
        coreg.to_csv(run.path / "coregistration.csv", index=False)
        limit = float(cm.v["coregistration"]["verify_max_median_px"])
        out = {}
        for s in cm.strips:
            est = coreg[(coreg["strip"] == s) & (coreg["set"] == "estimate")]
            held = coreg[(coreg["strip"] == s) & (coreg["set"] == "held_back")
                         & coreg["accepted"]]
            med = float(held["residual_px"].median()) if len(held) else float("nan")
            out[s] = {"blocks": int(len(est)), "blocks_accepted": int(est["accepted"].sum()),
                      "median_shift_px": float(np.median(np.hypot(est["applied_dy_px"],
                                                                  est["applied_dx_px"]))),
                      "held_back": int(((coreg["strip"] == s)
                                        & (coreg["set"] == "held_back")).sum()),
                      "held_back_accepted": int(len(held)),
                      "held_back_median_residual_px": med,
                      "verify_max_median_px": limit,
                      "verify_pass": bool(np.isfinite(med) and med <= limit)}
        summary["coregistration"] = out
    summary["timings_s"] = cm.timings
    atomic_write_json(run.path / "coreg_summary.json", summary)
    run.write_summary(summary)
    for name in ("coreg_summary.json", "coregistration.csv", "observable_mask_stats.json"):
        promote(run.path / name, f"{ART}/site_{site}_{name}", project.paths, replace=True)
    return run, summary


# --- 6. site predictors -------------------------------------------------------------------------


FEATURE_KEYS = ("slope_deg", "curv_plan", "curv_profile", "tpi_150", "tpi_500", "twi",
                "log10_area_cell_km2", "log10_area_first_drop_km2", "log10_area_main_km2")


def site_predictors(project: Project, sites: list[str], config: str = CONFIG
                    ) -> tuple[Any, dict[str, Any]]:
    from nakkhu.artifacts import promote
    from nakkhu.followup import prep as P
    from nakkhu.followup import surfaces as F
    from nakkhu.hydro.dem import read_raster
    from nakkhu.runs import atomic_write_json, create_run

    cfg = load_sites(project, config)
    spec = S.load_spec(project, cfg["addendum"])
    resolved = ResolvedConfig(values={"command": "p2-predictors", "config": config,
                                      "config_sha256": sha256(project.paths.resolve(config)),
                                      "sites": sites}, source=config)
    run = create_run(project, STAGE, resolved)
    sources = {name: P.DemSource(project, name, entry) for name, entry in cfg["dems"].items()}
    fdem = cfg["feature_dem"]
    v9 = sources[fdem]
    primary = read_raster(v9.path(f"{v9.combo}__tdrop", "hand.tif")).astype(np.float64).ravel()
    summary: dict[str, Any] = {"dems": {k: s.meta() for k, s in sources.items()}, "sites": {}}
    files = {}
    for site in sites:
        g = site_geometry(project, site)
        b = g["buildings"]
        if b is None:
            raise Phase2Error(f"site {site}: building table missing")
        x, y = b["x"].to_numpy(dtype=float), b["y"].to_numpy(dtype=float)
        points = {site: (x, y)}
        lines = g["lines"]
        with run.step(f"site-{site.lower()}"):
            cols: dict[str, dict[str, np.ndarray]] = {}
            info: dict[str, Any] = {}
            for name, src in sources.items():
                c, drains, _, dinfo = P.dem_columns(src, spec, lines, points,
                                                    features=name == fdem)
                cols[name] = c[site]
                info[name] = {"main_cells": dinfo["main_cells"],
                              "drain_shares": [d for d in drains if d["is_drop_threshold"]]}
            cells = F.point_cells(v9.grid, x, y)
            dist = shapely.distance(shapely.points(x, y), g["main"])
            table = {"building_uid": b["building_uid"].to_numpy(), "x": x, "y": y,
                     "block_id": b["block_id"].to_numpy(),
                     "in_overlap": b["in_overlap"].to_numpy(dtype=bool),
                     "hand_v9": F.sample(primary, cells), "distance": dist,
                     "area_m2": b["area_m2"].to_numpy(dtype=float)}
            gcols = cols[fdem]
            for k in FEATURE_KEYS:
                table[k] = gcols.pop(k)
            table["elev_glo30"] = gcols[f"elev__{fdem}"]
            table["elev_fabdem"] = cols["fabdem"]["elev__fabdem"]
            table["dsm_excess"] = table["elev_glo30"] - table["elev_fabdem"]
            table["density_100m"] = F.count_within(x, y, P.DENSITY_RADIUS_M).astype(float)
            for name in sources:
                for k, v in cols[name].items():
                    if k not in table:
                        table[k] = v
            frame = pd.DataFrame(table)
            name = f"site_{site}_predictors.parquet"
            frame.to_parquet(run.path / name, index=False, compression="zstd")
            files[name] = sha256(run.path / name)
            from nakkhu.followup import ml

            X = ml.matrix(frame)
            finite_all = np.all(np.isfinite(X), axis=1)
            hmr = frame[f"hand_mr_d8__{fdem}"].to_numpy(dtype=float)
            summary["sites"][site] = {
                "n_buildings": int(len(frame)), **{f"{k}_main_cells": v["main_cells"]
                                                   for k, v in info.items()},
                "hand_mr_d8_nan": int(np.isnan(hmr).sum()),
                "hand_mr_d8_reach_share": float(np.isfinite(hmr).mean()) if len(hmr) else None,
                "hand_v9_nan": int(np.isnan(table["hand_v9"]).sum()),
                "ml_rows_all_28_finite": int(finite_all.sum()),
                "ml_rows_nan": int((~finite_all).sum()),
                "hand_mr_d8_sha256": array_sha(hmr),
                "drain_shares_drop_threshold": info[fdem]["drain_shares"],
                "file": name, "sha256": files[name]}
    summary["run"] = run.ref
    summary["note"] = ("the predictor tables stay in the run (over the artifact size limit); "
                       "this summary pins their sha256")
    atomic_write_json(run.path / "predictors_summary.json", summary)
    run.write_summary(summary)
    promote(run.path / "predictors_summary.json",
            f"{ART}/site_predictors_{'_'.join(sites)}.json", project.paths, replace=True)
    return run, summary


# --- 7. site change map (Phase-2 labels: only after prereg_v9_a4 is frozen) ---------------------

A4 = "prereg_v9_a4"


def change_site(project: Project, site: str, config: str = CONFIG) -> tuple[Any, dict[str, Any]]:
    """The v9 change map over one site (labels). Refuses unless ``prereg_v9_a4`` is frozen,
    ``prereg verify`` is clean and the site's resolved change config has the sha256 that a4
    pins (``labels.site_files.<site>.config_sha256``). Outputs stay in the run (no promotion
    over ``artifacts/flood/``)."""
    from nakkhu.evaluate import guard
    from nakkhu.evaluate.addendum import addendum_path, frozen_addenda, load_addendum

    if A4 not in frozen_addenda(project):
        raise Phase2Error(f"{A4} is not frozen: no Phase-2 label may be built")
    problems = guard.verify(project)["problems"]
    if problems:
        raise Phase2Error("prereg verify: " + "; ".join(problems))
    doc = load_addendum(addendum_path(project, A4))
    want = (((doc.get("labels") or {}).get("site_files") or {}).get(site) or {}).get(
        "config_sha256")
    cm = SiteChangeMap(project, site, config)
    if not want or cm.cfg.sha256 != want:
        raise Phase2Error(f"site {site}: change config sha256 {cm.cfg.sha256[:12]} is not the "
                          f"{A4} pin {str(want)[:12]}")
    return cm.execute()
