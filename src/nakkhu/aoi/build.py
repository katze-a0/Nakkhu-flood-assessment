"""``nakkhu aoi build``: rebuild the derived AOI files into a run and compare with the registered.

The chain is lines -> catchment_c4 -> analysis_extent and lines -> routing_domain. With
``upstream="rebuilt"`` (default) every product is built from raw inputs only; ``"registered"``
feeds the registered upstream files instead, which isolates one step when a chain differs.
Registered files in ``configs/aoi/`` are never written here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd

from nakkhu.aoi import BUILD_CONFIG, PRODUCTS, compare
from nakkhu.aoi import basins as basinmod
from nakkhu.aoi import extent as extentmod
from nakkhu.aoi import lines as linesmod
from nakkhu.aoi import sites as sitesmod
from nakkhu.config import Project, ResolvedConfig, load_config, load_yaml
from nakkhu.manifest import sha256_file
from nakkhu.runs import Run, atomic_write_text, create_run

DEPENDS = {"catchment_c4": ("osm_lines",), "routing_domain": ("osm_lines",),
           "analysis_extent": ("catchment_c4",)}
UPSTREAM = ("rebuilt", "registered")


class AoiBuildError(RuntimeError):
    """Unknown product or invalid settings."""


def load_settings(project: Project, path: str = BUILD_CONFIG,
                  upstream: str = "rebuilt") -> ResolvedConfig:
    config = load_config(project.paths.resolve(path))
    values = dict(config.values)
    values["upstream"] = upstream
    return ResolvedConfig(values=values, source=project.paths.relative(
        project.paths.resolve(path)))


def closure(products: Iterable[str], upstream: str) -> list[str]:
    """Requested products plus (for ``rebuilt``) everything they depend on, in build order."""
    wanted = set(products)
    unknown = wanted - set(PRODUCTS)
    if unknown:
        raise AoiBuildError(f"unknown AOI products {sorted(unknown)}; choose from {PRODUCTS}")
    if upstream not in UPSTREAM:
        raise AoiBuildError(f"upstream must be one of {UPSTREAM}")
    if upstream == "rebuilt":
        frontier = list(wanted)
        while frontier:
            deps = DEPENDS.get(frontier.pop(), ())
            frontier.extend(d for d in deps if d not in wanted)
            wanted.update(deps)
    return [p for p in PRODUCTS if p in wanted]


def _write(frame: gpd.GeoDataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_file(tmp, driver="GeoJSON")
    tmp.replace(path)


class _Builder:
    def __init__(self, project: Project, config: ResolvedConfig, run: Run) -> None:
        self.p = project
        self.v = config.values
        self.run = run
        self.tol = self.v["tolerance"]
        self.work = self.v["work_crs"]
        self.frames: dict[str, gpd.GeoDataFrame] = {}
        self.results: dict[str, Any] = {}

    def resolve(self, value: str) -> Path:
        return self.p.paths.resolve(value)

    def registered(self, product: str) -> gpd.GeoDataFrame:
        return gpd.read_file(self.resolve(self.v["registered"][product]))

    def upstream(self, product: str) -> gpd.GeoDataFrame:
        if self.v["upstream"] == "registered" or product not in self.frames:
            return self.registered(product)
        return self.frames[product]

    def finish(self, product: str, frame: gpd.GeoDataFrame, record: dict[str, Any],
               kind: str) -> None:
        name = Path(self.v["registered"][product]).name
        out = self.run.path / name
        _write(frame, out)
        self.frames[product] = frame
        reg = self.registered(product)
        written = gpd.read_file(out)  # compare the file as written (GeoJSON precision)
        if kind == "polygon":
            cmp = compare.polygons(written.geometry.union_all(), reg.geometry.union_all(),
                                   self.work, self.tol)
        elif kind == "lines":
            cmp = compare.lines(written, reg, self.work, self.tol)
        else:
            cmp = compare.points(written, reg, self.work, self.tol)
        cmp["properties_equal"] = (written.drop(columns="geometry").to_dict("records")
                                   == reg.drop(columns="geometry").to_dict("records"))
        self.results[product] = {"output": name, "sha256": sha256_file(out),
                                 "registered": self.v["registered"][product],
                                 "registered_sha256": sha256_file(
                                     self.resolve(self.v["registered"][product])),
                                 "build": record, "comparison": cmp}

    # --- products ----------------------------------------------------------------------------

    def waterways_rivers(self) -> None:
        cfg = self.v["waterways_rivers"]
        pbf = self.resolve(self.v["inputs"]["osm_pbf"])
        doc = linesmod.ways_in_bbox(pbf, cfg["tags"], cfg["bbox"],
                                    self.v["inputs"].get("osm_pbf_timestamp", ""))
        name = Path(self.v["registered"]["waterways_rivers"]).name
        out = self.run.path / name
        atomic_write_text(out, json.dumps(doc, ensure_ascii=False))
        reg_path = self.resolve(self.v["registered"]["waterways_rivers"])
        cmp = compare.overpass(doc, json.loads(reg_path.read_text(encoding="utf-8")))
        self.results["waterways_rivers"] = {
            "output": name, "sha256": sha256_file(out),
            "registered": self.v["registered"]["waterways_rivers"],
            "registered_sha256": sha256_file(reg_path),
            "build": {"pbf_sha256": sha256_file(pbf), "bbox": cfg["bbox"], "tags": cfg["tags"],
                      "query": cfg["query"]},
            "comparison": cmp}

    def osm_lines(self) -> None:
        cfg = self.v["osm_lines"]
        frame, record = linesmod.read_ways(self.resolve(self.v["inputs"]["osm_pbf"]),
                                           cfg["ways"], cfg.get("check_names") or (),
                                           self.v["dem_window"])
        record["pbf_sha256"] = sha256_file(self.resolve(self.v["inputs"]["osm_pbf"]))
        self.finish("osm_lines", frame, record, "lines")

    def catchment_c4(self) -> None:
        poly, record = basinmod.catchment_c4(self.resolve(self.v["inputs"]["glo30_tile"]),
                                             self.upstream("osm_lines"), self.v)
        frame = gpd.GeoDataFrame([{
            "name": "c4_glo30_osm_burned",
            "source": "Copernicus GLO-30 N27E085 with OSM Nakkhu/Nallu/Lele/Bagmati lines burned "
                      "60 m (all_touched), pyflwdir from_dem, outlet = burned Nakkhu cell <300 m "
                      "from OSM confluence",
            "area_km2": round(record["area_km2"], 2), "geometry": poly}], crs="EPSG:4326")
        self.finish("catchment_c4", frame, record, "polygon")

    def routing_domain(self) -> None:
        poly, record = basinmod.routing_domain(self.resolve(self.v["inputs"]["glo30_tile"]),
                                               self.upstream("osm_lines"), self.v)
        frame = gpd.GeoDataFrame({"name": ["upper_bagmati_below_nakkhu_confluence_glo30"],
                                  "area_km2": [round(record["area_km2"], 1)]},
                                 geometry=[poly], crs="EPSG:4326")
        self.finish("routing_domain", frame, record, "polygon")

    def analysis_extent(self) -> None:
        cfg = self.v["analysis_extent"]
        stac = self.v["inputs"]["stac_items"]
        pre = extentmod.footprint([self.resolve(d) for d in stac["pre"]], cfg["pre_catalogs"],
                                  self.work)
        post = extentmod.footprint([self.resolve(d) for d in stac["post"]],
                                   cfg["post_catalogs"], self.work)
        catchment = self.upstream("catchment_c4").geometry.union_all()
        poly, record = extentmod.analysis_extent(catchment, pre, post, self.v)
        lon, lat = self.v["confluence"]
        frame = gpd.GeoDataFrame({
            "name": ["recommended_analysis_extent"],
            "definition": [f"(GLO-30 OSM-burned Nakkhu catchment c4 ∪ 1 km radius around OSM "
                           f"Nakkhu-Bagmati confluence {lon:.6f}E {lat:.6f}N) ∩ Maxar WV02 "
                           "10300100F5AB8000 (2024-01-13) ∩ GE01 105001003E1F0500/0600 "
                           "(2024-10-06) data footprints"],
            "area_km2": [round(record["area_km2"], 2)]}, geometry=[poly], crs="EPSG:4326")
        self.finish("analysis_extent", frame, record, "polygon")

    def flood_sites(self) -> None:
        path = self.resolve(self.v["flood_sites"])
        frame = sitesmod.site_frame(load_yaml(path))
        self.finish("flood_sites", frame, {"config": self.v["flood_sites"],
                                           "config_sha256": sha256_file(path)}, "points")


def build(project: Project, products: Iterable[str] = PRODUCTS, path: str = BUILD_CONFIG,
          upstream: str = "rebuilt") -> tuple[Run, dict[str, Any]]:
    order = closure(products, upstream)
    config = load_settings(project, path, upstream)
    run = create_run(project, "aoi", config)
    builder = _Builder(project, config, run)
    for product in order:
        with run.step(product):
            getattr(builder, product)()
    summary = {"upstream": upstream, "products": builder.results,
               "all_match": all(r["comparison"]["match"] for r in builder.results.values())}
    run.write_summary(summary)
    return run, summary
