"""``nakkhu aoi reach``: share of the 2024 flooded Nakkhu reach inside the analysis extent.

Settings and the reach definition: ``configs/aoi/reach.yaml``. The OSM main-stem ways are merged
into one line; each reach runs along that line from the point nearest its upstream site to the
point nearest the confluence. The share is length(reach within extent) / length(reach) in the
work CRS. The run's ``summary.json`` is promoted to ``artifacts/aoi/reach_share.json``.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, substring

from nakkhu.aoi.sites import site_frame
from nakkhu.artifacts import promote
from nakkhu.config import Project, ResolvedConfig, load_config, load_yaml
from nakkhu.manifest import sha256_file
from nakkhu.runs import Run, create_run

REACH_CONFIG = "configs/aoi/reach.yaml"
ARTIFACT = "aoi/reach_share.json"


class ReachError(RuntimeError):
    """The reach cannot be defined from the inputs."""


def merged_stem(lines: gpd.GeoDataFrame, names: list[str]) -> LineString:
    """Merge the named ways into one line; refuse gaps or branches."""
    parts = lines[lines["name"].isin(names)]
    missing = sorted(set(names) - set(parts["name"]))
    if missing:
        raise ReachError(f"no OSM ways named {missing}")
    merged = linemerge(MultiLineString([g for geom in parts.geometry
                                        for g in getattr(geom, "geoms", [geom])]))
    if not isinstance(merged, LineString):
        n = len(getattr(merged, "geoms", []))
        raise ReachError(f"stem ways {names} merge into {n} pieces, not one line (gap or branch)")
    return merged


def _length_inside(line: BaseGeometry, extent: BaseGeometry) -> float:
    return float(line.intersection(extent).length)


def reach_share(stem: LineString, extent: BaseGeometry, down: Any, up: Any,
                end_uncertainty_m: float, max_offset_m: float) -> dict[str, Any]:
    """Reach between the stem points nearest ``up`` and ``down`` (points in the stem CRS)."""
    offsets = {"upstream_site": float(up.distance(stem)), "downstream_site": float(down.distance(stem))}
    for key, offset in offsets.items():
        if offset > max_offset_m:
            raise ReachError(f"{key} lies {offset:.1f} m from the stem (> {max_offset_m} m)")
    pos_down = float(stem.project(down))
    pos_up = float(stem.project(up))
    if abs(pos_down - pos_up) < 1.0:
        raise ReachError("upstream and downstream sites project to the same stem point")
    direction = 1.0 if pos_up < pos_down else -1.0  # +1: the line runs downstream

    def measure(position_up: float) -> dict[str, float]:
        position_up = min(max(position_up, 0.0), stem.length)
        piece = substring(stem, min(position_up, pos_down), max(position_up, pos_down))
        inside = _length_inside(piece, extent)
        return {"length_m": float(piece.length), "inside_m": inside,
                "share": inside / piece.length if piece.length else float("nan")}

    result: dict[str, Any] = {
        "site_offset_from_stem_m": offsets,
        "upstream_position_m": pos_up, "downstream_position_m": pos_down,
        **measure(pos_up),
    }
    if end_uncertainty_m > 0:
        # moving the upstream end upstream by d lengthens the reach
        longer = measure(pos_up - direction * end_uncertainty_m)
        shorter = measure(pos_up + direction * end_uncertainty_m)
        result["end_uncertainty_m"] = end_uncertainty_m
        result["share_range"] = sorted([longer["share"], shorter["share"]])
        result["length_m_range"] = sorted([longer["length_m"], shorter["length_m"]])
    return result


def run_reach(project: Project, path: str = REACH_CONFIG, *,
              do_promote: bool = True) -> tuple[Run, dict[str, Any]]:
    config: ResolvedConfig = load_config(project.paths.resolve(path))
    values = config.values
    work = values["work_crs"]
    inputs = {key: project.paths.resolve(value) for key, value in values["inputs"].items()}
    run = create_run(project, "aoi-reach", config)
    with run.step("reach"):
        lines = gpd.read_file(inputs["lines"]).to_crs(work)
        sites = site_frame(load_yaml(inputs["sites"])).to_crs(work).set_index("name")
        extent = gpd.read_file(inputs["extent"]).to_crs(work).geometry.union_all()
        stem = merged_stem(lines, list(values["stem_names"]))

        def site(name: str) -> Any:
            if name not in sites.index:
                raise ReachError(f"site {name!r} not in {values['inputs']['sites']}")
            return sites.loc[name, "geometry"]

        down = site(values["downstream_site"])
        reaches = {}
        for key, spec in values["reaches"].items():
            reaches[key] = {"upstream_site": spec["upstream_site"],
                            "downstream_site": values["downstream_site"],
                            **reach_share(stem, extent, down, site(spec["upstream_site"]),
                                          float(spec.get("end_uncertainty_m", 0.0)),
                                          float(values["max_site_offset_m"]))}
        summary = {
            "definition": (
                "Reach = section of the OSM Nakkhu main stem (ways named "
                f"{', '.join(values['stem_names'])}, merged into one line) between the stem "
                "points nearest the upstream site and the confluence; share = length inside "
                f"the analysis extent / reach length, in {work}. main_stem: upstream end at "
                "Tikabhairab (head of the main stem). mapped_water: upstream end at the "
                "map-read upstream end of UNOSAT-mapped water (+-100 m). The upstream limit "
                "of the 2024 flooding is not established, so both are reported."),
            "work_crs": work,
            "stem_length_m": float(stem.length),
            "inputs": {key: {"path": values["inputs"][key], "sha256": sha256_file(p)}
                       for key, p in inputs.items()},
            "reaches": reaches,
        }
    run.write_summary(summary)
    if do_promote:
        promote(f"{run.ref}/summary.json", ARTIFACT, project.paths,
                replace=(project.paths.artifacts / ARTIFACT).exists())
    return run, summary
