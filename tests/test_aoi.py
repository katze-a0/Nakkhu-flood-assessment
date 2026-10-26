import json
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.affinity import translate
from shapely.geometry import LineString, Point, box

from nakkhu.aoi import PRODUCTS, compare
from nakkhu.aoi.build import AoiBuildError, closure
from nakkhu.aoi.extent import analysis_extent, footprint
from nakkhu.aoi.lines import LinesError, read_ways, ways_in_bbox
from nakkhu.aoi.sites import SitesError, site_frame
from nakkhu.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
TOL = {"area_rel": 1e-6, "iou_min": 0.999999, "hausdorff_m": 0.01, "length_rel": 1e-6}
WORK = "EPSG:32645"


def test_closure_adds_upstream_products_in_order():
    assert closure(["analysis_extent"], "rebuilt") == ["osm_lines", "catchment_c4",
                                                       "analysis_extent"]
    assert closure(["analysis_extent"], "registered") == ["analysis_extent"]
    assert closure(PRODUCTS, "rebuilt") == list(PRODUCTS)
    with pytest.raises(AoiBuildError):
        closure(["nope"], "rebuilt")


def test_compare_polygons_identical_and_shifted():
    a = box(85.30, 27.65, 85.31, 27.66)
    same = compare.polygons(a, a, WORK, TOL)
    assert same["match"] and same["coordinates_identical"] and same["iou"] == 1.0
    moved = compare.polygons(translate(a, xoff=1e-4), a, WORK, TOL)  # about 10 m east
    assert not moved["match"] and 5 < moved["hausdorff_m"] < 15 and moved["iou"] < 0.99


def test_compare_lines_and_points():
    line = LineString([(85.30, 27.65), (85.31, 27.66)])
    a = gpd.GeoDataFrame({"name": ["r"], "osm_way": [1]}, geometry=[line], crs="EPSG:4326")
    assert compare.lines(a, a, WORK, TOL)["match"]
    b = gpd.GeoDataFrame({"name": ["r", "s"], "osm_way": [1, 2]},
                         geometry=[line, translate(line, yoff=0.01)], crs="EPSG:4326")
    result = compare.lines(a, b, WORK, TOL)
    assert not result["match"] and result["ways"][1]["only_in"] == "registered"
    p = gpd.GeoDataFrame({"name": ["x"]}, geometry=[Point(85.3, 27.6)], crs="EPSG:4326")
    q = gpd.GeoDataFrame({"name": ["x"]}, geometry=[Point(85.3001, 27.6)], crs="EPSG:4326")
    assert compare.points(p, p, WORK, TOL)["match"]
    assert not compare.points(q, p, WORK, TOL)["match"]


def test_site_frame_requires_provenance():
    good = {"sites": [{"name": "a", "lon": 85.3, "lat": 27.6, "method": "map_read",
                       "provenance": "map-read from X"}]}
    frame = site_frame(good)
    assert frame.iloc[0]["coord_source"] == "map-read from X"
    with pytest.raises(SitesError):
        site_frame({"sites": [{"name": "a", "lon": 85.3, "lat": 27.6, "method": "map_read"}]})
    with pytest.raises(SitesError):
        site_frame({"sites": [{**good["sites"][0], "method": "guess"}]})


def test_repo_site_config_is_complete():
    values = yaml.safe_load((REPO / "configs/aoi/flood_reach_sites_2024.yaml").read_text(
        encoding="utf-8"))
    frame = site_frame(values)
    assert len(frame) == len(values["sites"]) and frame["name"].is_unique


def _write_osm(path: Path) -> None:
    import osmium
    import osmium.osm.mutable

    writer = osmium.SimpleWriter(str(path))
    try:
        coords = [(85.30, 27.65), (85.305, 27.655), (85.31, 27.66), (85.40, 27.70)]
        for i, (lon, lat) in enumerate(coords, start=1):
            writer.add_node(osmium.osm.mutable.Node(id=i, location=(lon, lat), version=1))
        writer.add_way(osmium.osm.mutable.Way(id=10, nodes=[1, 2, 3], version=1,
                                              tags={"waterway": "river", "name": "Nakhu River"}))
        writer.add_way(osmium.osm.mutable.Way(id=11, nodes=[3, 4], version=1,
                                              tags={"waterway": "river", "name": "Nakhu River"}))
    finally:
        writer.close()


def test_read_ways_by_id_and_reports_unlisted(tmp_path):
    osm = tmp_path / "t.osm"
    _write_osm(osm)
    frame, record = read_ways(osm, [{"id": 10, "name": "Nakhu River"}], ["Nakhu River"],
                              (85.0, 27.0, 86.0, 28.0))
    assert list(frame["osm_way"]) == [10] and len(frame.geometry.iloc[0].coords) == 3
    assert [w["id"] for w in record["unlisted_same_name"]] == [11]
    with pytest.raises(LinesError):
        read_ways(osm, [{"id": 99, "name": "x"}])
    with pytest.raises(LinesError):
        read_ways(osm, [{"id": 10, "name": "Renamed"}])


def test_ways_in_bbox_matches_overpass_layout(tmp_path):
    osm = tmp_path / "t.osm"
    _write_osm(osm)
    doc = ways_in_bbox(osm, {"waterway": "river"}, (27.64, 85.29, 27.665, 85.315), "T")
    assert [e["id"] for e in doc["elements"]] == [10, 11]  # way 11 starts on the box edge
    first = doc["elements"][0]
    assert first["geometry"][0] == {"lat": 27.65, "lon": 85.30}
    assert first["bounds"]["maxlon"] == 85.31 and doc["osm3s"]["timestamp_osm_base"] == "T"
    only = ways_in_bbox(osm, {"waterway": "river"}, (27.69, 85.39, 27.71, 85.41))
    assert [e["id"] for e in only["elements"]] == [11]
    same = compare.overpass(doc, doc)
    assert same["match"] and same["elements_identical"]
    fewer = compare.overpass(only, doc)
    assert not fewer["match"] and fewer["only_registered"] == [10]


def test_extent_footprint_and_disc(tmp_path):
    stac = tmp_path / "stac"
    stac.mkdir()
    tile = box(85.28, 27.64, 85.32, 27.68)
    for name, cat in (("a", "PRE"), ("b", "OTHER")):
        (stac / f"{name}.json").write_text(json.dumps({
            "properties": {"catalog_id": cat},
            "geometry": json.loads(gpd.GeoSeries([tile]).to_json())["features"][0]["geometry"]}))
    pre = footprint([stac], ["PRE"], WORK)
    cfg = {"work_crs": WORK, "confluence": [85.30, 27.66],
           "analysis_extent": {"disc_radius_m": 500.0, "min_part_m2": 1e4}}
    catchment = box(85.29, 27.62, 85.31, 27.65)  # does not reach the disc
    poly, record = analysis_extent(catchment, pre, pre, cfg)
    assert record["parts"] == 2
    assert poly.contains(Point(85.30, 27.66)) and poly.within(tile.buffer(1e-9))


def test_cli_registers_aoi_build():
    args = build_parser().parse_args(["aoi", "build", "--only", "flood_sites"])
    assert args.only == ["flood_sites"] and args.upstream == "rebuilt"


def test_reach_share_along_stem():
    from nakkhu.aoi.reach import ReachError, merged_stem, reach_share

    # two ways forming one 10 km stem running north (downstream) along x = 0
    lines = gpd.GeoDataFrame({"name": ["Upper", "Lower", "Other"]},
                             geometry=[LineString([(0, 0), (0, 4000)]),
                                       LineString([(0, 4000), (0, 10000)]),
                                       LineString([(50, 0), (50, 10000)])], crs=WORK)
    stem = merged_stem(lines, ["Upper", "Lower"])
    assert stem.length == pytest.approx(10000)
    extent = box(-100, 6000, 100, 12000)  # holds the last 4 km
    down, up = Point(0, 10000), Point(30, 2000)  # up lies 30 m off the stem
    result = reach_share(stem, extent, down, up, 100.0, 300.0)
    assert result["length_m"] == pytest.approx(8000)
    assert result["inside_m"] == pytest.approx(4000)
    assert result["share"] == pytest.approx(0.5)
    assert result["site_offset_from_stem_m"]["upstream_site"] == pytest.approx(30)
    assert result["share_range"] == pytest.approx([4000 / 8100, 4000 / 7900])
    with pytest.raises(ReachError, match="from the stem"):
        reach_share(stem, extent, down, Point(500, 2000), 0.0, 300.0)
    with pytest.raises(ReachError, match="pieces"):
        merged_stem(lines, ["Upper", "Other"])
    with pytest.raises(ReachError, match="no OSM ways"):
        merged_stem(lines, ["Upper", "Missing"])
