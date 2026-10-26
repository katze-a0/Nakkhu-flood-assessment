import geopandas as gpd
from shapely.geometry import box

from nakkhu.reference import greenhouse


def test_matching_tag_and_pairs():
    assert greenhouse.matching_tag({"building": "greenhouse"}) == "building=greenhouse"
    assert greenhouse.matching_tag({"landuse": "greenhouse_horticulture"}) == (
        "landuse=greenhouse_horticulture")
    assert greenhouse.matching_tag({"building": "yes", "landuse": "farmland"}) is None
    assert ("building", "glasshouse") in greenhouse.tag_pairs()


def test_greenhouse_overlap_iou_and_centroid():
    gh = gpd.GeoSeries([box(0, 0, 10, 10)], crs="EPSG:32645")
    fp = gpd.GeoSeries([
        box(0, 0, 10, 10),     # identical: IoU 1
        box(5, 0, 15, 10),     # IoU 50/150 = 0.333 >= 0.3
        box(8, 0, 18, 10),     # IoU 20/180 = 0.111; centroid (13, 5) outside -> kept
        box(-1, -1, 30, 30),   # IoU 100/961 = 0.104, but centroid (14.5, 14.5) outside -> kept
        box(2, 2, 4, 4),       # small, inside: IoU 0.04, centroid inside -> greenhouse
        box(50, 50, 60, 60),   # disjoint
    ], crs="EPSG:32645")
    flags = greenhouse.greenhouse_overlap(fp, gh, 0.3)
    assert flags["is_greenhouse"].tolist() == [True, True, False, False, True, False]
    assert flags.loc[1, "max_iou"] == __import__("pytest").approx(1 / 3)
    assert greenhouse.greenhouse_overlap(fp, gpd.GeoSeries([], crs="EPSG:32645"))[
        "is_greenhouse"].sum() == 0
