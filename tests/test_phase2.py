"""Plan 014 Phase 2 label-free site preparation (followup.phase2) on synthetic geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, box

from nakkhu.config import load_project
from nakkhu.followup import phase2 as P2
from nakkhu.splits import block_ids

REAL_ROOT = Path(__file__).resolve().parents[1]
LAT = {"west": 331200.0, "north": 3074880.0, "tile_m": 320.0, "block_tiles": 3}


def test_block_ids_follow_the_plan_005_lattice():
    x = np.array([331200.0 + 5, 331200.0 + 960 * 7 + 1, 331200.0 + 960 * 40 + 700])
    y = np.array([3074880.0 - 5, 3074880.0 - 960 * 3 - 1, 3074880.0 - 960 * 12 - 10])
    cols = np.floor((x - LAT["west"]) / 320).astype(int)
    rows = np.floor((LAT["north"] - y) / 320).astype(int)
    assert list(P2.block_of(x, y, LAT)) == list(block_ids(cols, rows, 3))
    blocks = P2.lattice_blocks((331200.0 + 960 * 7 + 10, 3074880.0 - 960 * 4 + 10,
                                331200.0 + 960 * 8 + 10, 3074880.0 - 960 * 3 - 10), LAT)
    assert set(blocks["block_id"]) == {"b_07_03", "b_08_03"}
    assert all(abs(g.area - 960 ** 2) < 1e-6 for g in blocks.geometry)


def test_corridor_blocks_exclusion_and_shared_blocks():
    cfg = {"lattice": LAT, "corridor_m": 1000.0}
    y0 = 3074880.0 - 960 * 10 - 480                    # through block-row centres
    a = LineString([(331200.0 + 960 * 5, y0), (331200.0 + 960 * 12, y0)])
    b = LineString([(331200.0 + 960 * 11, y0 - 960), (331200.0 + 960 * 16, y0 - 960)])
    excl = box(331200.0 + 960 * 5, y0 - 3000, 331200.0 + 960 * 6.5, y0 + 3000)
    blocks, info = P2.corridor_blocks({"A": a, "B": b}, cfg, excl)
    ids_a, ids_b = set(blocks["A"]["block_id"]), set(blocks["B"]["block_id"])
    assert not ids_a & ids_b                            # every block has one site
    assert info["A"]["dropped_nakkhu_buffer"] > 0 and info["B"]["shared_lost"] > 0
    for s, line in (("A", a), ("B", b)):
        cent = shapely.centroid(blocks[s].geometry.values)
        assert (shapely.distance(cent, line) <= 1000.0).all()
        assert not shapely.intersects(blocks[s].geometry.values, excl).any()


def test_site_change_config_keeps_every_v9_rule_section():
    project = load_project(REAL_ROOT)
    values = P2.site_change_values(project, "A")
    v9 = P2.load_yaml(project.paths.resolve(P2.CHANGE))
    for key in P2.RULE_SECTIONS:
        assert values[key] == v9[key]
    for key in P2.BUILDING_RULE_KEYS:
        if key == "inventories":
            assert values["buildings"][key] == ["microsoft"]
        else:
            assert values["buildings"][key] == v9["buildings"][key]
    assert values["region"] == "phase2_site_A"
    assert values["scenes"]["post"][0]["strip"] == "0600"


def test_sites_config_way_ids_exist_in_the_pinned_waterways():
    project = load_project(REAL_ROOT)
    cfg = P2.load_sites(project)
    ways = P2.river_ways(project, cfg["rivers"])
    known = set(ways["way_id"])
    for s, sc in cfg["sites"].items():
        assert set(sc["way_ids"]) <= known, s
    assert set(cfg["bagmati_way_ids"]) <= known
    all_ids = [i for sc in cfg["sites"].values() for i in sc["way_ids"]]
    assert len(all_ids) == len(set(all_ids))


def test_site_change_map_refuses_before_a4_is_frozen():
    import pytest

    from nakkhu.evaluate.addendum import frozen_addenda

    project = load_project(REAL_ROOT)
    assert P2.A4 not in frozen_addenda(project)
    with pytest.raises(P2.Phase2Error, match="not frozen"):
        P2.change_site(project, "A")
