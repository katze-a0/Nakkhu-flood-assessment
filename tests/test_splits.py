"""Plan 005 Step 1: spatial-block split rule on a synthetic grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from nakkhu import splits

TILE = 320.0


def _index(ncols: int = 15, nrows: int = 15) -> pd.DataFrame:
    rows = []
    for r in range(nrows):
        for c in range(ncols):
            west, north = c * TILE, (nrows - r) * TILE
            rows.append({"tile_id": f"t_{c:04d}_{r:04d}", "col": c, "row": r,
                         "bounds": [west, north - TILE, west + TILE, north],
                         "valid_frac": 1.0 if c > 0 else 0.3})
    return pd.DataFrame(rows)


RULE = {"block_tiles": 3, "buffer_m": 320.0, "min_valid_frac": 0.5,
        "val": {"fraction_of_eligible_blocks": 0.25, "seed": 0, "density_quantiles": 2,
                "distance_quantiles": 2}}


def _build(index: pd.DataFrame):
    extent = shapely.box(100, 100, 3 * TILE - 50, 4 * TILE - 50)   # south-west corner
    rng = np.random.default_rng(1)
    ms = rng.integers(0, 200, len(index))
    rivers = np.array([shapely.LineString([(0, 2500), (4800, 2600)])], dtype=object)
    return splits.assign(index, extent, RULE, ms, rivers)


def test_split_separation_cover_and_reasons():
    index = _index()
    frame, blocks = _build(index)
    boxes = splits.tile_boxes(index.sort_values("tile_id").reset_index(drop=True))
    checks = splits.check_separation(frame, boxes, 320.0)
    assert checks["separation_ok"] and checks["unique_tile_ids"] and checks["every_tile_assigned"]
    assert set(frame["tile_id"]) == set(index["tile_id"])
    assert frame["split"].isin(splits.SPLITS).all()
    # every train tile is >= 320 m from every test and val tile
    split = frame["split"].to_numpy()
    for other in ("test", "val"):
        d = shapely.distance(boxes[split == "train"],
                             shapely.union_all(boxes[split == other]))
        assert d.min() >= 320.0
    # val consists of whole eligible blocks; buffer tiles carry a reason
    val_blocks = set(frame.loc[frame["split"] == "val", "block_id"])
    assert val_blocks and val_blocks <= set(blocks.loc[blocks["eligible"], "block_id"])
    assert (frame.loc[frame["split"] == "val"].groupby("block_id").size() == 9).all()
    buf = frame.loc[frame["split"] == "buffer", "buffer_of"]
    assert buf.isin(["test", "val", "test+val"]).all()
    assert not frame.loc[frame["col"] == 0, "usable"].any()
    expected = round(0.25 * int(blocks["eligible"].sum()))
    assert len(val_blocks) == expected


def test_split_rebuild_is_byte_identical():
    index = _index()
    header = {"schema_version": 1, "version": "split_v1", "rule_sha256": "r",
              "grid_sha256": "g", "tile_index": "x", "block_tiles": 3, "buffer_m": 320.0}
    frame, blocks = _build(index)
    a = splits.split_yaml(frame, blocks, header)
    frame, blocks = _build(index.sample(frac=1.0, random_state=3))
    b = splits.split_yaml(frame, blocks, header)
    assert a == b
    assert "val_blocks: [b_" in a


def test_allocate_largest_remainder():
    assert splits.allocate({"a": 5, "b": 3, "c": 2}, 3) == {"a": 1, "b": 1, "c": 1}
    # quotas 2.4, 1.2, 0.4: remainders tie (a, c) -> sorted key a
    assert splits.allocate({"a": 6, "b": 3, "c": 1}, 4) == {"a": 3, "b": 1, "c": 0}
    assert sum(splits.allocate({"a": 13, "b": 20, "c": 19}, 6).values()) == 6


def test_centroid_counts_first_tile_wins():
    index = _index(2, 1)
    geoms = np.array([shapely.box(10, 10, 20, 20), shapely.box(310, 10, 330, 20)], dtype=object)
    counts = splits.centroid_counts(index, geoms)
    assert counts.tolist() == [2, 0]   # centroid x = 320 lies on the shared edge
