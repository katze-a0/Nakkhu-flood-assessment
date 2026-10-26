"""detector_final clustering rule (nakkhu.ensemble)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from nakkhu.ensemble import EnsembleError, fuse, greedy_clusters


def box(x: float, y: float = 0.0, w: float = 10.0, h: float = 6.0):
    return shapely.box(x, y, x + w, y + h)


def test_one_box_per_seed_and_no_chaining():
    # seed 0 has two adjacent houses; seed 1 one merged box overlapping both (IoU 0.5 each
    # would chain them under single linkage); seed 2 matches the first house.
    g = [box(0), box(10), box(0, w=20), box(0.2)]
    conf = [0.9, 0.8, 0.7, 0.6]
    member = [0, 0, 1, 2]
    lab = greedy_clusters(g, conf, member, 0.5)
    assert lab[0] != lab[1]                       # the two seed-0 houses stay apart
    assert lab[3] == lab[0]                       # seed 2 joins house 1
    for c in np.unique(lab):                      # never two boxes of one seed in a cluster
        m = np.asarray(member)[lab == c]
        assert len(m) == len(set(m.tolist()))


def test_joins_best_anchor_and_respects_iou_min():
    g = [box(0), box(30), box(1), box(29.5), box(60)]
    conf = [0.9, 0.85, 0.5, 0.4, 0.3]
    member = [0, 0, 1, 1, 1]
    lab = greedy_clusters(g, conf, member, 0.5)
    assert lab[2] == lab[0] and lab[3] == lab[1]
    assert lab[4] not in (lab[0], lab[1])        # no overlap: own cluster


def test_fuse_scores_support_and_medoid():
    g = [box(0), box(0.5), box(1.0), box(100)]
    conf = [0.9, 0.6, 0.3, 0.8]
    member = [0, 1, 2, 0]
    out = fuse(g, conf, member, n_members=5)
    assert len(out) == 2
    big = out.loc[out["n_seeds"] == 3].iloc[0]
    assert big["conf"] == pytest.approx((0.9 + 0.6 + 0.3) / 5)
    assert big["conf_s0"] == 0.9 and big["conf_s1"] == 0.6 and np.isnan(big["conf_s3"])
    assert big["medoid"] == 1                     # the middle box overlaps both others most
    assert shapely.equals(big["geometry"], g[1])
    lone = out.loc[out["n_seeds"] == 1].iloc[0]
    assert lone["conf"] == pytest.approx(0.8 / 5)


def test_deterministic_and_empty():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, size=(300, 2))
    g = [box(x, y) for x, y in xy]
    conf = rng.uniform(0.2, 1.0, 300)
    member = rng.integers(0, 5, 300)
    a = fuse(g, conf, member, 5)
    b = fuse(g, conf, member, 5)
    assert a.drop(columns="geometry").equals(b.drop(columns="geometry"))
    assert (a["n_seeds"] >= 1).all() and a["n_seeds"].sum() == 300
    assert len(fuse([], [], [], 5)) == 0
    with pytest.raises(EnsembleError):
        fuse([box(0)], [0.5], [5], 5)


def test_zeroshot_variant_selection():
    from nakkhu.zeroshot import select_variant

    zs = {"a": {"variant": "full", "val_f1_50": 0.10}, "b": {"variant": "quad",
                                                             "val_f1_50": 0.12},
          "selected": {"model": "a"}}
    assert select_variant(zs) == "b"
    zs["b"]["val_f1_50"] = 0.10
    assert select_variant(zs) == "a"            # tie -> full
    assert select_variant({}) is None
