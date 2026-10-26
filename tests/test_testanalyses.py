"""Plan 007 test analyses (C1, C5, C4) on synthetic per-block tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nakkhu.testanalyses import (
    AnalysisError,
    c1,
    c5,
    cross_sensor_stats,
    f1_point,
)

N_BLOCKS = 12


def table(f1: float, n_ref: int = 100, rng: np.random.Generator | None = None,
          noise: float = 0.0) -> pd.DataFrame:
    """Per-block counts with precision = recall = ``f1`` (+ optional noise per block)."""
    rng = rng or np.random.default_rng(0)
    p = np.clip(f1 + noise * rng.standard_normal(N_BLOCKS), 0.01, 0.99)
    n = np.full(N_BLOCKS, n_ref, dtype=float)
    tp = np.round(p * n)
    out = pd.DataFrame({"block_id": [f"b{i:02d}" for i in range(N_BLOCKS)], "n_pred": n,
                        "n_ref": n, "tp_pred_50": tp, "tp_ref_50": tp,
                        "tp_pred_30": np.minimum(n, tp + 10), "tp_ref_30": np.minimum(n, tp + 10)})
    return out


def test_f1_point():
    t = table(0.4)
    assert f1_point(t) == pytest.approx(0.4)


def test_c1_pass_and_fail():
    rng = np.random.default_rng(1)
    strong = [table(0.40 + 0.002 * s, rng=rng, noise=0.02) for s in range(3)]
    weak = [table(0.20 + 0.002 * s, rng=rng, noise=0.02) for s in range(3)]
    res = c1(strong, weak, n_boot=500, seed=0)
    assert res["pass"] and res["iou_50"]["ci_low"] > 0
    assert res["iou_50"]["diff"] == pytest.approx(0.2, abs=0.01)
    same = c1(strong, strong, n_boot=500, seed=0)
    assert not same["pass"]                          # difference 0: CI includes 0
    # a positive difference that does not exceed the larger seed SD fails
    wide = [table(0.40, rng=rng), table(0.30, rng=rng), table(0.50, rng=rng)]
    near = [table(0.38, rng=rng)] * 3
    res = c1(wide, near, n_boot=500, seed=0)
    assert res["iou_50"]["diff"] < res["iou_50"]["larger_seed_sd"]
    assert not res["pass"]


def test_c5_matching_rule():
    rng = np.random.default_rng(2)
    seeds = [table(0.30, rng=rng, noise=0.01) for _ in range(5)]
    better = c5(seeds, table(0.10, rng=rng, noise=0.01), n_boot=500, seed=0)
    assert better["seeds_matching"] == 0 and not better["matches"]
    equal = c5(seeds, table(0.30, rng=rng, noise=0.01), n_boot=500, seed=0)
    assert equal["matches"] and equal["seeds_matching"] >= 3
    worse = c5(seeds, table(0.60, rng=rng, noise=0.01), n_boot=500, seed=0)
    assert worse["matches"]                          # CI below 0 counts as a match


def test_cross_sensor_drop_and_pass():
    rng = np.random.default_rng(3)
    pairs = [(table(0.40, rng=rng, noise=0.02), table(0.30, rng=rng, noise=0.02))
             for _ in range(5)]
    res = cross_sensor_stats(pairs, n_boot=500, seed=0)
    assert res["drop_pp_mean"] == pytest.approx(10, abs=2)
    assert res["pass"]
    big = [(table(0.50, rng=rng), table(0.20, rng=rng)) for _ in range(5)]
    res = cross_sensor_stats(big, n_boot=500, seed=0)
    assert res["drop_pp_mean"] == pytest.approx(30, abs=1) and not res["pass"]
    lo, hi = res["per_seed"][0]["ci_low_pp"], res["per_seed"][0]["ci_high_pp"]
    assert lo <= res["per_seed"][0]["drop_pp"] <= hi


def test_unpaired_blocks_refused():
    a = table(0.3)
    b = table(0.3)
    b["block_id"] = b["block_id"][::-1].to_numpy()
    with pytest.raises(AnalysisError):
        c5([a], b, n_boot=10)
