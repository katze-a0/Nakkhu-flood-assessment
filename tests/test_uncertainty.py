"""Plan 010 (U1-U3) on synthetic data only: no audit answer, detection or test tile is read."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nakkhu.uncertainty import calibrate as C
from nakkhu.uncertainty import decompose as D
from nakkhu.uncertainty import ensemble as E
from nakkhu.uncertainty import ppi as P
from nakkhu.uncertainty.falsifiers import c2_outcome

# --- U1: ensemble ------------------------------------------------------------------------------


def test_unit_domains_from_audit_members():
    members = pd.DataFrame({
        "cluster": [0, 0, 0, 1, 2, 2],
        "run": ["s0", "s1", "ens", "s0", "s1", "ens"],
        "det": [0, 0, 0, 1, 1, 1],
        "conf": [0.9, 0.6, 0.3, 0.5, 0.7, 0.14]})
    out = E.unit_domains(members, "ens", ["s0", "s1"]).set_index("unit")
    assert out["in_final"].tolist() == [True, False, True]
    assert out.loc[0, "raw_score"] == pytest.approx(0.3)
    assert np.isnan(out.loc[1, "raw_score"])
    assert out["in_seed0"].tolist() == [True, True, False]
    assert out["in_seed1"].tolist() == [True, False, True]


def test_expected_count_sums_probabilities_per_tier():
    out = E.expected_count(np.array([0.2, 0.5, 1.0, np.nan]), np.array([0, 0, 2, 1]), 3)
    assert out.tolist() == pytest.approx([0.7, 0.0, 1.0])


# --- U1: calibration ---------------------------------------------------------------------------


def test_isotonic_is_monotone_and_weighted():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 1, 0, 1, 1])
    knots = C.fit_isotonic(s, y, np.array([1, 1, 3, 1, 1]))
    p = C.predict_isotonic(knots, np.linspace(0, 1, 50))
    assert np.all(np.diff(p) >= -1e-12)
    # 0.2 (y 1, w 1) and 0.3 (y 0, w 3) violate order and pool to 1 / 4
    assert knots[1].tolist() == pytest.approx([0.0, 0.25, 0.25, 1.0, 1.0])


def test_temperature_recovers_known_scale():
    gen = np.random.default_rng(0)
    z = gen.normal(0, 2, 20000)
    s = 1 / (1 + np.exp(-z))
    y = (gen.random(len(z)) < 1 / (1 + np.exp(-z / 2.0))).astype(float)
    assert C.fit_temperature(s, y) == pytest.approx(2.0, rel=0.05)


def test_cross_fit_never_sees_its_own_block():
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    y = np.array([0, 1, 0, 1, 0, 1])
    blocks = np.array(["a", "a", "b", "b", "c", "c"])
    seen = []

    def spy(method, st, yt, wt):
        seen.append(set(np.round(st, 3)))
        return lambda x: np.full(len(x), 0.5)

    C.cross_fit(s, y, np.ones(6), blocks, "isotonic", fit=spy)
    assert seen == [{0.2, 0.8, 0.3, 0.7}, {0.1, 0.9, 0.3, 0.7}, {0.1, 0.9, 0.2, 0.8}]


def test_ece_weighted_and_perfect_calibration():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    gen = np.random.default_rng(1)
    y = (gen.random(len(p)) < p).astype(float)
    assert C.ece(p, y, bins=10) < 0.03
    # a badly calibrated constant
    assert C.ece(np.full(100, 0.9), np.zeros(100)) == pytest.approx(0.9)
    w = np.r_[np.full(50, 3.0), np.ones(50)]
    yy = np.r_[np.ones(50), np.zeros(50)]
    assert C.ece(np.full(100, 0.5), yy, w, bins=1) == pytest.approx(abs(0.75 - 0.5))


def test_calibrate_u1_chooses_lower_ece_and_reports_both():
    gen = np.random.default_rng(2)
    s = gen.random(400)
    y = (gen.random(400) < s**2).astype(float)
    blocks = gen.integers(0, 8, 400)
    out = C.calibrate_u1(s, y, np.ones(400), blocks)
    eces = {m: d["ece"] for m, d in out["methods"].items()}
    assert set(eces) == {"temperature", "isotonic"}
    assert out["chosen"] == min(eces, key=lambda m: (eces[m], C.METHODS.index(m)))
    assert np.all(np.diff(out["predict"](np.linspace(0, 1, 20))) >= -1e-12)


# --- U3: PPI++, ratio, pooling ---------------------------------------------------------------


def test_ppi_lambda_zero_is_ht_and_one_is_difference_estimator():
    strata = np.array(["a"] * 4 + ["b"] * 4)
    frame_n = np.array([40] * 4 + [80] * 4)
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    f = np.array([1, 0, 1, 0, 0, 1, 1, 0], dtype=float)
    w = P.design_weights(strata, frame_n)
    ht = P.ppi_total(strata, frame_n, y, f, 30.0, lam=0.0)
    assert ht["estimate"] == pytest.approx(np.sum(w * y))
    diff = P.ppi_total(strata, frame_n, y, f, 30.0, lam=1.0)
    assert diff["estimate"] == pytest.approx(30.0 + np.sum(w * (y - f)))
    tuned = P.ppi_total(strata, frame_n, y, f, 30.0)
    assert 0.0 <= tuned["lambda"] <= 1.0


def test_ppi_lambda_minimises_linearised_variance():
    gen = np.random.default_rng(3)
    strata = np.repeat(["a", "b", "c"], 60)
    frame_n = np.repeat([3000, 2000, 1000], 60)
    y = (gen.random(180) < 0.5).astype(float)
    f = np.where(gen.random(180) < 0.8, y, 1 - y)
    best = P.ppi_total(strata, frame_n, y, f, 1000.0)
    for lam in np.linspace(0, 1, 11):
        assert best["se_lin"] <= P.ppi_total(strata, frame_n, y, f, 1000.0,
                                             lam=lam)["se_lin"] + 1e-9


def test_strat_cov_take_all_stratum_adds_no_variance():
    strata = np.array(["a"] * 5)
    v = P.strat_cov(strata, np.full(5, 5), np.arange(5.0), np.arange(5.0))
    assert v == 0.0


def test_pool_tiers_merges_small_groups_towards_fewer_answered():
    assert P.pool_tiers([30, 30, 30], 20) == [0, 1, 2]
    # tier 0 (5) merges with tier 1; tier 5 (3) merges with tier 4 (only neighbour)
    assert P.pool_tiers([5, 40, 25, 30, 22, 3], 20) == [0, 0, 1, 2, 3, 3]
    # middle small tier joins the neighbour with fewer answered (tie -> lower HAND)
    assert P.pool_tiers([50, 4, 30], 20) == [0, 1, 1]
    assert P.pool_tiers([25, 4, 25], 20) == [0, 0, 1]
    assert P.pool_tiers([1, 1, 1], 20) == [0, 0, 0]


def test_ratio_counts_on_known_precision_and_recall():
    prec = {"stratum": np.array(["s"] * 10), "frame_n": np.full(10, 100),
            "y": np.r_[np.ones(8), np.zeros(2)], "group": np.zeros(10, int),
            "final": np.ones(10, bool)}
    rec = {"stratum": np.array(["r"] * 10), "frame_n": np.full(10, 1000),
           "y": np.ones(10), "group": np.zeros(10, int), "d": np.r_[np.ones(5), np.zeros(5)]}
    pw = P.replicate_weights(prec["stratum"], prec["frame_n"], 50, 0)
    rw = P.replicate_weights(rec["stratum"], rec["frame_n"], 50, 1)
    out = P.ratio_counts(np.array([100.0]), [0], prec, rec, pw, rw)
    row = out.iloc[0]
    assert row["precision"] == pytest.approx(0.8) and row["recall"] == pytest.approx(0.5)
    assert row["ratio"] == pytest.approx(160.0) and row["precision_only"] == pytest.approx(80.0)


POP_N = {"GMO": 3000, "GM-": 2900, "G--": 1800, "-M-": 400, "--O": 300}
POP_P = {"GMO": 0.9, "GM-": 0.75, "G--": 0.5, "-M-": 0.45, "--O": 0.6}
SAMPLE_N = {"GMO": 64, "GM-": 63, "G--": 52, "-M-": 30, "--O": 27}


def test_ppi_simulation_unbiased_with_nominal_coverage():
    """Plan 010 Step 5: bias within Monte Carlo error, 95 % CI coverage in [93 %, 97 %] over
    1,000 replicates (synthetic population, known truth)."""
    pop = P.synthetic_population(POP_N, POP_P, (0.15, 0.7), n_groups=2, n_blocks=30, seed=0)
    sim = P.simulate_coverage(pop, SAMPLE_N, replicates=1000, seed=0, n_boot=300)
    for g in sim["groups"].values():
        assert abs(g["bias"]) <= 3 * g["mc_se"]
    assert 0.93 <= sim["coverage_pooled"] <= 0.97


def test_split_sample_counts_and_expected_rates():
    pop = P.synthetic_population(POP_N, POP_P, (0.15, 0.7), n_groups=2, n_blocks=30, seed=1)
    gen = np.random.default_rng(5)
    s = P.draw_sample(pop, SAMPLE_N, gen)
    f_tot = {g: float(pop.loc[pop["group"] == g, "f"].sum()) for g in (0, 1)}
    res = P.split_sample(s, f_tot, repeats=20, seed=0, n_boot=200)
    assert res["valid"] + res["invalid"] == 20 and res["valid"] > 0
    assert 0.0 <= res["miss_rate"] <= 1.0
    exp = P.simulate_split_miss(pop, SAMPLE_N, samples=150, seed=0, n_boot=200)
    # correct intervals: the combined rule misses about 5 %; the naive "A's CI covers B"
    # rule misses far more (B's own error), which is why it is not the falsifier
    assert exp["expected_miss_rate"] < 0.10
    assert exp["expected_a_ci_misses_b_rate"] > exp["expected_miss_rate"]


# --- U2: decomposition ------------------------------------------------------------------------


def test_additive_factorial_shares_recovered_exactly():
    a = np.array([-2.0, 0.0, 2.0])            # centred effects
    b = np.array([-1.0, 1.0])
    c = np.array([-3.0, 0.0, 0.0, 3.0])
    y = 10 + a[:, None, None] + b[None, :, None] + c[None, None, :]
    terms = D.anova_terms(y, ["seed", "surface", "audit"])
    ss_a, ss_b, ss_c = 2 * 4 * np.sum(a**2), 3 * 4 * np.sum(b**2), 3 * 2 * np.sum(c**2)
    tot = ss_a + ss_b + ss_c
    got = terms.set_index("term")["share"]
    assert got["seed"] == pytest.approx(ss_a / tot)
    assert got["surface"] == pytest.approx(ss_b / tot)
    assert got["audit"] == pytest.approx(ss_c / tot)
    assert terms.loc[terms["order"] > 1, "ss"].abs().max() < 1e-9
    assert terms["share"].sum() == pytest.approx(1.0)


def test_interaction_and_total_shares_sum():
    gen = np.random.default_rng(4)
    y = gen.normal(size=(5, 7, 3))
    terms = D.anova_terms(y, ["seed", "surface", "audit"])
    assert terms["share"].sum() == pytest.approx(1.0)
    fs = D.factor_shares(terms, ["seed", "surface", "audit"]).set_index("factor")
    assert (fs["total"] >= fs["first_order"] - 1e-12).all()
    dom = D.terrain_dominates(terms, "surface", ["seed", "audit"])
    det = terms.loc[terms["term"].isin(["seed", "audit", "seed:audit"]), "share"].sum()
    assert dom["detector_terms"] == pytest.approx(det)


def test_share_ci_resamples_within_strata_and_brackets():
    gen = np.random.default_rng(6)
    y = gen.normal(size=(5, 28)) + np.linspace(0, 3, 28)[None, :]
    out = D.share_ci(y, ["seed", "surface"], axis=1, strata=["glo30"] * 14 + ["fabdem"] * 14,
                     n_boot=100, seed=0)
    row = out.set_index("term").loc["surface"]
    assert row["ci_lo"] <= row["share"] <= row["ci_hi"]


def test_unbalanced_factorial_refused_and_table_checks():
    with pytest.raises(ValueError, match="unbalanced"):
        D.anova_terms(np.array([[1.0, np.nan], [2.0, 3.0]]), ["a", "b"])
    long = pd.DataFrame({"inventory": "x", "member": [0, 0, 1, 1], "tier": [0, 1, 0, 1],
                         "count": [3, 7, 4, 6]})
    assert D.check_factorial(long) == []
    bad = long.assign(count=[3, 7, 4, 7])
    assert "totals differ" in D.check_factorial(bad)[0]
    arr = D.to_array(long, ["member", "tier"], "count")
    assert arr.tolist() == [[3, 7], [4, 6]]


def test_corrected_counts_ratio_per_tier_group():
    d = np.ones((2, 3, 4)) * 10.0                # seeds x surfaces x tiers
    p = np.full((2, 3, 2, 5), 0.8)               # ... x groups x replicates
    r = np.full((2, 3, 2, 5), 0.5)
    y = D.corrected_counts(d, p, r, tier_group=[0, 0, 1, 1], target_tiers=[0, 1])
    assert y.shape == (2, 3, 5) and np.allclose(y, 2 * 10 * 0.8 / 0.5)


# --- falsifiers -------------------------------------------------------------------------------


def test_c2_outcome_thresholds():
    u1 = {"chosen": "isotonic", "methods": {"isotonic": {"ece": 0.08},
                                             "temperature": {"ece": 0.12}}}
    split = {"miss_rate": 0.11, "valid": 990, "rule": "r"}
    out = c2_outcome(u1, split, {"terrain_dominates": True})
    assert out["ece"]["pass"] and not out["ppi"]["pass"]
    assert out["decomposition"]["gating"] is False


def test_halve_stratified_keeps_half_of_each_stratum_and_block_keeps_blocks_whole():
    s = pd.DataFrame({"stratum": ["a"] * 7 + ["b"] * 4, "block": list("xxyyzzwxyzw")})
    gen = np.random.default_rng(0)
    a = P.halve(s, "stratified_random", gen)
    assert int(a[:7].sum()) == 3 and int(a[7:].sum()) == 2
    b = P.halve(s, "block", gen)
    for blk in "xyzw":
        sel = (s["block"] == blk).to_numpy()
        assert b[sel].all() or not b[sel].any()
    with pytest.raises(ValueError):
        P.halve(s, "other", gen)


def test_stratified_split_is_calibrated_where_block_split_is_not():
    """Tier groups clustered in space (group = block band): with exact intervals the block
    split misses far above 10 %, the stratified random split near 5 % (prereg v6 reason)."""
    gen = np.random.default_rng(7)
    rows = []
    for h, big, p in (("A", 4000, 0.8), ("B", 3000, 0.5)):
        block = gen.integers(0, 16, big)
        group = block // 8                     # spatial: groups follow blocks
        q = np.where(group == 0, p, p - 0.3)
        y = (gen.random(big) < q).astype(float)
        f = np.where(y > 0, gen.random(big) < 0.7, gen.random(big) < 0.15).astype(float)
        rows.append(pd.DataFrame({"stratum": h, "frame_n": big, "y": y, "f": f,
                                  "group": group, "block": block}))
    pop = pd.concat(rows, ignore_index=True)
    n_h = {"A": 80, "B": 70}
    strat = P.simulate_split_miss(pop, n_h, samples=120, seed=0, n_boot=150)
    block = P.simulate_split_miss(pop, n_h, samples=120, seed=0, n_boot=150, mode="block")
    assert strat["expected_miss_rate"] < 0.10
    assert block["expected_miss_rate"] > strat["expected_miss_rate"] + 0.05
