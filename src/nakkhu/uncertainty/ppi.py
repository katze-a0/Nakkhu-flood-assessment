"""Bias-corrected tier counts (U3; prereg v5 ``uncertainty``).

Estimands (per tier group g of the primary HAND surface):

- **T_g** (primary): real buildings among the recall-frame units in g. The frame is the
  plan 006 ``recall_points`` frame (de-duplicated Google v3 / Microsoft / OSM representative
  points in the extent, greenhouses removed), stratified by source membership; Y = 1 if the
  unit is a building under the recall answer coding (protocol v2 primary).
- **N_t** (secondary): real buildings in tier t, = D_t p_g / r_g (``ratio``), with D_t the
  ``detector_final`` clusters in t, p_g their audited precision and r_g their audited recall.

Estimators of T_g (design weights w = N_h / n_h over answered units, domain indicators for g):

- HT: sum_S w Y 1[g].
- PPI++: lambda F_U + sum_S w (Y - lambda f) 1[g], with f the detector prediction for a unit
  (1 if a ``detector_final`` OBB lies within 1 m of the unit's own point), F_U = sum of f over
  every frame unit in g, and lambda = Cov(HT_Y, HT_f) / Var(HT_f) (stratified design
  variances, clipped to [0, 1]; 0 if Var(HT_f) = 0). lambda = 0 is HT; lambda = 1 is classic
  PPI. lambda is estimated once and held fixed in the bootstrap.

Intervals: Rao-Wu rescaled stratified bootstrap (the plan 006 weights), percentile. The
split-sample check (falsifier C2.ppi) and its simulation are here too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

LEVEL = 0.95
Z = float(stats.norm.ppf(0.5 + LEVEL / 2))


# --- design weights and variances --------------------------------------------------------------


def design_weights(strata: np.ndarray, frame_n: np.ndarray) -> np.ndarray:
    """w = N_h / n_h with n_h the number of (answered) rows of stratum h."""
    strata = np.asarray(strata)
    frame_n = np.asarray(frame_n, dtype=float)
    _, inv, counts = np.unique(strata, return_inverse=True, return_counts=True)
    return frame_n / counts[inv]


def strat_cov(strata: np.ndarray, frame_n: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Design covariance of the HT totals of ``a`` and ``b`` under stratified SRS without
    replacement: sum_h N_h^2 (1 - n_h/N_h) s_ab,h / n_h."""
    strata = np.asarray(strata)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = 0.0
    for h in np.unique(strata):
        sel = strata == h
        n = int(sel.sum())
        big = float(np.asarray(frame_n, dtype=float)[sel][0])
        if n < 2:
            continue
        s_ab = float(np.cov(a[sel], b[sel], ddof=1)[0, 1])
        out += big * big * (1.0 - n / big) * s_ab / n
    return out


def replicate_weights(strata: np.ndarray, frame_n: np.ndarray, n_boot: int,
                      seed: int) -> np.ndarray:
    """(n_boot, n) Rao-Wu bootstrap weights (plan 006 implementation)."""
    from nakkhu.audit.estimators import rao_wu_weights

    w = design_weights(strata, frame_n)
    return rao_wu_weights(np.asarray(strata), w, np.asarray(frame_n, dtype=float), n_boot,
                          seed)


def _ci(reps: np.ndarray) -> tuple[float, float]:
    reps = reps[np.isfinite(reps)]
    if not len(reps):
        return float("nan"), float("nan")
    lo, hi = np.quantile(reps, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return float(lo), float(hi)


# --- PPI++ and HT totals ---------------------------------------------------------------------


def ppi_lambda(strata: np.ndarray, frame_n: np.ndarray, y: np.ndarray, f: np.ndarray) -> float:
    var_f = strat_cov(strata, frame_n, f, f)
    if var_f <= 0:
        return 0.0
    return float(np.clip(strat_cov(strata, frame_n, y, f) / var_f, 0.0, 1.0))


def ppi_total(strata: np.ndarray, frame_n: np.ndarray, y: np.ndarray, f: np.ndarray,
              f_frame_total: float, *, lam: float | None = None,
              rep_w: np.ndarray | None = None) -> dict[str, Any]:
    """PPI++ total (``lam=None`` estimates lambda; ``lam=0`` gives HT). ``y`` and ``f`` are
    already multiplied by the domain indicator; ``f_frame_total`` is sum f over the frame
    domain. With ``rep_w`` (Rao-Wu replicate weights) the percentile CI and bootstrap SD."""
    y, f = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    w = design_weights(strata, frame_n)
    lam = ppi_lambda(strata, frame_n, y, f) if lam is None else float(lam)
    z = y - lam * f
    est = lam * float(f_frame_total) + float(np.sum(w * z))
    var = strat_cov(strata, frame_n, z, z)
    out = {"estimate": est, "lambda": lam, "se_lin": float(np.sqrt(max(var, 0.0))),
           "n": int(len(y))}
    if rep_w is not None:
        reps = lam * float(f_frame_total) + rep_w @ z
        out["ci_lo"], out["ci_hi"] = _ci(reps)
        out["se_boot"] = float(np.std(reps, ddof=1))
    return out


# --- ratio estimator ------------------------------------------------------------------------


def weighted_ratio(w: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray | float:
    """sum(w num) / sum(w den); ``w`` may be (n,) or (B, n)."""
    d = w @ np.asarray(den, dtype=float)
    n = w @ np.asarray(num, dtype=float)
    return np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)


def ratio_counts(d_t: np.ndarray, tier_group: Sequence[int],
                 prec: dict[str, np.ndarray], rec: dict[str, np.ndarray],
                 prec_rep: np.ndarray, rec_rep: np.ndarray) -> pd.DataFrame:
    """Per tier t: naive D_t, precision-only D_t p_g, ratio D_t p_g / r_g, with percentile CIs
    from independent replicate weights of the two audits.

    ``prec``: ``stratum``, ``frame_n``, ``y`` (1 = real), ``group``, ``final`` (unit is a
    detector_final cluster). ``rec``: ``stratum``, ``frame_n``, ``y`` (1 = building),
    ``group``, ``d`` (detected by detector_final)."""
    wp = design_weights(prec["stratum"], prec["frame_n"])
    wr = design_weights(rec["stratum"], rec["frame_n"])
    rows = []
    for t, g in enumerate(tier_group):
        pg = (np.asarray(prec["group"]) == g) & np.asarray(prec["final"], dtype=bool)
        rg = np.asarray(rec["group"]) == g
        py = np.asarray(prec["y"], dtype=float)
        ry = np.asarray(rec["y"], dtype=float)
        rd = np.asarray(rec["d"], dtype=float)
        p = float(weighted_ratio(wp, py * pg, pg.astype(float)))
        r = float(weighted_ratio(wr, ry * rd * rg, ry * rg))
        p_b = weighted_ratio(prec_rep, py * pg, pg.astype(float))
        r_b = weighted_ratio(rec_rep, ry * rd * rg, ry * rg)
        d = float(d_t[t])
        prec_only = d * p
        ratio = d * p / r if r and np.isfinite(r) and r > 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_b = d * p_b / r_b
        rows.append({"tier": t, "group": g, "naive": d, "precision": p, "recall": r,
                     "precision_only": prec_only,
                     "precision_only_ci": _ci(d * np.asarray(p_b, dtype=float)),
                     "ratio": ratio, "ratio_ci": _ci(np.asarray(ratio_b, dtype=float))})
    return pd.DataFrame(rows)


# --- tier pooling -----------------------------------------------------------------------------


def pool_tiers(answered: Sequence[int], min_answered: int) -> list[int]:
    """Group id per tier (tiers ordered from low to high HAND). Repeatedly merge the group with
    the fewest answered units below ``min_answered`` into its adjacent group with fewer
    answered units (tie: the lower-HAND neighbour), until every group has enough or one group
    remains. Uses counts of answered units only, never their answers."""
    groups = [[t] for t in range(len(answered))]
    n = [int(a) for a in answered]
    while len(groups) > 1:
        small = [i for i, v in enumerate(n) if v < min_answered]
        if not small:
            break
        i = min(small, key=lambda k: (n[k], k))
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if n[i - 1] <= n[i + 1] else i + 1
        lo, hi = sorted((i, j))
        groups[lo:hi + 1] = [groups[lo] + groups[hi]]
        n[lo:hi + 1] = [n[lo] + n[hi]]
    out = [0] * len(answered)
    for g, members in enumerate(groups):
        for t in members:
            out[t] = g
    return out


# --- split-sample coverage check (falsifier C2.ppi) --------------------------------------------


SPLIT_MODES = ("stratified_random", "block")


def halve(sample: pd.DataFrame, mode: str, gen: np.random.Generator) -> np.ndarray:
    """Boolean mask of half A.

    - ``stratified_random`` (prereg v6 primary): within each audit stratum a seeded random
      permutation of its units; the first floor(n_h / 2) go to A. Matches the stratified random
      design the audit was drawn under, so each half is itself a stratified random sample.
    - ``block`` (v5 rule, descriptive since v6): a seeded random permutation of the plan 005
      blocks, the first floor(B / 2) blocks go to A (a cluster subsample).
    """
    if mode == "stratified_random":
        in_a = np.zeros(len(sample), dtype=bool)
        strata = sample["stratum"].to_numpy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            in_a[gen.permutation(idx)[: len(idx) // 2]] = True
        return in_a
    if mode == "block":
        blocks = np.unique(sample["block"].to_numpy())
        perm = gen.permutation(blocks)
        return np.isin(sample["block"].to_numpy(), perm[: len(blocks) // 2])
    raise ValueError(f"unknown split mode {mode!r}")


def split_sample(sample: pd.DataFrame, f_frame_total: dict[int, float], *, repeats: int,
                 seed: int, n_boot: int, min_per_stratum: int = 2,
                 mode: str = "stratified_random") -> dict[str, Any]:
    """``sample``: answered recall units with ``stratum``, ``frame_n``, ``y``, ``f``,
    ``group`` (and ``block`` for ``mode="block"``). Halves A / B by :func:`halve`; weights
    recomputed within each half. A gives PPI++, B gives HT. Miss if |A - B| > z sqrt(se_A^2 +
    se_B^2) (bootstrap SDs); the share of repeats whose A percentile CI misses B's point
    estimate is reported too."""
    from nakkhu.uncertainty.spec import rng, stream_seed

    gen = rng(seed, "split_sample" if mode == "stratified_random" else "split_sample_block")
    boot_seed = stream_seed(seed, "split_sample_bootstrap")
    strata_all = np.unique(sample["stratum"].to_numpy())
    groups = sorted(f_frame_total)
    misses = {g: 0 for g in groups}
    naive = {g: 0 for g in groups}
    valid = 0
    invalid = 0
    for r in range(repeats):
        in_a = halve(sample, mode, gen)
        halves = (sample.loc[in_a], sample.loc[~in_a])
        if any(int((h["stratum"] == s).sum()) < min_per_stratum
               for h in halves for s in strata_all):
            invalid += 1
            continue
        valid += 1
        reps = [replicate_weights(h["stratum"].to_numpy(), h["frame_n"].to_numpy(), n_boot,
                                  boot_seed + 2 * r + k) for k, h in enumerate(halves)]
        for g in groups:
            res = []
            for k, h in enumerate(halves):
                dom = (h["group"].to_numpy() == g).astype(float)
                res.append(ppi_total(h["stratum"].to_numpy(), h["frame_n"].to_numpy(),
                                     h["y"].to_numpy(dtype=float) * dom,
                                     h["f"].to_numpy(dtype=float) * dom, f_frame_total[g],
                                     lam=None if k == 0 else 0.0, rep_w=reps[k]))
            a, b = res
            se = np.sqrt(a["se_boot"] ** 2 + b["se_boot"] ** 2)
            misses[g] += int(abs(a["estimate"] - b["estimate"]) > Z * se)
            naive[g] += int(not (a["ci_lo"] <= b["estimate"] <= a["ci_hi"]))
    pairs = valid * len(groups)
    return {"mode": mode, "repeats": repeats, "valid": valid, "invalid": invalid,
            "miss_rate": (sum(misses.values()) / pairs) if pairs else float("nan"),
            "miss_rate_by_group": {g: (misses[g] / valid if valid else float("nan"))
                                   for g in groups},
            "a_ci_misses_b_rate": (sum(naive.values()) / pairs) if pairs else float("nan"),
            "rule": "miss if |A - B| > z_0.975 sqrt(se_A^2 + se_B^2)"}


# --- simulation (Step 5 verification; expected split-sample miss rate) ---------------------------


def synthetic_population(n_h: dict[str, int], p_y: dict[str, float], p_f_given: tuple[float, float],
                         n_groups: int, n_blocks: int, seed: int) -> pd.DataFrame:
    """A frame with strata, building indicator Y, detector prediction f (P(f=1|Y=0),
    P(f=1|Y=1) = ``p_f_given``), tier group and block, all random with known truth."""
    gen = np.random.default_rng([seed, 0])
    rows = []
    for h, big in n_h.items():
        y = gen.random(big) < p_y[h]
        f = np.where(y, gen.random(big) < p_f_given[1], gen.random(big) < p_f_given[0])
        rows.append(pd.DataFrame({"stratum": h, "frame_n": big, "y": y.astype(float),
                                  "f": f.astype(float),
                                  "group": gen.integers(0, n_groups, big),
                                  "block": gen.integers(0, n_blocks, big)}))
    return pd.concat(rows, ignore_index=True)


def draw_sample(pop: pd.DataFrame, n_h: dict[str, int], gen: np.random.Generator) -> pd.DataFrame:
    parts = []
    for h, n in n_h.items():
        idx = np.flatnonzero(pop["stratum"].to_numpy() == h)
        parts.append(pop.iloc[gen.choice(idx, size=n, replace=False)])
    return pd.concat(parts, ignore_index=True)


def simulate_coverage(pop: pd.DataFrame, n_h: dict[str, int], *, replicates: int, seed: int,
                      n_boot: int) -> dict[str, Any]:
    """Coverage of the PPI++ percentile CI for the true per-group totals, and bias, over
    ``replicates`` stratified samples of the synthetic population."""
    gen = np.random.default_rng([seed, 1])
    groups = sorted(np.unique(pop["group"]).tolist())
    truth = {g: float(pop.loc[pop["group"] == g, "y"].sum()) for g in groups}
    f_tot = {g: float(pop.loc[pop["group"] == g, "f"].sum()) for g in groups}
    est = {g: [] for g in groups}
    cover = {g: 0 for g in groups}
    for r in range(replicates):
        s = draw_sample(pop, n_h, gen)
        rep = replicate_weights(s["stratum"].to_numpy(), s["frame_n"].to_numpy(), n_boot,
                                seed * 100_003 + r)
        for g in groups:
            dom = (s["group"].to_numpy() == g).astype(float)
            res = ppi_total(s["stratum"].to_numpy(), s["frame_n"].to_numpy(),
                            s["y"].to_numpy() * dom, s["f"].to_numpy() * dom, f_tot[g],
                            rep_w=rep)
            est[g].append(res["estimate"])
            cover[g] += int(res["ci_lo"] <= truth[g] <= res["ci_hi"])
    out: dict[str, Any] = {"replicates": replicates, "groups": {}}
    for g in groups:
        e = np.asarray(est[g])
        out["groups"][g] = {"truth": truth[g], "mean_estimate": float(e.mean()),
                            "bias": float(e.mean() - truth[g]),
                            "mc_se": float(e.std(ddof=1) / np.sqrt(len(e))),
                            "coverage": cover[g] / replicates}
    out["coverage_pooled"] = sum(cover.values()) / (replicates * len(groups))
    return out


def simulate_split_miss(pop: pd.DataFrame, n_h: dict[str, int], *, samples: int, seed: int,
                        n_boot: int, mode: str = "stratified_random") -> dict[str, Any]:
    """Expected split-sample rates when the intervals are correct: one split (``mode``) of
    each of ``samples`` fresh stratified samples of the synthetic population (reported next to
    the observed rates; not gating)."""
    gen = np.random.default_rng([seed, 2])
    groups = sorted(np.unique(pop["group"]).tolist())
    f_tot = {g: float(pop.loc[pop["group"] == g, "f"].sum()) for g in groups}
    miss, naive, valid = 0.0, 0.0, 0
    for r in range(samples):
        s = draw_sample(pop, n_h, gen)
        res = split_sample(s, f_tot, repeats=1, seed=seed * 1_000_003 + r, n_boot=n_boot,
                           mode=mode)
        if res["valid"]:
            valid += 1
            miss += res["miss_rate"]
            naive += res["a_ci_misses_b_rate"]
    return {"mode": mode, "samples": samples, "valid": valid,
            "expected_miss_rate": miss / valid if valid else float("nan"),
            "expected_a_ci_misses_b_rate": naive / valid if valid else float("nan")}
