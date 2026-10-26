"""Variance decomposition of tier counts over a balanced full factorial (U2; prereg v5).

For an array Y with one axis per factor (complete, balanced, one value per cell) every
subset S of factors has the effect tau_S = sum_{T subset S} (-1)^{|S|-|T|} M_T, where M_T is
the mean of Y over the factors not in T. The sums of squares SS_S = sum over cells of tau_S^2
add up exactly to the total SS. Shares: first-order = SS_{i} / SS_total; total = sum of SS_S
over S containing i. Share CIs resample one factor's levels within strata (HAND surfaces within
DEMs), which keeps the design balanced.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def anova_terms(y: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Rows: term (``a`` or ``a:b``...), order, ss, share."""
    y = np.asarray(y, dtype=float)
    if y.ndim != len(names):
        raise ValueError(f"{y.ndim} axes, {len(names)} factor names")
    if not np.all(np.isfinite(y)):
        raise ValueError("factorial has missing or non-finite cells (unbalanced)")
    axes = range(y.ndim)
    means = {}
    for k in range(y.ndim + 1):
        for keep in itertools.combinations(axes, k):
            drop = tuple(a for a in axes if a not in keep)
            means[keep] = y.mean(axis=drop, keepdims=True) if drop else y
    total = float(np.sum((y - y.mean()) ** 2))
    rows = []
    for k in range(1, y.ndim + 1):
        for s in itertools.combinations(axes, k):
            tau = np.zeros([1] * y.ndim)
            for j in range(k + 1):
                for t in itertools.combinations(s, j):
                    tau = tau + (-1) ** (k - j) * means[t]
            ss = float(np.sum(np.broadcast_to(tau, y.shape) ** 2))
            rows.append({"term": ":".join(names[a] for a in s), "order": k, "ss": ss,
                         "share": ss / total if total > 0 else float("nan")})
    return pd.DataFrame(rows)


def factor_shares(terms: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """First-order and total share per factor."""
    rows = []
    for n in names:
        first = terms.loc[terms["term"] == n, "share"].sum()
        tot = terms.loc[terms["term"].str.split(":").apply(lambda p, n=n: n in p),
                        "share"].sum()
        rows.append({"factor": n, "first_order": float(first), "total": float(tot)})
    return pd.DataFrame(rows)


def share_ci(y: np.ndarray, names: Sequence[str], axis: int, strata: Sequence[Any], *,
             n_boot: int, seed: int, level: float = 0.95) -> pd.DataFrame:
    """Percentile CIs of every term share, resampling levels of ``axis`` with replacement
    within ``strata`` (one label per level)."""
    y = np.asarray(y, dtype=float)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    gen = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = np.concatenate([gen.choice(g, size=len(g), replace=True) for g in groups])
        reps.append(anova_terms(np.take(y, pick, axis=axis), names)["share"].to_numpy())
    r = np.asarray(reps)
    base = anova_terms(y, names)
    lo, hi = np.quantile(r, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return base.assign(ci_lo=lo, ci_hi=hi)


def terrain_dominates(terms: pd.DataFrame, terrain: str, detector: Sequence[str]) -> dict:
    """Pre-registered C2 finding: first-order share of the terrain factor vs the summed
    shares of every term made only of detector factors (seed, audit and seed x audit)."""
    det = set(detector)
    t_share = float(terms.loc[terms["term"] == terrain, "share"].sum())
    d_share = float(terms.loc[terms["term"].str.split(":").apply(
        lambda p: set(p) <= det), "share"].sum())
    return {"terrain_first_order": t_share, "detector_terms": d_share,
            "terrain_dominates": bool(t_share > d_share)}


# --- factorial table checks (U2 Step 3) ---------------------------------------------------------


def check_factorial(long: pd.DataFrame, *, inventory: str = "inventory",
                    member: str = "member", tier: str = "tier",
                    count: str = "count") -> list[str]:
    """Problems in a long table of tier counts: unbalanced cells, totals that change across
    HAND members of one inventory."""
    problems = []
    tiers = sorted(long[tier].unique())
    members = sorted(long[member].unique())
    for inv, g in long.groupby(inventory):
        cells = g.groupby([member, tier]).size()
        if len(cells) != len(members) * len(tiers) or (cells != 1).any():
            problems.append(f"{inv}: not one count per (member, tier)")
            continue
        totals = g.groupby(member)[count].sum()
        if totals.nunique() != 1:
            problems.append(f"{inv}: totals differ across HAND members "
                            f"({totals.min()}-{totals.max()})")
    return problems


def to_array(long: pd.DataFrame, axes: Sequence[str], value: str) -> np.ndarray:
    """Long table -> dense array with one axis per column in ``axes`` (sorted levels)."""
    levels = [sorted(long[a].unique()) for a in axes]
    arr = np.full([len(lv) for lv in levels], np.nan)
    idx = tuple(long[a].map({v: i for i, v in enumerate(lv)}).to_numpy()
                for a, lv in zip(axes, levels, strict=True))
    arr[idx] = long[value].to_numpy(dtype=float)
    return arr


def corrected_counts(d: np.ndarray, p: np.ndarray, r: np.ndarray, tier_group: Sequence[int],
                     target_tiers: Sequence[int]) -> np.ndarray:
    """U2 primary response Y[s, h, b] = sum over target tiers t of D[s, h, t] p[s, h, g(t), b]
    / r[s, h, g(t), b] (ratio-corrected count of seed s under HAND surface h and audit
    replicate b; tier groups fixed from the primary surface)."""
    d = np.asarray(d, dtype=float)
    p = np.asarray(p, dtype=float)
    r = np.asarray(r, dtype=float)
    out = np.zeros((d.shape[0], d.shape[1], p.shape[-1]))
    for t in target_tiers:
        g = tier_group[t]
        with np.errstate(divide="ignore", invalid="ignore"):
            out += d[:, :, t, None] * p[:, :, g, :] / r[:, :, g, :]
    return out
