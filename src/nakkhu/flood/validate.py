"""Event validation of HAND against the September 2024 flood (plan 009 Steps 6-10).

V1  reference: R1 = change-map class on observable buildings; R1-aud = audited items only
    (design weights). Pre-registered drop rule: R1 is not used if the Wilson 95 % lower bound
    of the audited PPV is < 0.5 (``artifacts/audit/flood_trace_estimates.json``).
V2  damage yes/no among audited flood-positive buildings: Wilson share overall and per tier,
    binary logistic on HAND when both classes have >= ``min_per_class`` items (unweighted;
    the design-weighted share is ``nakkhu audit estimate --task damage``).
V3  per-tier hit rates and capture shares (Wilson), H / FAR / CSI / F1 at cumulative HAND cuts,
    ROC-AUC and PR-AUC of -HAND, tier odds ratios (reference > 12 m) with cluster-robust SEs,
    trend test, Moran's I of the trend-model residuals (k nearest neighbours, permutations).
V4  competitors (distance to the OSM channel, GLO-30 elevation, 100 m river buffer, ASF GLO-30
    HAND): delta AUC by DeLong and by a spatial-block bootstrap.
V5  external points: deferred (Decisions: data requests skipped).
V6  SAR/optical negative control: skipped unless the user has an Earthdata login.
H6  logistic and isotonic P(affected | HAND) out of fold over spatial blocks; Brier score,
    reliability, Youden-J cut, likelihood classes >= 0.5, 0.25-0.5, 0.1-0.25, < 0.1.
C3  pass if the block-bootstrap 95 % CI of delta AUC (HAND - distance) lies above 0 and the
    tier odds ratios are monotone.

R1 statistics are unweighted; every R1-aud statistic (rates, cumulative skill, curves, AUCs,
ORs, H6 fits, Brier, reliability, Youden) uses the design weights N_h / n_h of the answered
items. C3 and the H6 method are read once, on the prereg headline cell (detector_final x the
primary reference x the primary surface; :func:`primary_outcome`); everything else is reported.

Every scoring entry point runs behind the plan 005 guard: :func:`run_validate` refuses unless
the prereg is frozen (and holds the change-map signs), the sitting-3 sample, its completed
answers and current estimates exist, and every inventory has its change map and HAND table; it
then loads the non-reference inputs (failing early) and journals one access before scoring.
The defaults of :func:`run_validate`, :func:`block_bootstrap_delta_auc`, :func:`morans_i`,
:func:`damage_model` and :func:`reliability`, and ``LEVEL`` / ``CUTS_M``, are pinned by
prereg v5 (``validation``): never change them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import KDTree

from nakkhu.evaluate.ci import wilson

LEVEL = 0.95
Z = float(stats.norm.ppf(0.5 + LEVEL / 2))
CUTS_M = (0.0, 3.0, 6.0, 9.0, 12.0)
STATUS_V5 = "deferred: UNOSAT / Khatiwada data requests skipped (plans/README.md Decisions)"
STATUS_V6 = "skipped: needs an Earthdata login (plan 009 open question; default skip)"


# --- V3: rates and skill ----------------------------------------------------------------------


def _kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w * w).sum()) if len(w) and (w * w).sum() > 0 else 0.0


def _wilson_w(yes_w: float, n_w: float, n_eff: float) -> tuple[float, float]:
    """Wilson interval of a weighted share on the Kish effective sample size."""
    if n_w <= 0 or n_eff <= 0:
        return float("nan"), float("nan")
    p = yes_w / n_w
    return wilson(int(round(p * n_eff)), int(round(n_eff)), LEVEL) if n_eff >= 1 else (
        float("nan"), float("nan"))


def tier_hit_rates(tiers: np.ndarray, y: np.ndarray, labels: Sequence[str],
                   w: np.ndarray | None = None) -> pd.DataFrame:
    """Per tier: n, affected, hit rate with Wilson CI, and capture (share of all affected).
    With design weights ``w`` the shares are weighted and the Wilson intervals use the Kish
    effective sample size."""
    tiers = np.asarray(tiers)
    y = np.asarray(y).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    weighted = not np.allclose(w, 1.0)
    total_w = float(w[y].sum())
    rows = []
    for k, label in enumerate(labels):
        sel = tiers == k
        n, a = int(sel.sum()), int(y[sel].sum())
        if weighted:
            nw, aw = float(w[sel].sum()), float(w[sel & y].sum())
            lo, hi = _wilson_w(aw, nw, _kish(w[sel]))
            clo, chi = _wilson_w(aw, total_w, _kish(w[y]))
            hit, cap = (aw / nw if nw else float("nan")), (aw / total_w if total_w else
                                                           float("nan"))
        else:
            lo, hi = wilson(a, n, LEVEL)
            clo, chi = wilson(a, int(y.sum()), LEVEL)
            hit, cap = (a / n if n else float("nan")), (a / y.sum() if y.sum() else
                                                        float("nan"))
        rows.append({"tier": label, "n": n, "affected": a, "hit_rate": hit, "hit_lo": lo,
                     "hit_hi": hi, "capture": cap, "capture_lo": clo, "capture_hi": chi,
                     "weighted": weighted})
    return pd.DataFrame(rows)


def cumulative_skill(hand: np.ndarray, y: np.ndarray,
                     cuts: Sequence[float] = CUTS_M,
                     w: np.ndarray | None = None) -> pd.DataFrame:
    """Rule "affected if HAND <= cut": hit rate H (POD), false alarm ratio, CSI and F1.

    ``tp``/``fp``/``fn``/``tn`` are item counts. With design weights ``w`` (R1-aud) H, FAR,
    CSI and F1 come from the weighted counts (``*_w`` columns) and the H interval is a Wilson
    interval on the Kish effective size of the affected items."""
    hand = np.asarray(hand, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    weighted = w is not None and not np.allclose(w, 1.0)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    rows = []
    for cut in cuts:
        pred = hand <= cut
        cells = {"tp": pred & y, "fp": pred & ~y, "fn": ~pred & y, "tn": ~pred & ~y}
        row: dict[str, Any] = {"cut_m": cut, **{k: int(v.sum()) for k, v in cells.items()}}
        if weighted:
            row.update({f"{k}_w": float(sw[v].sum()) for k, v in cells.items()})
            tp, fp, fn = row["tp_w"], row["fp_w"], row["fn_w"]
            h_lo, h_hi = _wilson_w(tp, tp + fn, _kish(sw[y]))
        else:
            tp, fp, fn = row["tp"], row["fp"], row["fn"]
            h_lo, h_hi = wilson(int(tp), int(tp + fn), LEVEL)
        row.update({"H": tp / (tp + fn) if tp + fn else float("nan"), "H_lo": h_lo,
                    "H_hi": h_hi, "FAR": fp / (tp + fp) if tp + fp else float("nan"),
                    "CSI": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
                    "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"),
                    "weighted": weighted})
        rows.append(row)
    return pd.DataFrame(rows)


def roc_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """ROC-AUC (Mann-Whitney, ties count one half); higher score = more likely affected.
    With weights: P(score_pos > score_neg) over weighted pairs."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    if w is None:
        ranks = stats.rankdata(s)
        return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    w = np.asarray(w, dtype=np.float64)
    neg_s, neg_w = s[~y], w[~y]
    order = np.argsort(neg_s, kind="stable")
    neg_s, cum = neg_s[order], np.r_[0.0, np.cumsum(neg_w[order])]
    below = cum[np.searchsorted(neg_s, s[y], side="left")]
    tied = cum[np.searchsorted(neg_s, s[y], side="right")] - below
    return float(np.sum(w[y] * (below + 0.5 * tied)) / (w[y].sum() * neg_w.sum()))


def roc_curve(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> pd.DataFrame:
    """ROC points at every distinct score (design-weighted rates when ``w`` is given)."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s, yy, ww = s[order], y[order], sw[order]
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = np.cumsum(ww * yy)[last], np.cumsum(ww * ~yy)[last]
    pos, neg = float(sw[y].sum()), float(sw[~y].sum())
    return pd.DataFrame({"threshold": s[last], "tpr": tp / (pos if pos > 0 else 1.0),
                         "fpr": fp / (neg if neg > 0 else 1.0)})


def pr_curve(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> pd.DataFrame:
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    s, yy, ww = s[order], y[order], w[order]
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = np.cumsum(ww * yy)[last], np.cumsum(ww * ~yy)[last]
    return pd.DataFrame({"threshold": s[last], "precision": tp / (tp + fp),
                         "recall": tp / max(1e-12, float(w[y].sum()))})


def average_precision(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """PR-AUC as average precision (step sum over distinct thresholds)."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    curve = pr_curve(score, y, w)
    recall = np.r_[0.0, curve["recall"].to_numpy()]
    return float(np.sum(np.diff(recall) * curve["precision"].to_numpy()))


# --- V4: DeLong and block bootstrap -----------------------------------------------------------


def _placements(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos, neg = s[y], s[~y]
    m, n = len(pos), len(neg)
    r_all = stats.rankdata(np.r_[pos, neg])
    v10 = (r_all[:m] - stats.rankdata(pos)) / n           # per positive
    v01 = 1.0 - (r_all[m:] - stats.rankdata(neg)) / m     # per negative
    return v10, v01


def delong(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """DeLong et al. (1988) test of two correlated ROC-AUCs (fast form of Sun and Xu 2014)."""
    y = np.asarray(y).astype(bool)
    a = [np.asarray(s, dtype=np.float64) for s in (s1, s2)]
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return {"auc1": float("nan"), "auc2": float("nan"), "delta": float("nan"),
                "se": float("nan"), "z": float("nan"), "p": float("nan")}
    v10 = np.vstack([_placements(s, y)[0] for s in a])
    v01 = np.vstack([_placements(s, y)[1] for s in a])
    auc = v10.mean(axis=1)
    cov = np.cov(v10) / m + np.cov(v01) / n
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    delta = float(auc[0] - auc[1])
    se = float(np.sqrt(max(var, 0.0)))
    z = delta / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"auc1": float(auc[0]), "auc2": float(auc[1]), "delta": delta, "se": se, "z": z,
            "p": p, "ci_lo": delta - Z * se, "ci_hi": delta + Z * se}


def block_bootstrap_delta_auc(y: np.ndarray, s1: np.ndarray, s2: np.ndarray,
                              blocks: np.ndarray, n_boot: int = 2000, seed: int = 0,
                              w: np.ndarray | None = None) -> dict[str, float]:
    """Percentile CI of AUC(s1) - AUC(s2), resampling whole spatial blocks."""
    y = np.asarray(y).astype(bool)
    s1, s2 = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    blocks = np.asarray(blocks)
    ids, inverse = np.unique(blocks, return_inverse=True)
    members = [np.nonzero(inverse == k)[0] for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[k] for k in pick])
        wi = None if w is None else np.asarray(w)[idx]
        d = roc_auc(s1[idx], y[idx], wi) - roc_auc(s2[idx], y[idx], wi)
        if np.isfinite(d):
            deltas.append(d)
    d = np.asarray(deltas)
    lo, hi = (np.quantile(d, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2]) if d.size
              else (float("nan"), float("nan")))
    return {"delta": roc_auc(s1, y, w) - roc_auc(s2, y, w), "boot_lo": float(lo),
            "boot_hi": float(hi), "n_boot_valid": int(d.size), "n_blocks": len(ids)}


# --- V3: logistic models with cluster-robust errors ------------------------------------------


def logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *,
             max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """(Pseudo-)maximum-likelihood coefficients by Newton-IRLS; ``w`` = design weights."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sw = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        v = sw * np.clip(p * (1 - p), 1e-12, None)
        step = np.linalg.solve((X * v[:, None]).T @ X, X.T @ (sw * (y - p)))
        beta += step
        if np.abs(step).max() < tol:
            break
    return beta


def cluster_robust_cov(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                       clusters: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """CR1 sandwich covariance: G/(G-1) * (N-1)/(N-K) small-sample factor."""
    X = np.asarray(X, dtype=np.float64)
    sw = np.ones(X.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    bread = np.linalg.inv((X * (sw * p * (1 - p))[:, None]).T @ X)
    scores = X * (sw * (np.asarray(y, dtype=np.float64) - p))[:, None]
    ids, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    summed = np.zeros((len(ids), X.shape[1]))
    np.add.at(summed, inverse, scores)
    meat = summed.T @ summed
    g, n, k = len(ids), X.shape[0], X.shape[1]
    factor = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    return factor * bread @ meat @ bread


def tier_odds_ratios(tiers: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                     labels: Sequence[str], reference: int | None = None,
                     w: np.ndarray | None = None) -> pd.DataFrame:
    """Logistic regression of affected on tier dummies (reference = top tier by default);
    OR with cluster-robust 95 % CI. Tiers without any building are left out.

    Separation: a tier whose buildings are all unaffected (or all affected) has no finite
    maximum-likelihood OR. With saturated tier dummies its rows do not inform the other
    coefficients, so it is reported with its limit OR (0 or inf), no interval and
    ``separated`` = True, and the model is fitted on the other tiers (the other ORs equal the
    limit of the full fit). If the reference tier itself is separated no OR is defined: every
    row gets OR NaN and :func:`monotone` returns None (undetermined)."""
    tiers = np.asarray(tiers)
    y = np.asarray(y, dtype=np.float64)
    reference = len(labels) - 1 if reference is None else reference
    present = [k for k in range(len(labels)) if k != reference and (tiers == k).any()]

    def sep(k: int) -> str | None:
        yk = y[tiers == k]
        return "none_affected" if not yk.any() else "all_affected" if yk.all() else None

    nan = float("nan")
    ref_sep = sep(reference) if (tiers == reference).any() else "no_buildings"
    order = {lab: i for i, lab in enumerate(labels)}
    if ref_sep is not None:
        rows = [{"tier": labels[k], "odds_ratio": nan, "or_lo": nan, "or_hi": nan,
                 "se_log": nan, "p": nan, "reference": k == reference,
                 "separated": sep(k) is not None, "separation": sep(k) or ""}
                for k in [*present, reference] if (tiers == k).any()]
        frame = pd.DataFrame(rows)
        frame.attrs["reference_separated"] = ref_sep
        return frame.sort_values("tier", key=lambda s: s.map(order)).reset_index(drop=True)
    fitted = [k for k in present if sep(k) is None]
    keep = np.isin(tiers, [reference, *fitted])
    X = np.column_stack([np.ones(int(keep.sum()))]
                        + [(tiers[keep] == k).astype(float) for k in fitted])
    wk = None if w is None else np.asarray(w)[keep]
    beta = logistic(X, y[keep], wk)
    cov = cluster_robust_cov(X, y[keep], beta, np.asarray(clusters)[keep], wk)
    rows = [{"tier": labels[reference], "odds_ratio": 1.0, "or_lo": 1.0, "or_hi": 1.0,
             "se_log": 0.0, "p": nan, "reference": True, "separated": False,
             "separation": ""}]
    for j, k in enumerate(fitted, start=1):
        se = float(np.sqrt(cov[j, j]))
        rows.append({"tier": labels[k], "odds_ratio": float(np.exp(beta[j])),
                     "or_lo": float(np.exp(beta[j] - Z * se)),
                     "or_hi": float(np.exp(beta[j] + Z * se)), "se_log": se,
                     "p": float(2 * stats.norm.sf(abs(beta[j] / se))) if se > 0 else nan,
                     "reference": False, "separated": False, "separation": ""})
    for k in present:
        how = sep(k)
        if how is not None:
            rows.append({"tier": labels[k], "odds_ratio": 0.0 if how == "none_affected"
                         else float("inf"), "or_lo": nan, "or_hi": nan, "se_log": nan,
                         "p": nan, "reference": False, "separated": True, "separation": how})
    return pd.DataFrame(rows).sort_values("tier", key=lambda s: s.map(order)).reset_index(
        drop=True)


def trend_test(tiers: np.ndarray, y: np.ndarray, clusters: np.ndarray,
               w: np.ndarray | None = None) -> dict[str, Any]:
    """Logistic regression on the ordinal tier index (cluster-robust z test of the slope)."""
    X = np.column_stack([np.ones(len(tiers)), np.asarray(tiers, dtype=np.float64)])
    beta = logistic(X, y, w)
    cov = cluster_robust_cov(X, y, beta, clusters, w)
    se = float(np.sqrt(cov[1, 1]))
    z = float(beta[1] / se) if se > 0 else float("nan")
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    return {"slope_log_or_per_tier": float(beta[1]), "se": se, "z": z,
            "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
            "residuals": np.asarray(y, dtype=np.float64) - p}


def monotone(ors: pd.DataFrame) -> bool | None:
    """Point odds ratios non-increasing from the lowest present tier to the highest (ties
    allowed; a separated tier counts with its limit 0 or inf). None if any OR is undefined
    (reference tier separated)."""
    v = ors["odds_ratio"].to_numpy(dtype=float)
    if np.isnan(v).any():
        return None
    with np.errstate(invalid="ignore"):
        return bool(all(b <= a for a, b in zip(v[:-1], v[1:], strict=True)))


def morans_i(values: np.ndarray, x: np.ndarray, y: np.ndarray, k: int = 8,
             n_perm: int = 999, seed: int = 0) -> dict[str, float]:
    """Moran's I with row-standardised k-nearest-neighbour weights; permutation p-value
    (two-sided, (1 + #|I_perm - E| >= |I - E|) / (1 + n_perm))."""
    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    n = len(z)
    if n <= k + 1 or not np.any(z):
        return {"I": float("nan"), "expected": float("nan"), "p_perm": float("nan"), "k": k}
    _, nb = KDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=k + 1)
    nb = nb[:, 1:]

    def stat(v: np.ndarray) -> float:
        return float((v * v[nb].mean(axis=1)).sum() / (v * v).sum())

    observed = stat(z)
    expected = -1.0 / (n - 1)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    extreme = int((np.abs(perms - expected) >= abs(observed - expected)).sum())
    return {"I": observed, "expected": expected, "p_perm": (1 + extreme) / (1 + n_perm),
            "k": k, "n_perm": n_perm}


# --- V1 references ------------------------------------------------------------------------------


def r1_reference(change: pd.DataFrame, audit_estimates: dict[str, Any]) -> dict[str, Any]:
    """R1 (change-map class on observable buildings) and whether the drop rule removes it."""
    est = audit_estimates.get("estimates", audit_estimates)
    ppv = est["ppv"]
    usable = not bool(est.get("drop_change_map", ppv["wilson_lo"] < 0.5))
    obs = change["observable"].to_numpy(dtype=bool)
    y = (change["class"].astype(str) == "positive").to_numpy()
    return {"usable": usable, "ppv": ppv, "npv": est.get("npv"),
            "building_id": change["building_id"].to_numpy()[obs], "y": y[obs]}


def r1_aud(items: pd.DataFrame, answers: dict[str, str]) -> pd.DataFrame:
    """Audited flood_trace items with a yes/no answer: building_id, y and design weight."""
    a = items.assign(answer=items["item_id"].map(answers))
    a = a.loc[a["answer"].isin(["yes", "no"])].copy()
    a["y"] = (a["answer"] == "yes").astype(int)
    a["w"] = a["frame_n"].astype(float) / a.groupby("stratum")["y"].transform("size")
    return a[["item_id", "building_id", "stratum", "y", "w"]] if "building_id" in a.columns \
        else a[["item_id", "unit", "stratum", "y", "w"]]


# --- V2 ---------------------------------------------------------------------------------------


def damage_model(hand: np.ndarray, damaged: np.ndarray, clusters: np.ndarray,
                 min_per_class: int = 10) -> dict[str, Any]:
    """Damage share among audited flood-positive buildings; binary logistic on HAND if both
    classes have at least ``min_per_class`` items (ordinal grades are not collected)."""
    d = np.asarray(damaged).astype(bool)
    k, n = int(d.sum()), len(d)
    lo, hi = wilson(k, n, LEVEL)
    out: dict[str, Any] = {"n": n, "damaged": k, "share": k / n if n else float("nan"),
                           "wilson_lo": lo, "wilson_hi": hi, "model": "descriptive only"}
    if k >= min_per_class and n - k >= min_per_class:
        X = np.column_stack([np.ones(n), np.asarray(hand, dtype=np.float64)])
        beta = logistic(X, d.astype(float))
        cov = cluster_robust_cov(X, d.astype(float), beta, clusters)
        se = float(np.sqrt(cov[1, 1]))
        out.update({"model": "binary logistic on HAND", "or_per_m": float(np.exp(beta[1])),
                    "or_lo": float(np.exp(beta[1] - Z * se)),
                    "or_hi": float(np.exp(beta[1] + Z * se))})
    return out


# --- H6 ---------------------------------------------------------------------------------------


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10, *,
                w: np.ndarray | None = None) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean predicted vs observed rate (both
    design-weighted within the bin when ``w`` is given)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p)
    sw = np.ones(int(ok.sum())) if w is None else np.asarray(w, dtype=np.float64)[ok]
    p, y = p[ok], y[ok]
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return pd.DataFrame([{"bin": i, "n": len(ix),
                          "p_mean": float(np.average(p[ix], weights=sw[ix])),
                          "observed": float(np.average(y[ix], weights=sw[ix]))}
                         for i, ix in enumerate(parts) if len(ix)])


H6_METHODS = ("logistic", "isotonic")


def h6_primary_method(briers: dict[str, float]) -> str | None:
    """Prereg ``validation.h6.primary``: the lower out-of-fold Brier score; tie -> logistic."""
    lo, iso = briers.get("logistic", float("nan")), briers.get("isotonic", float("nan"))
    if not np.isfinite(lo) and not np.isfinite(iso):
        return None
    if not np.isfinite(iso) or (np.isfinite(lo) and lo <= iso):
        return "logistic"
    return "isotonic"


def h6_calibration(hand: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                   tiers: np.ndarray, tier_labels: Sequence[str],
                   w: np.ndarray | None = None,
                   edges: Sequence[float] = CUTS_M) -> dict[str, Any]:
    """Event-calibrated classes (H6): out-of-fold (leave-one-block-out) logistic and isotonic
    P(affected | HAND), Brier score and reliability (design-weighted with ``w``), likelihood
    classes, the lower-Brier method, the Youden-J HAND cut, and both models refitted on every
    row (curve values at the tier edges, for the text)."""
    from nakkhu import hazard

    hand = np.asarray(hand, dtype=np.float64)
    y = np.asarray(y).astype(float)
    sw = None if w is None else np.asarray(w, dtype=np.float64)
    out: dict[str, Any] = {"n": int(len(y)), "affected": int(y.sum()),
                           "weighted": sw is not None, "methods": {}}
    for method in H6_METHODS:
        p = hazard.calibrate_out_of_fold(hand, y, blocks, method, sw)
        ok = np.isfinite(p)
        wk = None if sw is None else sw[ok]
        cls = hazard.likelihood_class(np.where(ok, p, 0.0))
        cross = pd.crosstab(pd.Series(np.asarray(tier_labels)[tiers[ok]], name="tier"),
                            pd.Series(np.asarray(hazard.LIKELIHOOD_LABELS)[cls[ok]],
                                      name="class"))
        entry: dict[str, Any] = {
            "brier": hazard.brier(p[ok], y[ok], wk) if ok.any() else float("nan"),
            "n_scored": int(ok.sum()),
            "reliability": reliability(p, y, w=sw).to_dict("records"),
            "class_counts": {lab: int((cls[ok] == i).sum())
                             for i, lab in enumerate(hazard.LIKELIHOOD_LABELS)},
            "tier_by_class": cross.to_dict()}
        if wk is not None:
            tot = float(wk.sum())
            entry["class_shares_weighted"] = {
                lab: float(wk[cls[ok] == i].sum() / tot) if tot else float("nan")
                for i, lab in enumerate(hazard.LIKELIHOOD_LABELS)}
        out["methods"][method] = entry
    out["lower_brier_method"] = h6_primary_method(
        {m: d["brier"] for m, d in out["methods"].items()})
    if 0 < y.sum() < len(y):
        cut, j = hazard.youden_threshold(hand, y.astype(bool), sw)
        a, b = hazard.fit_logistic(hand, y, sw)
        knots = hazard.fit_isotonic(hand, y, sw)
        at = np.asarray(edges, dtype=np.float64)
        out["youden"] = {"hand_cut_m": cut, "J": j}
        out["fit_all_rows"] = {
            "logistic": {"intercept": a, "slope_per_m": b,
                         "p_at_hand_m": dict(zip(map(str, at),
                                                 hazard.predict_logistic((a, b), at).tolist(),
                                                 strict=True))},
            "isotonic": {"n_knots": int(len(knots[0])),
                         "p_at_hand_m": dict(zip(map(str, at),
                                                 hazard.predict_isotonic(knots, at).tolist(),
                                                 strict=True))}}
    else:
        out["youden"] = {"hand_cut_m": float("nan"), "J": float("nan")}
    out["note"] = "calibrated to one event (Sept 2024), not a return-period hazard"
    return out


# --- C3 ---------------------------------------------------------------------------------------


def c3_outcome(boot_vs_distance: dict[str, float], delong_vs_distance: dict[str, float],
               ors: pd.DataFrame) -> dict[str, Any]:
    """C3 (prereg ``falsifiers.C3``): block-bootstrap CI of delta AUC (HAND - distance) above
    0 and monotone tier ORs. ``pass`` is None (undetermined, reported for a user decision)
    when the CI condition holds but the ORs are undefined (reference tier separated)."""
    ci_ok = bool(np.isfinite(boot_vs_distance["boot_lo"]) and boot_vs_distance["boot_lo"] > 0)
    mono = monotone(ors)
    verdict = False if not ci_ok else (None if mono is None else bool(mono))
    return {"pass": verdict, "delta_auc_ci_above_0": ci_ok, "ors_monotone": mono,
            "delta_auc": boot_vs_distance["delta"],
            "boot_ci": [boot_vs_distance["boot_lo"], boot_vs_distance["boot_hi"]],
            "delong_p": delong_vs_distance.get("p")}


# --- orchestration (behind the guard) ---------------------------------------------------------


HASH_KEYS = ("change_map_sha256", "inventory_sha256", "buildings_sha256")
FILE_HASH_KEY = "change_map_file_sha256"         # the change-map table bytes (plan 006 sample)
SAMPLE_JSON = ("audit", "flood_trace", "sample.json")   # under the project root
ESTIMATES = "audit/flood_trace_estimates.json"         # under artifacts/
CHANNEL_LINES = "configs/aoi/osm_nakkhu_bagmati_lines.geojson"
ELEVATION = "raw/dem_glo30/dem_glo30_routing2km.tif"   # under data_root
ASF_HAND = "raw/hand_asf_glo30/hand_asf_glo30_routing2km.tif"
BLOCKS = "splits/blocks.geojson"                       # under artifacts/ (prereg validation.blocks)
CHANGE_COLUMNS = ["building_id", "class", "observable", "class_all_strips",
                  "observable_all_strips"]


def change_map_path(project: Any, inventory: str) -> Path | None:
    """The promoted change map of an audit inventory, else the latest run's table."""
    art = project.paths.artifacts / "flood" / f"change_map_buildings_{inventory}.parquet"
    return art if art.is_file() else _latest_file(
        project, "flood", f"change_map_buildings_{inventory}.parquet")


def hand_table_path(project: Any, inventory: str) -> Path | None:
    """The latest plan 008 building HAND table of an inventory (``nakkhu hand attribute``)."""
    return _latest_file(project, "hand-attribute", f"building_hand_{inventory}.parquet")


def sample_inputs(project: Any) -> dict[str, Any] | None:
    """The input hashes recorded by ``nakkhu audit sample --task flood_trace`` (None if the
    sitting-3 sample does not exist)."""
    path = project.paths.root.joinpath(*SAMPLE_JSON)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("inputs") or {}


def sample_hash_drift(project: Any) -> list[str]:
    """Hashes recorded in the sitting-3 sample that no longer match the change-map sidecar of
    the sampled inventory or the change-map file itself (empty if the sample predates hash
    recording or does not exist)."""
    from nakkhu.evaluate.guard import sha256_file

    inputs = sample_inputs(project) or {}
    inv = inputs.get("inventory")
    if not inv:
        return []
    side = project.paths.artifacts / "flood" / f"change_map_buildings_{inv}.json"
    live = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
    drift = [k for k in HASH_KEYS if k in inputs and live.get(k) != inputs[k]]
    if inputs.get(FILE_HASH_KEY) is not None:
        table = project.paths.artifacts / "flood" / f"change_map_buildings_{inv}.parquet"
        if not table.is_file() or sha256_file(table) != inputs[FILE_HASH_KEY]:
            drift.append(FILE_HASH_KEY)
    return drift


def audited_inventory(project: Any) -> str:
    """The sitting-3 inventory fixed by the prereg (``audit_flood_trace.values.inventory``)."""
    from nakkhu.evaluate.guard import load_prereg

    try:
        return str(load_prereg(project)["audit_flood_trace"]["values"]["inventory"])
    except Exception:          # no prereg: verify() reports it; the frozen value is the default
        return "detector_final"


def estimates_drift(project: Any) -> list[str]:
    """Audit files that changed since ``nakkhu audit estimate --task flood_trace`` hashed them
    (the estimates must come from the final items and responses)."""
    from nakkhu.audit import core as audit_core

    path = project.paths.artifacts / ESTIMATES
    if not path.is_file():
        return []
    material = json.loads(path.read_text(encoding="utf-8")).get("material") or {}
    out = []
    for key, file in (("items", audit_core.items_path(project, "flood_trace")),
                      ("responses", audit_core.responses_path(project, "flood_trace"))):
        live = audit_core.sha256_file(file) if file.is_file() else None
        if material.get(key) != live:
            out.append(key)
    return out


def prerequisites(project: Any, inventory_path: Path | None,
                  inventories: Sequence[str] = ()) -> list[str]:
    """What is missing before ``nakkhu flood validate`` may score anything. Reads hashes,
    file presence and the sitting-3 completion state only (no answer values)."""
    from nakkhu.audit import core as audit_core
    from nakkhu.evaluate.guard import load_prereg, verify

    problems: list[str] = []
    state = verify(project)
    problems += [f"prereg: {p}" for p in state["problems"]]
    try:
        cm = (load_prereg(project).get("change_map") or {})
    except Exception as error:   # missing prereg is already reported by verify
        cm = {}
        problems.append(str(error))
    if not cm.get("signs"):
        problems.append("prereg has no change_map.signs (change-map section not yet frozen; plan 009 Outcome)")
    inputs = sample_inputs(project)
    want = audited_inventory(project)
    if inputs is None:
        problems.append("audit/flood_trace/sample.json missing (sitting-3 sample not drawn: "
                        "`nakkhu audit sample --task flood_trace`)")
    elif inputs.get("inventory") != want:
        problems.append(f"sitting-3 sample was drawn on {inputs.get('inventory')!r}; prereg "
                        f"audit_flood_trace.inventory is {want!r}")
    items = audit_core.items_path(project, "flood_trace")
    if items.is_file():
        try:
            from nakkhu.audit.workflow import require_complete, require_followups

            audit_cfg = audit_core.load_audit_config(project)
            require_complete(project, audit_cfg, "flood_trace")
            require_followups(project, audit_cfg, "flood_trace")
        except Exception as error:
            problems.append(f"sitting 3: {error}")
    elif inputs is not None:
        problems.append("audit/flood_trace/items.parquet missing")
    est = project.paths.artifacts / ESTIMATES
    if not est.is_file():
        problems.append("artifacts/audit/flood_trace_estimates.json missing (audit sitting 3)")
    stale = estimates_drift(project)
    if stale:
        problems.append("artifacts/audit/flood_trace_estimates.json was computed from other "
                        f"flood_trace {' and '.join(stale)} than the current ones")
    for inv in inventories:
        if change_map_path(project, inv) is None:
            problems.append(f"change map for {inv} missing (nakkhu flood change)")
        table = hand_table_path(project, inv)
        if table is None:
            problems.append(f"building HAND table for {inv} missing "
                            f"(`nakkhu hand attribute --inventory {inv}`)")
        elif not (table.parent / "count_stability.csv").is_file():
            problems.append(f"{table.parent.name}: count_stability.csv missing next to the "
                            f"HAND table of {inv} (routing-gate flags)")
    for label, path in (("channel lines", project.paths.resolve(CHANNEL_LINES)),
                        ("GLO-30 elevation", project.paths.data_root / ELEVATION),
                        ("ASF GLO-30 HAND", project.paths.data_root / ASF_HAND),
                        ("plan 005 blocks", project.paths.artifacts / BLOCKS)):
        if not path.is_file():
            problems.append(f"{label} {path} missing")
    drift = sample_hash_drift(project)
    if drift:
        problems.append("change map or inventory changed since the sitting-3 sample was drawn ("
                        + ", ".join(drift) + ")")
    if inventory_path is not None and not inventory_path.is_file():
        problems.append(f"inventory {inventory_path} missing (plan 007 final inventory)")
    return problems


def load_audit_estimates(project: Any) -> dict[str, Any]:
    path = project.paths.artifacts / ESTIMATES
    return json.loads(path.read_text(encoding="utf-8"))


def audit_answers(project: Any) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Sitting-3 items and the effective primary answers (repeats ignored, undo honoured):
    ``(items, flood_trace answers, damage answers)``."""
    from nakkhu.audit import core as audit_core

    items = audit_core.load_items(project, "flood_trace")
    state = audit_core.replay(audit_core.read_events(audit_core.responses_path(
        project, "flood_trace")))
    ft = {e["item_id"]: e["answer"] for e in state.effective
          if e["question"] == "flood_trace" and not e.get("repeat")}
    dmg = {e["item_id"]: e["answer"] for e in state.effective
           if e["question"] == "damage" and not e.get("repeat")}
    return items, ft, dmg


# --- the validation run -----------------------------------------------------------------------

VALIDATION_ARTIFACTS = ("tier_hit_rates", "cumulative_skill", "roc_curves", "pr_curves",
                        "odds_ratios", "competitors_auc", "config_skill_grid",
                        "config_skill_range", "calibration_h6", "h6_reliability",
                        "by_inventory")
META_COLUMNS = ["surface", "cfg_id", "dem", "conditioning", "threshold", "threshold_km2",
                "statistic", "diagnostic", "routing_gate_pass", "in_sensitivity_range"]
RANGE_FACTORS = ("dem", "conditioning", "threshold")


def _latest_file(project: Any, stage: str, name: str) -> Path | None:
    from nakkhu.runs import list_runs

    runs = [r for r in list_runs(project.paths, stage) if r.status == "complete"
            and (r.path / name).is_file()]
    return runs[-1].path / name if runs else None


def _sample_raster(path: Path, x: np.ndarray, y: np.ndarray, crs: Any) -> np.ndarray:
    """Nearest-cell values of band 1 at points ``(x, y)`` given in ``crs``.

    The points are reprojected to the raster's CRS first. Points outside the raster and nodata
    or non-finite cells are NaN, never the boundless fill value. Before this fix the points
    (EPSG:32645 metres) were passed unprojected to EPSG:4326 rasters without nodata, so every
    point fell outside and read the fill 0: a constant predictor with AUC 0.5 (journal 47,
    plans/competitor_auc_bug.md). Raises when no point lands on a valid cell."""
    import rasterio
    from pyproj import CRS
    from rasterio.warp import transform as warp_transform

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    vals = np.full(len(x), np.nan)
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"{path}: raster has no CRS")
        src = CRS.from_user_input(crs)
        if src == CRS.from_wkt(ds.crs.to_wkt()):
            xs, ys = x, y
        else:
            xs, ys = (np.asarray(v, dtype=np.float64)
                      for v in warp_transform(src.to_wkt(), ds.crs, x.tolist(), y.tolist()))
        inv = ~ds.transform
        cols, rows = inv.a * xs + inv.b * ys + inv.c, inv.d * xs + inv.e * ys + inv.f
        with np.errstate(invalid="ignore"):
            inside = (np.isfinite(cols) & np.isfinite(rows) & (cols >= 0) & (rows >= 0)
                      & (cols < ds.width) & (rows < ds.height))
        if inside.any():
            got = np.array([v[0] for v in ds.sample(zip(xs[inside], ys[inside], strict=True))],
                           dtype=np.float64)
            if ds.nodata is not None:
                got[(got == ds.nodata) | (np.isnan(ds.nodata) & np.isnan(got))] = np.nan
            vals[inside] = got
    vals[~np.isfinite(vals)] = np.nan
    if not np.isfinite(vals).any():
        raise ValueError(f"{path}: no point lands on a valid cell (CRS or extent mismatch?)")
    return vals


def predictors(project: Any, inventory: str, sweep_config: str) -> pd.DataFrame:
    """Per building (plan 008 order and centroids): every HAND column, the V4 competitors and
    the spatial block. This is validation code: it reads HAND, DEM and channel layers."""
    import geopandas as gpd
    import shapely

    from nakkhu.hydro.sweep import load_greenhouses, load_inventory, load_settings
    from nakkhu.splits import read_lines

    settings, _ = load_settings(project, sweep_config)
    extent = settings.read_polygon(settings.v["analysis_extent"])
    frame, _ = load_inventory(settings, inventory, extent, load_greenhouses(settings))
    table_path = hand_table_path(project, inventory)
    if table_path is None:
        raise FileNotFoundError(f"no building HAND table for {inventory} (plan 008)")
    hand = pd.read_parquet(table_path)
    if len(hand) != len(frame) or not (hand["building_uid"].to_numpy()
                                       == frame["building_uid"].to_numpy()).all():
        raise ValueError(f"{inventory}: building order differs from the HAND table")
    cent = frame.geometry.centroid
    x, y = cent.x.to_numpy(), cent.y.to_numpy()
    lines = shapely.union_all(read_lines([project.paths.resolve(CHANNEL_LINES)]))
    dist = shapely.distance(shapely.points(x, y), lines)
    data_root = project.paths.data_root
    elev = _sample_raster(data_root / ELEVATION, x, y, frame.crs)
    asf = _sample_raster(data_root / ASF_HAND, x, y, frame.crs)
    for name, v in (("GLO-30 elevation", elev), ("ASF GLO-30 HAND", asf)):
        if np.unique(v[np.isfinite(v)]).size < 2:
            raise ValueError(f"{inventory}: {name} is constant over the buildings")
    blocks = gpd.read_file(project.paths.artifacts / BLOCKS).to_crs(frame.crs)
    pts = gpd.GeoDataFrame({"_i": np.arange(len(frame))}, geometry=cent, crs=frame.crs)
    joined = gpd.sjoin(pts, blocks[["block_id", "geometry"]], how="left", predicate="within")
    joined = joined.sort_values("_i").drop_duplicates("_i")
    out = hand.copy()
    out["x"], out["y"] = x, y
    out["dist_channel_m"] = dist
    out["elev_glo30_m"] = elev
    out["asf_hand_m"] = asf
    out["block_id"] = joined["block_id"].fillna("none").to_numpy()
    return out


def surface_meta(table_path: Path, inventory: str) -> pd.DataFrame:
    """Per HAND column of a building table: DEM, conditioning, threshold, statistic and the
    plan 008 flags (``routing_gate_pass``, ``in_sensitivity_range``) from the
    ``count_stability.csv`` of the same attribute run."""
    cs = pd.read_csv(table_path.parent / "count_stability.csv")
    cs = cs.loc[cs["inventory"].astype(str) == inventory]
    cs = cs.assign(surface=cs["cfg_id"].astype(str) + "__" + cs["statistic"].astype(str))
    return cs[[c for c in META_COLUMNS if c in cs.columns]].drop_duplicates("surface")


def _w(w: np.ndarray | None, sel: np.ndarray) -> np.ndarray | None:
    return None if w is None else np.asarray(w)[sel]


def config_skill_ranges(grid: pd.DataFrame) -> pd.DataFrame:
    """H2-H5 (prereg ``validation.h2_h5``): AUC / PR-AUC of the primary surface and their
    min-max over the routing-gate members only (``in_sensitivity_range``), overall, per
    statistic and per DEM / conditioning / threshold among centroid members; the excluded
    surfaces are counted and listed, never part of a range."""
    if "in_sensitivity_range" not in grid.columns or grid.empty:
        return pd.DataFrame()
    inside = grid["in_sensitivity_range"].fillna(False).astype(bool)
    prim = grid.loc[grid["is_primary"].astype(bool)]
    members = grid.loc[inside]
    groups: list[tuple[str, pd.DataFrame]] = [("all surfaces x statistics", members)]
    groups += [(f"statistic={k}", g) for k, g in members.groupby("statistic", sort=False)]
    cent = members.loc[members["statistic"] == "centroid"]
    for factor in RANGE_FACTORS:
        if factor in cent.columns:
            groups += [(f"centroid, {factor}={k}", g) for k, g in cent.groupby(factor,
                                                                              sort=False)]
    rows: list[dict[str, Any]] = []
    for name, g in groups:
        row: dict[str, Any] = {"group": name, "n_members": len(g)}
        for m in ("auc", "pr_auc"):
            row[f"{m}_primary"] = float(prim[m].iloc[0]) if len(prim) else float("nan")
            row[f"{m}_min"] = float(g[m].min()) if len(g) else float("nan")
            row[f"{m}_max"] = float(g[m].max()) if len(g) else float("nan")
        rows.append(row)
    excluded = grid.loc[~inside]
    rows.append({"group": "excluded (diagnostic or routing gate)", "n_members": len(excluded),
                 "excluded_surfaces": ";".join(sorted(excluded["surface"].astype(str)))})
    return pd.DataFrame(rows)


def _primary_rows(pred: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, primary_col: str
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray]:
    """The rows of one reference with a finite primary HAND: ``(pred, y, w, hand)``."""
    hand = pred[primary_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(hand)
    return pred.loc[ok].reset_index(drop=True), np.asarray(y)[ok], _w(w, ok), hand[ok]


def competitor_rows(pred: pd.DataFrame, y: np.ndarray, w: np.ndarray | None,
                    score: np.ndarray, blocks: np.ndarray, n_boot: int, seed: int
                    ) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """V4 on one reference (rows already restricted by :func:`_primary_rows`; ``score`` =
    -HAND): one row per competitor, plus the distance block bootstrap and DeLong for C3."""
    comps = {"distance_to_channel": -pred["dist_channel_m"].to_numpy(dtype=float),
             "glo30_elevation": -pred["elev_glo30_m"].to_numpy(dtype=float),
             "river_buffer_100m": (pred["dist_channel_m"].to_numpy() <= 100).astype(float),
             "asf_glo30_hand": -pred["asf_hand_m"].to_numpy(dtype=float)}
    rows = []
    c3_boot: dict[str, float] = {}
    c3_delong: dict[str, float] = {}
    for name, comp in comps.items():
        good = np.isfinite(comp)
        dl = delong(y[good], score[good], comp[good]) if w is None else {}
        bb = block_bootstrap_delta_auc(y[good], score[good], comp[good], blocks[good], n_boot,
                                       seed, _w(w, good))
        rows.append({"competitor": name, "auc_hand": roc_auc(score[good], y[good], _w(w, good)),
                     "auc_competitor": roc_auc(comp[good], y[good], _w(w, good)),
                     "delta_auc": bb["delta"], "boot_lo": bb["boot_lo"],
                     "boot_hi": bb["boot_hi"], "n_boot_valid": bb["n_boot_valid"],
                     "n_blocks": bb["n_blocks"], "delong_se": dl.get("se"),
                     "delong_p": dl.get("p"), "n": int(good.sum())})
        if name == "distance_to_channel":
            c3_boot, c3_delong = bb, dl
    return pd.DataFrame(rows), c3_boot, c3_delong


def evaluate_reference(pred: pd.DataFrame, y: np.ndarray, w: np.ndarray | None,
                       primary_col: str, labels: Sequence[str], edges: Sequence[float],
                       n_boot: int, seed: int,
                       meta: pd.DataFrame | None = None) -> dict[str, Any]:
    """V3 + V4 + H2-H5 + H6 on one reference (rows of ``pred`` aligned with ``y``; ``w`` =
    design weights for R1-aud, None for R1)."""
    from nakkhu import hazard

    pred, y, w, hand = _primary_rows(pred, y, w, primary_col)
    tiers = hazard.tier_index(hand, edges)
    blocks = pred["block_id"].to_numpy()
    res: dict[str, Any] = {"n": int(len(y)), "affected": int(np.sum(y)),
                           "weighted": w is not None}
    res["tier_hit_rates"] = tier_hit_rates(tiers, y, labels, w)
    res["cumulative_skill"] = cumulative_skill(hand, y, w=w)
    score = -hand
    res["auc"] = roc_auc(score, y, w)
    res["pr_auc"] = average_precision(score, y, w)
    res["prevalence"] = float(np.average(y, weights=w))
    res["roc_curve"] = roc_curve(score, y, w)
    res["pr_curve"] = pr_curve(score, y, w)
    ors = tier_odds_ratios(tiers, y.astype(float), blocks, labels, w=w)
    res["odds_ratios"] = ors
    res["or_reference_separated"] = ors.attrs.get("reference_separated")
    trend = trend_test(tiers, y.astype(float), blocks, w)
    resid = trend.pop("residuals")
    res["trend"] = trend
    res["morans_i"] = morans_i(resid, pred["x"].to_numpy(), pred["y"].to_numpy(), seed=seed)
    res["competitors_auc"], c3_boot, c3_delong = competitor_rows(pred, y, w, score, blocks,
                                                                n_boot, seed)
    res["c3"] = c3_outcome(c3_boot, c3_delong, ors)
    grid = []
    for col in [c for c in pred.columns if "__" in c]:
        v = pred[col].to_numpy(dtype=float)
        good = np.isfinite(v)
        grid.append({"surface": col, "auc": roc_auc(-v[good], y[good], _w(w, good)),
                     "pr_auc": average_precision(-v[good], y[good], _w(w, good)),
                     "n": int(good.sum()), "is_primary": col == primary_col})
    skill = pd.DataFrame(grid)
    if meta is not None and len(skill):
        skill = skill.merge(meta, on="surface", how="left")
        skill["in_sensitivity_range"] = skill["in_sensitivity_range"].fillna(False).astype(bool)
    res["config_skill_grid"] = skill
    res["config_skill_range"] = config_skill_ranges(skill)
    res["h6"] = h6_calibration(hand, y, blocks, tiers, labels, w, edges)
    return res


def damage_by_tier(hand: np.ndarray, damaged: np.ndarray, edges: Sequence[float],
                   labels: Sequence[str]) -> list[dict[str, Any]]:
    """V2 descriptive: damaged share (Wilson) of the audited flood-positive items per tier."""
    from nakkhu import hazard

    tiers = hazard.tier_index(np.asarray(hand, dtype=np.float64), edges)
    d = np.asarray(damaged).astype(bool)
    rows = []
    for k, label in enumerate(labels):
        sel = tiers == k
        n, a = int(sel.sum()), int(d[sel].sum())
        lo, hi = wilson(a, n, LEVEL)
        rows.append({"tier": label, "n": n, "damaged": a,
                     "share": a / n if n else float("nan"), "wilson_lo": lo, "wilson_hi": hi})
    return rows


Reference = tuple[pd.DataFrame, np.ndarray, np.ndarray | None]


def inventory_references(pred: pd.DataFrame, change: pd.DataFrame, estimates: dict[str, Any],
                         items: pd.DataFrame, ft: dict[str, str]
                         ) -> tuple[dict[str, Reference], dict[str, Any], pd.DataFrame,
                                    str | None]:
    """The references of one inventory, ``{name: (rows of pred, y, design weights)}``: R1 and
    R1-all-strips if the drop rule keeps the change map, R1-aud if audited items of this
    inventory exist. Also returns the R1 record, the audited items and the primary reference."""
    r1 = r1_reference(change, estimates)
    refs: dict[str, Reference] = {}
    if r1["usable"]:
        # primary: gated strips only; sensitivity: every strip (gap-fill buildings in)
        sens = change.rename(columns={"class": "_c", "observable": "_o"}).rename(
            columns={"class_all_strips": "class", "observable_all_strips": "observable"})
        for ref, r in (("R1", r1), ("R1-all-strips", r1_reference(sens, estimates))):
            ymap = dict(zip(r["building_id"], r["y"], strict=True))
            sub = pred.loc[pred["building_uid"].isin(set(r["building_id"]))]
            sub = sub.reset_index(drop=True)
            refs[ref] = (sub, sub["building_uid"].map(ymap).to_numpy(dtype=int), None)
    # answer / weight columns renamed: ``pred`` already has the centroid coordinate ``y``
    aud = r1_aud(items, ft).rename(columns={"y": "affected", "w": "design_w"})
    if "building_id" in aud.columns and aud["building_id"].isin(pred["building_uid"]).any():
        sub = pred.merge(aud, left_on="building_uid", right_on="building_id")
        refs["R1-aud"] = (sub, sub["affected"].to_numpy(dtype=int),
                          sub["design_w"].to_numpy())
    primary_ref = "R1" if r1["usable"] else ("R1-aud" if "R1-aud" in refs else None)
    return refs, r1, aud, primary_ref


def score_inventory(inventory: str, pred: pd.DataFrame, change: pd.DataFrame,
                    estimates: dict[str, Any], items: pd.DataFrame, ft: dict[str, str],
                    dmg: dict[str, str], primary_col: str, labels: Sequence[str],
                    edges: Sequence[float], n_boot: int, seed: int,
                    meta: pd.DataFrame | None = None
                    ) -> tuple[dict[str, Any], dict[str, list[pd.DataFrame]]]:
    """Every reference of one inventory: R1 and R1-all-strips (if the drop rule keeps the
    change map) and R1-aud (audited items of this inventory, design weights); V2 on the
    audited flood-positive items. Returns the summary entry and the table parts."""
    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in VALIDATION_ARTIFACTS}
    refs, r1, aud, primary_ref = inventory_references(pred, change, estimates, items, ft)
    inv_sum: dict[str, Any] = {"r1_usable": r1["usable"], "ppv": r1["ppv"], "npv": r1["npv"],
                               "primary_column": primary_col, "primary_reference": primary_ref,
                               "references": {}}
    for ref, (sub, y, w) in refs.items():
        res = evaluate_reference(sub, y, w, primary_col, labels, edges, n_boot, seed, meta)
        tag = {"inventory": inventory, "reference": ref}
        for key in ("tier_hit_rates", "cumulative_skill", "odds_ratios", "competitors_auc",
                    "config_skill_grid", "config_skill_range"):
            if len(res[key]):
                tables[key].append(res[key].assign(**tag))
        tables["roc_curves"].append(res["roc_curve"].assign(**tag))
        tables["pr_curves"].append(res["pr_curve"].assign(**tag))
        h6 = res["h6"]
        tables["calibration_h6"].append(pd.DataFrame(
            [{**tag, "method": m, "brier": d["brier"], "n_scored": d["n_scored"],
              "weighted": h6["weighted"], "lower_brier": m == h6["lower_brier_method"],
              **{f"n_{k}": v for k, v in d["class_counts"].items()}}
             for m, d in h6["methods"].items()]))
        tables["h6_reliability"].append(pd.DataFrame(
            [{**tag, "method": m, **row} for m, d in h6["methods"].items()
             for row in d["reliability"]]))
        tables["by_inventory"].append(pd.DataFrame([{
            **tag, "n": res["n"], "affected": res["affected"], "prevalence": res["prevalence"],
            "auc": res["auc"], "pr_auc": res["pr_auc"], "weighted": res["weighted"],
            "primary_reference": ref == primary_ref}]))
        inv_sum["references"][ref] = {
            k: res[k] for k in ("n", "affected", "prevalence", "auc", "pr_auc", "weighted",
                                "trend", "morans_i", "c3", "h6", "or_reference_separated")}
    if dmg and "affected" in aud.columns and "building_id" in aud.columns:
        pos = aud.loc[aud["affected"] == 1].copy()
        pos["d_answer"] = pos["item_id"].map(dmg)
        pos = pos.loc[pos["d_answer"].isin(["yes", "no"])]
        if len(pos):
            pos["d"] = (pos["d_answer"] == "yes").astype(int)
            sub = pred.merge(pos, left_on="building_uid", right_on="building_id")
            if len(sub):
                inv_sum["V2_damage"] = damage_model(sub[primary_col].to_numpy(),
                                                    sub["d"].to_numpy(),
                                                    sub["block_id"].to_numpy())
                inv_sum["V2_damage"]["by_tier"] = damage_by_tier(
                    sub[primary_col].to_numpy(), sub["d"].to_numpy(), edges, labels)
    return inv_sum, tables


def primary_outcome(summary: dict[str, Any], inventory: str = "detector_final"
                    ) -> dict[str, Any]:
    """The pre-registered headline cell (prereg ``validation.c3`` / ``primary_reference`` /
    ``h6.primary``): detector_final x the primary reference (R1 if the drop rule keeps it,
    else R1-aud) x the primary HAND surface; C3 and the H6 method are read only here."""
    inv = summary["inventories"].get(inventory)
    if not inv or "references" not in inv:
        return {"status": f"{inventory} not validated: C3 and the H6 method are not evaluated"}
    ref = inv.get("primary_reference")
    if ref is None or ref not in inv["references"]:
        return {"status": f"{inventory}: no primary reference (R1 dropped and no R1-aud items)"}
    r = inv["references"][ref]
    h6 = r["h6"]
    method = h6.get("lower_brier_method")
    return {"inventory": inventory, "reference": ref, "surface": inv["primary_column"],
            "n": r["n"], "affected": r["affected"], "auc": r["auc"], "pr_auc": r["pr_auc"],
            "C3": r.get("c3"), "h6_method": method,
            "h6_brier": {m: d["brier"] for m, d in h6["methods"].items()},
            "h6_classes": h6["methods"][method]["class_counts"] if method else None,
            "h6_youden": h6.get("youden"), "h6_fit_all_rows": h6.get("fit_all_rows"),
            "note": "C3 is evaluated once, on this cell; other inventories and references are "
                    "reported, not gating (prereg validation.c3)"}


def v1_summary(estimates: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """V1: the audited change-map PPV / NPV and the drop decision, with the hashes the
    sitting-3 sample was drawn on (``artifacts/flood/change_map_audit_summary.json``)."""
    est = estimates.get("estimates", estimates)
    ppv = est.get("ppv") or {}
    drop = bool(est.get("drop_change_map", ppv.get("wilson_lo", 0.0) < 0.5))
    return {"inventory": inputs.get("inventory"), "ppv": ppv, "npv": est.get("npv"),
            "not_observable": est.get("not_observable"), "drop_change_map": drop,
            "r1_used": not drop,
            "rule": "R1 is not used if the Wilson 95 % lower bound of the audited PPV is < 0.5 "
                    "(prereg change_map.drop_rule); V3 then runs on R1-aud only",
            "estimates_material": estimates.get("material"),
            "sample_hashes": {k: inputs.get(k) for k in (*HASH_KEYS, FILE_HASH_KEY)}}


def jsonable(value: Any) -> Any:
    """Plain-JSON copy of a summary (numpy scalars and arrays, tuples, frames, non-string
    keys), so writing it cannot fail after the test access is journalled."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict("records"))
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sweep_primary(project: Any, sweep_config: str) -> tuple[Path, str, list[float]]:
    """(detector_final inventory path, primary HAND column, tier edges) of the sweep config."""
    from nakkhu.config import load_config
    from nakkhu.hydro.sweep import load_settings

    sweep = load_config(project.paths.resolve(sweep_config)).values
    final_path = project.paths.resolve(sweep["inventories"]["detector_final"]["path"])
    primary_cfg = load_settings(project, sweep_config)[0].primary().cfg_id
    pcol = f"{primary_cfg}__{sweep['primary']['statistic']}"
    return final_path, pcol, [float(e) for e in sweep["tiers_m"]]


def run_validate(project: Any, inventories: Sequence[str], *, n_boot: int = 2000,
                 seed: int = 0, sweep_config: str = "configs/hand/sweep.yaml"
                 ) -> tuple[Any, dict[str, Any]]:
    """``nakkhu flood validate``: refuses until every prerequisite holds, loads and checks the
    inputs that hold no reference or answer (HAND tables, competitors, blocks, change-map
    building sets), then journals one test access per call and writes
    ``artifacts/validation/*`` and ``artifacts/flood/change_map_audit_summary.json``."""
    from nakkhu import hazard
    from nakkhu.artifacts import promote
    from nakkhu.config import ResolvedConfig
    from nakkhu.evaluate.guard import record_test_access, require_frozen
    from nakkhu.runs import atomic_write_json, create_run

    inventories = list(dict.fromkeys(inventories))
    final_path, pcol, edges = _sweep_primary(project, sweep_config)
    problems = prerequisites(project, final_path if "detector_final" in inventories else None,
                             inventories)
    if problems:
        raise RuntimeError("flood validate refused: " + "; ".join(problems))
    guard = require_frozen(project)
    labels = hazard.tier_labels(edges)
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for inv in inventories:
        pred = predictors(project, inv, sweep_config)
        if pcol not in pred.columns:
            raise KeyError(f"primary HAND column {pcol} missing for {inv}")
        cm_path = change_map_path(project, inv)
        if cm_path is None:
            raise FileNotFoundError(f"no change map for {inv}")
        change = pd.read_parquet(cm_path, columns=CHANGE_COLUMNS)
        differ = set(change["building_id"]) ^ set(pred["building_uid"])
        if differ:
            raise ValueError(f"{inv}: change map {cm_path.name} and the HAND table hold "
                             f"different buildings ({len(differ)}); rebuild the older one")
        table = hand_table_path(project, inv)
        assert table is not None
        loaded[inv] = (pred, change, surface_meta(table, inv))
    cfg = ResolvedConfig(values={"command": "flood-validate", "inventories": list(inventories),
                                 "n_boot": n_boot, "seed": seed}, source="cli")
    access = record_test_access(project, f"flood/validate:{','.join(inventories)}",
                                cfg.cfghash, guard["prereg_sha256"], guard["split_sha256"],
                                note="nakkhu flood validate")
    estimates = load_audit_estimates(project)
    items, ft, dmg = audit_answers(project)
    run = create_run(project, "validation", cfg)
    v1 = v1_summary(estimates, sample_inputs(project) or {})
    summary: dict[str, Any] = {"test_access": access, "V1": v1, "V5": STATUS_V5,
                               "V6": STATUS_V6, "inventories": {}}
    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in VALIDATION_ARTIFACTS}
    with run.step("validate"):
        for inv in inventories:
            pred, change, meta = loaded[inv]
            inv_sum, parts = score_inventory(inv, pred, change, estimates, items, ft, dmg,
                                             pcol, labels, edges, n_boot, seed, meta)
            summary["inventories"][inv] = inv_sum
            for key, part in parts.items():
                tables[key] += part
        summary["primary"] = primary_outcome(summary, audited_inventory(project))
        for key, parts in tables.items():
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            frame.to_csv(run.path / f"{key}.csv", index=False)
        summary = jsonable(summary)
        atomic_write_json(run.path / "validation_summary.json", summary)
        atomic_write_json(run.path / "change_map_audit_summary.json", jsonable(v1))
        run.write_summary({"inventories": list(inventories),
                           "test_access_flagged": access["flagged"],
                           "primary": {k: summary["primary"].get(k)
                                       for k in ("inventory", "reference", "status")}})
    for key in VALIDATION_ARTIFACTS:
        promote(run.path / f"{key}.csv", f"validation/{key}.csv", project.paths, replace=True)
    promote(run.path / "validation_summary.json", "validation/validation_summary.json",
            project.paths, replace=True)
    promote(run.path / "change_map_audit_summary.json", "flood/change_map_audit_summary.json",
            project.paths, replace=True)
    return run, summary


# --- competitor re-run after the CRS fix (plans/competitor_auc_bug.md) ------------------------
#
# The GLO-30 elevation and ASF GLO-30 HAND competitors of a validate run were sampled in the
# wrong CRS (constant 0, AUC 0.5). :func:`rerun_competitors` recomputes only the V4 competitor
# table of that run, for the run's own inventories, references and settings, with the fixed
# sampler. It is a journalled test access of its own (it reads the change-map classes and the
# sitting-3 answers), runs at most once per source run, writes into a new run and promotes
# nothing; the source run and its promoted artifacts are never touched. The rows the bug did
# not touch must come out byte-identical to the source run's (reproduction check).

RERUN_REF = "flood/validate-competitors"      # journal run ref: <RERUN_REF>:<source run id>
RERUN_STAGE = "validation-competitors"
RERUN_CSV = "competitors_auc.csv"
RERUN_JSON = "competitors_rerun.json"
RERUN_CORRECTED = ("glo30_elevation", "asf_glo30_hand")
RERUN_KEPT_COLUMNS = ("auc_hand", "n", "n_blocks", "n_boot_valid")  # correct in those rows too
ROW_KEY = ["inventory", "reference", "competitor"]
_UTC = "%Y-%m-%dT%H:%M:%SZ"


def _validate_ref(inventories: Sequence[str]) -> str:
    return f"flood/validate:{','.join(inventories)}"


def source_validate_run(project: Any, source: str) -> tuple[Path, dict[str, Any], list[str]]:
    """``(run dir, config values, problems)`` of the validate run whose competitors are re-run
    (``validation/<run_id>``, complete, made by ``nakkhu flood validate``)."""
    from nakkhu.config import load_yaml
    from nakkhu.runs import CONFIG_FILE, is_complete

    stage, _, run_id = source.strip("/").partition("/")
    path = project.paths.runs / stage / run_id
    if stage != "validation" or not run_id or not path.is_dir():
        return path, {}, [f"{source} is not a validate run (validation/<run_id>)"]
    problems: list[str] = []
    cfg = load_yaml(path / CONFIG_FILE) if (path / CONFIG_FILE).is_file() else {}
    if cfg.get("command") != "flood-validate":
        problems.append(f"{source} was not made by `nakkhu flood validate`")
    if not is_complete(path):
        problems.append(f"{source} is not complete")
    if not (path / RERUN_CSV).is_file():
        problems.append(f"{source}/{RERUN_CSV} missing")
    return path, cfg, problems


def _source_end(path: Path) -> Any:
    import datetime as dt

    from nakkhu.runs import read_journal

    ends = [r["utc"] for r in read_journal(path) if r.get("event") == "end"]
    return dt.datetime.strptime(ends[-1], _UTC).replace(tzinfo=dt.UTC) if ends else None


def rerun_inputs(project: Any, inventories: Sequence[str],
                 sweep_config: str = "configs/hand/sweep.yaml") -> dict[str, Path]:
    """Every file the competitor re-run reads (label -> path; missing files are left out)."""
    from nakkhu.audit import core as audit_core
    from nakkhu.config import load_config

    files: dict[str, Path | None] = {
        "sweep_config": project.paths.resolve(sweep_config),
        "glo30_elevation": project.paths.data_root / ELEVATION,
        "asf_glo30_hand": project.paths.data_root / ASF_HAND,
        "channel_lines": project.paths.resolve(CHANNEL_LINES),
        "blocks": project.paths.artifacts / BLOCKS,
        "flood_trace_estimates": project.paths.artifacts / ESTIMATES,
        "flood_trace_items": audit_core.items_path(project, "flood_trace"),
        "flood_trace_responses": audit_core.responses_path(project, "flood_trace")}
    sweep_path = project.paths.resolve(sweep_config)
    sweep = load_config(sweep_path).values if sweep_path.is_file() else {}
    for inv in inventories:
        table = hand_table_path(project, inv)
        files[f"hand_table:{inv}"] = table
        files[f"count_stability:{inv}"] = None if table is None else \
            table.parent / "count_stability.csv"
        files[f"change_map:{inv}"] = change_map_path(project, inv)
        spec = (sweep.get("inventories") or {}).get(inv) or {}
        if spec.get("path"):
            files[f"inventory:{inv}"] = project.paths.resolve(spec["path"])
    return {k: Path(v) for k, v in files.items() if v is not None and Path(v).is_file()}


def rerun_problems(project: Any, source: str, reason: str,
                   sweep_config: str = "configs/hand/sweep.yaml"
                   ) -> tuple[list[str], dict[str, Any]]:
    """Why the competitor re-run of ``source`` may not run (empty = ready), and its context:
    the source config, the source's journal entry (1-based number) and the input files."""
    import datetime as dt

    from nakkhu.evaluate.guard import read_test_journal, verify
    from nakkhu.uncertainty.run import hold_state

    path, cfg, problems = source_validate_run(project, source)
    ctx: dict[str, Any] = {"source_run": source, "source_config": cfg}
    if not reason.strip():
        problems.append("a re-run needs --reason (journalled with the access)")
    state = verify(project)
    problems += [f"prereg: {p}" for p in state["problems"]]
    held, why = hold_state(project)
    if held:
        problems.append(f"hold: {why}")
    if not cfg:
        return problems, ctx
    inventories = list(cfg.get("inventories") or [])
    ctx["inventories"] = inventories
    journal = read_test_journal(project)
    suffix = path.name.rsplit("-", 1)[-1]
    hits = [i for i, e in enumerate(journal, start=1)
            if e.get("run") == _validate_ref(inventories) and e.get("eval_cfghash") == suffix]
    if len(hits) != 1:
        problems.append(f"expected one test-journal entry of {source} "
                        f"({_validate_ref(inventories)}, cfghash {suffix}), found {len(hits)}")
    else:
        entry = journal[hits[0] - 1]
        ctx["source_journal_entry"] = hits[0]
        for key in ("prereg_sha256", "split_sha256"):
            if state.get(key) != entry.get(key):
                problems.append(f"live {key} {str(state.get(key))[:12]} != the source run's "
                                f"{str(entry.get(key))[:12]} (settings changed since)")
    ref = f"{RERUN_REF}:{path.name}"
    ctx["journal_ref"] = ref
    done = [e for e in journal if e.get("run") == ref]
    if done:
        problems.append(f"the competitors of {source} were already re-run once "
                        f"({done[0].get('utc')}); refused")
    final_path = _sweep_primary(project, sweep_config)[0]
    problems += prerequisites(project, final_path if "detector_final" in inventories else None,
                              inventories)
    files = rerun_inputs(project, inventories, sweep_config)
    ctx["inputs"] = files
    end = _source_end(path)
    if end is not None:
        late = [k for k, f in files.items()
                if dt.datetime.fromtimestamp(f.stat().st_mtime, dt.UTC)
                > end + dt.timedelta(seconds=1)]
        if late:
            problems.append(f"input(s) modified after {source} ended ({end:{_UTC}}): "
                            + ", ".join(late))
    return problems, ctx


def compare_competitor_tables(source_csv: Path, new_csv: Path) -> dict[str, Any]:
    """Reproduction check: every row of a competitor the bug did not touch must be the same CSV
    line in both files; the corrected rows must keep ``RERUN_KEPT_COLUMNS``; same header and
    the same (inventory, reference, competitor) rows in the same order."""
    from nakkhu.evaluate.guard import sha256_file

    old_lines = source_csv.read_text(encoding="utf-8").splitlines()
    new_lines = new_csv.read_text(encoding="utf-8").splitlines()
    old, new = pd.read_csv(source_csv), pd.read_csv(new_csv)
    out: dict[str, Any] = {"source_sha256": sha256_file(source_csv), "rows": len(new),
                           "header_equal": old_lines[:1] == new_lines[:1]}
    out["keys_equal"] = (len(old) == len(new)
                         and old[ROW_KEY].astype(str).equals(new[ROW_KEY].astype(str)))
    if not (out["header_equal"] and out["keys_equal"]):
        out["reproduced"] = False
        return out
    fixed = new["competitor"].isin(RERUN_CORRECTED).to_numpy()
    differ = [" / ".join(map(str, new.loc[i, ROW_KEY])) for i in np.flatnonzero(~fixed)
              if old_lines[i + 1] != new_lines[i + 1]]
    kept_differ = []
    for i in np.flatnonzero(fixed):
        for col in RERUN_KEPT_COLUMNS:
            a, b = old.loc[i, col], new.loc[i, col]
            if not (a == b or (pd.isna(a) and pd.isna(b))):
                kept_differ.append(f"{' / '.join(map(str, new.loc[i, ROW_KEY]))}: {col} "
                                   f"{a} -> {b}")
    out.update({"unchanged_rows": int((~fixed).sum()), "unchanged_rows_differ": differ,
                "corrected_rows": int(fixed.sum()), "kept_columns_differ": kept_differ,
                "reproduced": not differ and not kept_differ})
    return out


def rerun_competitors(project: Any, source: str, *, reason: str, dry_run: bool = False,
                      sweep_config: str = "configs/hand/sweep.yaml") -> dict[str, Any]:
    """``nakkhu flood validate --rerun-competitors validation/<run_id> --reason ...``.

    Refuses unless prereg verify passes, the test hold is off, the source is a complete
    validate run with exactly one test-journal entry whose prereg and split hashes equal the
    live ones, every validate prerequisite holds, no input changed after the source run ended
    and this source was never re-run. ``dry_run`` stops after loading the predictors (no
    reference class, no answer, no journal entry). Otherwise it journals one test access
    (``flood/validate-competitors:<source run id>``, the reason in the note), recomputes the V4
    table of every (inventory, reference) of the source run with its ``n_boot`` and ``seed``,
    writes ``competitors_auc.csv`` and ``competitors_rerun.json`` into a new
    ``validation-competitors`` run and checks the untouched rows against the source."""
    from nakkhu.config import ResolvedConfig
    from nakkhu.evaluate.guard import (
        read_test_journal,
        record_test_access,
        require_frozen,
        sha256_file,
    )
    from nakkhu.runs import atomic_write_json, create_run

    problems, ctx = rerun_problems(project, source, reason, sweep_config)
    cfg_src = ctx.get("source_config") or {}
    settings = {"source_run": source, "inventories": ctx.get("inventories"),
                "n_boot": cfg_src.get("n_boot"), "seed": cfg_src.get("seed"),
                "source_journal_entry": ctx.get("source_journal_entry"),
                "journal_ref": ctx.get("journal_ref"), "reason": reason}
    if problems:
        return {"ready": False, "problems": problems, **settings}
    inventories = list(ctx["inventories"])
    n_boot, seed = int(cfg_src["n_boot"]), int(cfg_src["seed"])
    pcol = _sweep_primary(project, sweep_config)[1]
    loaded: dict[str, tuple[pd.DataFrame, Path]] = {}
    checks: dict[str, Any] = {}
    for inv in inventories:          # no reference class or answer is read here
        pred = predictors(project, inv, sweep_config)
        if pcol not in pred.columns:
            raise KeyError(f"primary HAND column {pcol} missing for {inv}")
        cm_path = change_map_path(project, inv)
        if cm_path is None:
            raise FileNotFoundError(f"no change map for {inv}")
        ids = pd.read_parquet(cm_path, columns=["building_id"])["building_id"]
        differ = set(ids) ^ set(pred["building_uid"])
        if differ:
            raise ValueError(f"{inv}: change map {cm_path.name} and the HAND table hold "
                             f"different buildings ({len(differ)})")
        loaded[inv] = (pred, cm_path)
        checks[inv] = {"buildings": len(pred), **{
            col: {"finite": int(np.isfinite(pred[col]).sum()),
                  "min": float(np.nanmin(pred[col])), "max": float(np.nanmax(pred[col]))}
            for col in ("elev_glo30_m", "asf_hand_m")}}
    if dry_run:
        return {"ready": True, "problems": [], "dry_run": True, **settings,
                "predictors": checks}
    guard = require_frozen(project)
    inputs = {k: sha256_file(f) for k, f in ctx["inputs"].items()}
    cfg = ResolvedConfig(values={"command": "flood-validate-competitors", "source_run": source,
                                 "inventories": inventories, "n_boot": n_boot, "seed": seed,
                                 "reason": reason}, source="cli")
    run = create_run(project, RERUN_STAGE, cfg)
    access = record_test_access(project, ctx["journal_ref"], cfg.cfghash,
                                guard["prereg_sha256"], guard["split_sha256"],
                                note=f"nakkhu flood validate --rerun-competitors {source} "
                                     f"(run {run.ref}): {reason}")
    entry_no = len(read_test_journal(project))
    with run.step("competitors"):
        estimates = load_audit_estimates(project)
        items, ft, _ = audit_answers(project)
        parts = []
        for inv in inventories:
            pred, cm_path = loaded[inv]
            change = pd.read_parquet(cm_path, columns=CHANGE_COLUMNS)
            refs, _, _, _ = inventory_references(pred, change, estimates, items, ft)
            for ref, (sub, y, w) in refs.items():
                sub, y, w, hand = _primary_rows(sub, y, w, pcol)
                frame, _, _ = competitor_rows(sub, y, w, -hand, sub["block_id"].to_numpy(),
                                              n_boot, seed)
                parts.append(frame.assign(inventory=inv, reference=ref))
        table = pd.concat(parts, ignore_index=True)
        table.to_csv(run.path / RERUN_CSV, index=False)
        check = compare_competitor_tables(project.paths.runs / source / RERUN_CSV,
                                          run.path / RERUN_CSV)
        result = jsonable({**settings, "run": run.ref, "test_access": access,
                           "journal_entry": entry_no, "corrected": list(RERUN_CORRECTED),
                           "reproduction": check, "predictors": checks, "inputs": inputs,
                           "sampling": "nearest cell at the building centroid, points "
                                       "reprojected to the raster CRS (nakkhu 45bea9a)"})
        atomic_write_json(run.path / RERUN_JSON, result)
        run.write_summary({k: result[k] for k in ("source_run", "journal_entry", "journal_ref",
                                                  "reason", "corrected")}
                          | {"reproduced": check.get("reproduced")})
    return {"ready": True, **result}
