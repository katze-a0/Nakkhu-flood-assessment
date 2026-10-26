"""Design-based estimators for the audit (plan 006 Step 4; prereg ``references.r_aud``).

- Per-stratum Wilson 95 % intervals (:func:`stratum_table`).
- Horvitz-Thompson (stratified expansion) proportions, ratios and totals with weights
  w = N_h / n_h, where n_h counts the *answered* yes/no items of the stratum: "unsure" /
  "not observable" answers are excluded (treated as missing at random within the stratum) and
  counted separately, never coded as no.
- Stratified bootstrap with the Rao-Wu (1988) rescaling (resample n_h - 1 units with
  replacement, lambda_h = sqrt(1 - f_h)), so take-all strata add no variance; percentile
  intervals.
- Pooled model precision (clusters a run contains) and recall (yes points inside a run's OBB
  with a 1 m tolerance), source completeness, change-map PPV/NPV, and Cohen's kappa on repeats.
- Recall protocols (prereg ``audit_recall``): v2 (primary) = sitting-1 Yes, or sitting-1
  No/Unsure and a near-miss Yes whose click lies within the circle radius; v1 = sitting 1
  only. Both with Unsure bounds (every Unsure coded Yes / coded No), inter-rater agreement on
  the overlap items and sitting-1 self-agreement on its repeats.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nakkhu.evaluate.ci import wilson

LEVEL = 0.95


def answered(items: pd.DataFrame, answers: dict[str, str]) -> pd.DataFrame:
    """Items with their effective first answer (primary slot) as ``y`` (1 yes, 0 no) and the
    design weight recomputed on answered items; unsure/missing rows are dropped."""
    frame = items.copy()
    frame["answer"] = frame["item_id"].map(answers)
    frame = frame.loc[frame["answer"].isin(["yes", "no"])].copy()
    frame["y"] = (frame["answer"] == "yes").astype(float)
    n_ans = frame.groupby("stratum")["y"].transform("size")
    frame["w"] = frame["frame_n"].astype(float) / n_ans
    return frame


def stratum_table(items: pd.DataFrame, answers: dict[str, str]) -> pd.DataFrame:
    rows = []
    a = items.assign(answer=items["item_id"].map(answers))
    for stratum, g in a.groupby("stratum"):
        yes, no = int((g["answer"] == "yes").sum()), int((g["answer"] == "no").sum())
        uns = int((g["answer"] == "unsure").sum())
        lo, hi = wilson(yes, yes + no, LEVEL)
        rows.append({"stratum": stratum, "frame_n": int(g["frame_n"].iloc[0]),
                     "n_items": len(g), "yes": yes, "no": no, "unsure": uns,
                     "unanswered": int(g["answer"].isna().sum()),
                     "p_yes": yes / (yes + no) if yes + no else float("nan"),
                     "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def ratio(w: np.ndarray, num: np.ndarray, den: np.ndarray) -> float:
    d = float(np.sum(w * den))
    return float(np.sum(w * num)) / d if d > 0 else float("nan")


def rao_wu_weights(strata: np.ndarray, base: np.ndarray, frame_n: np.ndarray, n_boot: int,
                   seed: int) -> np.ndarray:
    """(n_boot, n) bootstrap weights: w* = w (1 - lam + lam n/(n-1) r), r ~ Mult(n-1)."""
    rng = np.random.default_rng(seed)
    out = np.repeat(base[None, :], n_boot, axis=0).astype(float)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        n = len(idx)
        big_n = float(frame_n[idx[0]])
        if n < 2 or big_n <= 0:
            continue
        f = min(1.0, n / big_n)
        lam = np.sqrt(1.0 - f)
        if lam == 0:
            continue
        r = rng.multinomial(n - 1, np.full(n, 1.0 / n), size=n_boot)
        out[:, idx] = base[idx] * (1.0 - lam + lam * n / (n - 1) * r)
    return out


def bootstrap_ratio(frame: pd.DataFrame, num: np.ndarray, den: np.ndarray,
                    n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Design-weighted ratio sum(w num) / sum(w den) with a Rao-Wu stratified bootstrap CI."""
    w = frame["w"].to_numpy(dtype=float)
    point = ratio(w, num, den)
    if len(frame) == 0:
        return {"estimate": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": 0}
    wb = rao_wu_weights(frame["stratum"].to_numpy(), w, frame["frame_n"].to_numpy(dtype=float),
                        n_boot, seed)
    dens = wb @ den
    reps = np.where(dens > 0, (wb @ num) / np.where(dens > 0, dens, 1.0), np.nan)
    reps = reps[np.isfinite(reps)]
    lo, hi = (np.quantile(reps, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2]) if len(reps)
              else (np.nan, np.nan))
    return {"estimate": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(np.sum(den > 0))}


def proportion(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    y = frame["y"].to_numpy(dtype=float)
    return bootstrap_ratio(frame, y, np.ones_like(y), n_boot, seed)


def total(frame: pd.DataFrame, values: np.ndarray, n_boot: int = 2000,
          seed: int = 0) -> dict[str, float]:
    w = frame["w"].to_numpy(dtype=float)
    wb = rao_wu_weights(frame["stratum"].to_numpy(), w, frame["frame_n"].to_numpy(dtype=float),
                        n_boot, seed)
    reps = wb @ values
    lo, hi = np.quantile(reps, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return {"estimate": float(np.sum(w * values)), "ci_lo": float(lo), "ci_hi": float(hi)}


def cohen_kappa(a: list[str], b: list[str]) -> dict[str, float]:
    """Cohen's kappa over the categories present; raw agreement too."""
    if not a:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b, strict=True):
        table[idx[x], idx[y]] += 1
    n = table.sum()
    po = np.trace(table) / n
    pe = float((table.sum(0) * table.sum(1)).sum() / n / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"n": int(n), "agreement": float(po), "kappa": float(kappa)}


# --- task-level estimates ---------------------------------------------------------------------


def recall_estimates(items: pd.DataFrame, answers: dict[str, str], sources: list[str],
                     detected: dict[str, set[str]] | None = None,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Source completeness, source point precision and model recall from recall_points.

    ``detected[run]`` is the set of item ids whose point lies inside a detection OBB of the run
    (1 m tolerance). The ``outside`` stratum (points outside every footprint) is an area sample:
    it is reported as a share of yes points (Wilson) and never mixed into unit estimates.
    """
    units = items.loc[items["stratum"] != "outside"]
    outside = items.loc[items["stratum"] == "outside"]
    frame = answered(units, answers)
    yes = frame["y"].to_numpy(dtype=float)
    out: dict[str, Any] = {"roof_share_of_union_points": proportion(frame, n_boot, seed),
                           "roof_units_total": total(frame, yes, n_boot, seed),
                           "frame_units": int(units.drop_duplicates("stratum")["frame_n"].sum())}
    out["completeness"] = {}
    out["point_precision"] = {}
    for s in [*sources, "google_t90"]:
        col = f"in_{s}"
        if col not in frame:
            continue
        member = frame[col].to_numpy(dtype=float)
        out["completeness"][s] = bootstrap_ratio(frame, yes * member, yes, n_boot, seed)
        out["point_precision"][s] = bootstrap_ratio(frame, yes * member, member, n_boot, seed)
    o = outside.assign(answer=outside["item_id"].map(answers))
    k = int((o["answer"] == "yes").sum())
    n = int(o["answer"].isin(["yes", "no"]).sum())
    out["outside_yes_share"] = {"yes": k, "n": n, "estimate": k / n if n else float("nan"),
                                "wilson": wilson(k, n, LEVEL)}
    out["strata"] = stratum_table(items, answers).to_dict("records")
    if detected:
        out["recall"] = {}
        out["recall_strata"] = {}
        for run, ids in detected.items():
            d = frame["item_id"].isin(ids).to_numpy(dtype=float)
            out["recall"][run] = bootstrap_ratio(frame, yes * d, yes, n_boot, seed)
            out["recall_strata"][run] = recall_stratum_table(frame, ids)
    return out


def recall_stratum_table(frame: pd.DataFrame, detected_ids: set[str]) -> list[dict[str, Any]]:
    """Per stratum: share of the answered yes units a run detects, with a Wilson interval."""
    rows = []
    for stratum, g in frame.groupby("stratum"):
        yes = g.loc[g["y"] == 1.0]
        k, n = int(yes["item_id"].isin(detected_ids).sum()), len(yes)
        lo, hi = wilson(k, n, LEVEL)
        rows.append({"stratum": stratum, "yes_units": n, "detected": k,
                     "recall": k / n if n else float("nan"), "wilson_lo": lo, "wilson_hi": hi})
    return rows


def precision_estimates(items: pd.DataFrame, answers: dict[str, str],
                        members: pd.DataFrame, n_boot: int = 2000,
                        seed: int = 0) -> dict[str, Any]:
    """Per run: design-weighted share of yes among the audited clusters the run contains."""
    frame = answered(items, answers)
    out: dict[str, Any] = {"pooled": proportion(frame, n_boot, seed), "runs": {}}
    yes = frame["y"].to_numpy(dtype=float)
    for run, g in members.groupby("run"):
        has = frame["unit"].isin(set(g["cluster"])).to_numpy(dtype=float)
        out["runs"][run] = {**bootstrap_ratio(frame, yes * has, has, n_boot, seed),
                            "model": str(g["model"].iloc[0])}
    by_model: dict[str, list[float]] = {}
    for est in out["runs"].values():
        by_model.setdefault(est["model"], []).append(est["estimate"])
    out["models"] = {m: {"seed_mean": float(np.mean(v)),
                         "seed_sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                         "runs": len(v)} for m, v in by_model.items()}
    return out


def change_map_estimates(items: pd.DataFrame, answers: dict[str, str],
                         n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """PPV on the positive stratum (Wilson), NPV over the negative strata (HT + bootstrap)."""
    a = items.assign(answer=items["item_id"].map(answers))
    pos = a.loc[a["stratum"] == "positive"]
    k = int((pos["answer"] == "yes").sum())
    n = int(pos["answer"].isin(["yes", "no"]).sum())
    lo, hi = wilson(k, n, LEVEL)
    neg = answered(items.loc[items["stratum"] != "positive"], answers)
    npv = bootstrap_ratio(neg, 1.0 - neg["y"].to_numpy(dtype=float),
                          np.ones(len(neg)), n_boot, seed)
    return {"ppv": {"yes": k, "n": n, "estimate": k / n if n else float("nan"),
                    "wilson_lo": lo, "wilson_hi": hi},
            "npv": npv,
            "drop_change_map": bool(n and lo < 0.5),
            "not_observable": int((a["answer"] == "unsure").sum())}


def polygon_precision_estimates(items: pd.DataFrame, answers: dict[str, str],
                                n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Microsoft polygon precision (plan 013, ``precision_microsoft``): the HT share of Yes
    overall and within each stratum (Rao-Wu CI), Unsure excluded and counted, with the two
    Unsure bounds of the overall share (every Unsure coded Yes / coded No)."""
    frame = answered(items, answers)
    out: dict[str, Any] = {"overall": proportion(frame, n_boot, seed), "by_stratum": {},
                           "unsure": int(sum(answers.get(i) == "unsure"
                                             for i in items["item_id"]))}
    for stratum, g in frame.groupby("stratum"):
        out["by_stratum"][str(stratum)] = proportion(g, n_boot, seed)
    for name, value in (("unsure_as_yes", "yes"), ("unsure_as_no", "no")):
        out[name] = proportion(answered(items, unsure_as(answers, value)), n_boot, seed)
    return out



# --- recall protocols (sitting 1 + near-miss sitting 1b) --------------------------------------


def combine_recall(s1: dict[str, str], nearmiss: dict[str, dict[str, Any]],
                   radius_m: float) -> dict[str, str]:
    """Protocol v2 answer per sitting-1 item.

    ``nearmiss[parent]`` = {"answer", "click_m"} of the near-miss question. Sitting-1 Yes stays
    Yes. A sitting-1 No/Unsure with a near-miss answer becomes Yes if that answer is Yes with a
    click at most ``radius_m`` from the point, No if it is No (or a Yes clicked beyond the
    radius), Unsure if it is Unsure. Items without a near-miss answer (the ``outside`` stratum)
    keep their sitting-1 answer.
    """
    out = dict(s1)
    for item, a in s1.items():
        if a == "yes" or item not in nearmiss:
            continue
        nm = nearmiss[item]
        if nm["answer"] == "yes":
            click = nm.get("click_m")
            out[item] = "yes" if click is not None and float(click) <= radius_m else "no"
        else:
            out[item] = str(nm["answer"])
    return out


def unsure_as(answers: dict[str, str], value: str) -> dict[str, str]:
    return {k: (value if v == "unsure" else v) for k, v in answers.items()}


def recall_with_bounds(items: pd.DataFrame, answers: dict[str, str], sources: list[str],
                       detected: dict[str, set[str]] | None, n_boot: int, seed: int
                       ) -> dict[str, Any]:
    """Estimates with Unsure excluded (reported) plus the two Unsure bounds."""
    counts = {a: int(sum(v == a for v in answers.values())) for a in ("yes", "no", "unsure")}
    return {"answers": counts,
            "unsure_excluded": recall_estimates(items, answers, sources, detected, n_boot, seed),
            "unsure_as_yes": recall_estimates(items, unsure_as(answers, "yes"), sources,
                                              detected, n_boot, seed),
            "unsure_as_no": recall_estimates(items, unsure_as(answers, "no"), sources,
                                             detected, n_boot, seed)}


def agreement(a: list[str], b: list[str]) -> dict[str, Any]:
    """Cohen's kappa and raw agreement over the three answers, and over pairs without Unsure."""
    yes_no = [(x, y) for x, y in zip(a, b, strict=True) if "unsure" not in (x, y)]
    return {"all": cohen_kappa(a, b),
            "yes_no": cohen_kappa([x for x, _ in yes_no], [y for _, y in yes_no])}


# --- multi-rater agreement (remote raters; never part of the primary estimates) -----------------


def _percentile_ci(reps: np.ndarray) -> tuple[float, float]:
    reps = reps[np.isfinite(reps)]
    if not len(reps):
        return float("nan"), float("nan")
    lo, hi = np.quantile(reps, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return float(lo), float(hi)


def kappa_ci(a: list[str], b: list[str], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Cohen's kappa with a percentile bootstrap CI over items (resampled with replacement)."""
    out: dict[str, Any] = dict(cohen_kappa(a, b))
    n = len(a)
    if n < 2:
        return {**out, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    x, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    reps = np.array([cohen_kappa(list(x[i]), list(y[i]))["kappa"]
                     for i in rng.integers(0, n, size=(n_boot, n))], dtype=float)
    lo, hi = _percentile_ci(reps)
    return {**out, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot}


def fleiss_counts(rows: list[list[str]], cats: list[str] | None = None) -> np.ndarray:
    """(items, categories) count matrix of a complete ratings table (one row per item)."""
    cats = cats or sorted({c for r in rows for c in r})
    idx = {c: i for i, c in enumerate(cats)}
    out = np.zeros((len(rows), len(cats)))
    for i, row in enumerate(rows):
        for c in row:
            out[i, idx[c]] += 1
    return out


def fleiss_from_counts(counts: np.ndarray) -> float:
    """Fleiss' kappa from an (items, categories) count matrix with equal raters per item."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return float("nan")
    m = counts.sum(axis=1)
    if len(np.unique(m)) != 1 or m[0] < 2:
        raise ValueError("Fleiss' kappa needs the same number (>= 2) of ratings per item")
    k = float(m[0])
    n = counts.shape[0]
    p_j = counts.sum(axis=0) / (n * k)
    p_i = (np.sum(counts * counts, axis=1) - k) / (k * (k - 1))
    p_bar, pe = float(p_i.mean()), float(np.sum(p_j * p_j))
    return (p_bar - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(rows: list[list[str]], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Fleiss' kappa over items rated by every rater, with a bootstrap CI over items."""
    if not rows:
        return {"n": 0, "raters": 0, "kappa": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_boot": 0}
    counts = fleiss_counts(rows)
    kappa = fleiss_from_counts(counts)
    rng = np.random.default_rng(seed)
    n = len(rows)
    reps = np.array([fleiss_from_counts(counts[i]) for i in rng.integers(0, n, size=(n_boot, n))],
                    dtype=float) if n > 1 else np.array([])
    lo, hi = _percentile_ci(reps)
    return {"n": n, "raters": len(rows[0]), "kappa": float(kappa), "ci_lo": lo, "ci_hi": hi,
            "n_boot": n_boot if n > 1 else 0}


def multi_rater_agreement(answers: dict[str, dict[str, str]], n_boot: int = 2000,
                          seed: int = 0) -> dict[str, Any]:
    """Pairwise Cohen's kappa (with CIs) on the items both raters answered, over the three
    answers and over pairs without Unsure; Fleiss' kappa (with CIs) on the items every rater
    answered when there are three or more raters. Unweighted (sample agreement, not a design-
    based population figure)."""
    names = sorted(answers)
    out: dict[str, Any] = {"raters": names, "pairwise": {}, "fleiss": None}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(answers[a]) & set(answers[b]))
            x = [answers[a][k] for k in common]
            y = [answers[b][k] for k in common]
            yes_no = [(p, q) for p, q in zip(x, y, strict=True) if "unsure" not in (p, q)]
            out["pairwise"][f"{a}|{b}"] = {
                "all": kappa_ci(x, y, n_boot, seed),
                "yes_no": kappa_ci([p for p, _ in yes_no], [q for _, q in yes_no], n_boot, seed)}
    if len(names) >= 3:
        common = sorted(set.intersection(*(set(answers[r]) for r in names)))
        rows = [[answers[r][k] for r in names] for k in common]
        out["fleiss"] = {"all": fleiss_kappa(rows, n_boot, seed),
                         "yes_no": fleiss_kappa([r for r in rows if "unsure" not in r],
                                                n_boot, seed)}
    return out
