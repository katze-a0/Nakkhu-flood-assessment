"""``P2`` of prereg_v9_a3: the one-shot confirmation on the Phase-2 sites (plan 014).

Run once through the receipt flow of :mod:`nakkhu.followup.remote` (reserved on the laptop as
``prereg_v9_a3:P2``, executed remotely). The rules are those of a3 ``phase2`` as frozen in
prereg_v9_a4; the inputs are named in ``configs/followup/phase2/p2_inputs.yaml`` (written after
the site change maps and the P2-audit exist; every file there is pinned by sha256):

1. per site, the reference R1 = change-map class on observable buildings (``flood.validate.
   r1_reference``) and the audit drop rule (PPV Wilson lower bound < 0.5) of the declared
   primary rater; a dropped site leaves the pool;
2. the size check BEFORE any predictor is read: blocks with >= 1 observable building and R1
   affected count over the pooled kept sites, per hypothesis on its rows (H2a: rows where the
   frozen HAND rule is finite; H2b: rows where every M-C feature is finite - prereg_v9_a4
   rows_clarification), the MDE from the block count, and the mode (confirmatory: >= 26
   blocks, >= 150 affected, a kept change map; descriptive otherwise);
3. H2a (frozen HAND rule vs distance) and H2b (frozen M-C vs distance), distance always on
   exactly the rows of the hypothesis: delta AUC, two-sided 95 % percentile CI and one-sided
   97.5 % lower bound of the paired block bootstrap stratified by site (2000 resamples of whole
   blocks within each site, seed 0), p = (1 + #{delta* <= 0}) / (2000 + 1) (a resample with an
   undefined delta counts as delta* <= 0), DeLong;
4. Holm over {H2a, H2b} at one-sided alpha 0.05 (smaller p <= 0.025, then the other <= 0.05); a
   missing or descriptive hypothesis counts as p = 1 and has no decision;
5. secondary rows (no decision): hand_v9 vs distance (C3 replication), M-C vs frozen HAND, per
   site rows, tier hit rates and ORs of the frozen HAND rule, UNOSAT R-UN / R-UN-pts, and the
   negative-control river.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.flood import validate as V
from nakkhu.followup import spec as S

INPUTS = "configs/followup/phase2/p2_inputs.yaml"
OUTPUTS = ("p2_confirm.json", "p2_rows.csv", "p2_tiers.csv", "p2_negative_control.csv")
HAND = "hand_mr_d8__glo30"
HAND_V9 = "hand_v9"
DIST = "distance"
R1_COL = "r1_affected"          # the R1 label column of the pooled rows (y is a coordinate)
ALPHA = 0.05
MIN_BLOCKS = 26
MIN_AFFECTED = 150
MDE_26 = 0.048                  # a3 power_planning: one-sided 0.025, power 0.8, at 26 blocks


class ConfirmError(S.FollowupError):
    """A P2 input is missing, altered or inconsistent."""


@dataclass
class Site:
    """One site: its predictor table (building order), change map and audit estimate."""

    site: str
    role: str                                   # test | negative_control
    table: pd.DataFrame
    change: pd.DataFrame | None = None
    estimate: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --- statistics ---------------------------------------------------------------------------------


def stratified_deltas(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, blocks: np.ndarray,
                      strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """AUC(s1) - AUC(s2) over resamples of whole blocks drawn with replacement within each
    stratum (site), the number of blocks per stratum kept; NaN where undefined."""
    y = np.asarray(y).astype(bool)
    blocks, strata = np.asarray(blocks, dtype=object), np.asarray(strata, dtype=object)
    groups = []
    for s in sorted(set(strata)):
        ids = sorted(set(blocks[strata == s]))
        groups.append([np.flatnonzero((strata == s) & (blocks == b)) for b in ids])
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = np.concatenate([g[k] for g in groups for k in rng.integers(0, len(g), len(g))])
        out[i] = V.roc_auc(s1[idx], y[idx]) - V.roc_auc(s2[idx], y[idx])
    return out


def one_sided_p(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    return float((1 + int(np.sum(~(d > 0)))) / (len(d) + 1))


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: the smaller p at alpha / m, the next at alpha / (m - 1), ..."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    out = {k: False for k in pvalues}
    m = len(order)
    for i, k in enumerate(order):
        if pvalues[k] <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def mde(n_blocks: int) -> float:
    return float(MDE_26 * np.sqrt(MIN_BLOCKS / n_blocks)) if n_blocks else float("nan")


def size_check(blocks: np.ndarray, y: np.ndarray, kept: bool) -> dict[str, Any]:
    n_blocks, affected = int(len(set(np.asarray(blocks)))), int(np.sum(y))
    confirmatory = bool(kept and n_blocks >= MIN_BLOCKS and affected >= MIN_AFFECTED)
    return {"n": int(len(y)), "n_blocks": n_blocks, "affected": affected,
            "change_map_kept": bool(kept), "mde": mde(n_blocks),
            "mode": "confirmatory" if confirmatory else "descriptive"}


def compare(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, blocks: np.ndarray,
            strata: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    d = stratified_deltas(y, s1, s2, blocks, strata, n_boot, seed)
    fin = d[np.isfinite(d)]
    lo, hi = (np.quantile(fin, [0.025, 0.975]) if fin.size else (np.nan, np.nan))
    dl = V.delong(y, s1, s2)
    return {"auc_candidate": V.roc_auc(s1, y), "auc_distance_or_comparator": V.roc_auc(s2, y),
            "pr_auc_candidate": V.average_precision(s1, y),
            "pr_auc_comparator": V.average_precision(s2, y),
            "delta_auc": V.roc_auc(s1, y) - V.roc_auc(s2, y), "ci95_lo": float(lo),
            "ci95_hi": float(hi), "lb_one_sided_97_5": float(lo), "p_one_sided": one_sided_p(d),
            "n_boot": int(n_boot), "n_boot_valid": int(fin.size), "delong_se": dl.get("se"),
            "delong_p_two_sided": dl.get("p")}


# --- pooled rows ----------------------------------------------------------------------------------


def site_rows(site: Site) -> pd.DataFrame:
    """R1 rows of one site (observable buildings), joined to its predictor table."""
    if site.change is None:
        raise ConfirmError(f"site {site.site}: no change map")
    est = site.estimate or {"ppv": {"wilson_lo": float("nan")}}
    r1 = V.r1_reference(site.change, est)
    lab = pd.DataFrame({"building_uid": r1["building_id"], R1_COL: r1["y"].astype(int)})
    rows = site.table.merge(lab, on="building_uid", how="inner", validate="one_to_one")
    if len(rows) != len(lab):
        raise ConfirmError(f"site {site.site}: change map and predictor table differ")
    rows["site"] = site.site
    rows["pool_block"] = site.site + ":" + rows["block_id"].astype(str)
    site.extra["r1_usable"] = bool(r1["usable"])
    site.extra["ppv"] = r1["ppv"]
    site.extra["npv"] = r1["npv"]
    return rows


def predictions(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    from nakkhu.followup import explore as E

    return E.predict(model, frame)


def score_sites(sites: list[Site], model: dict[str, Any], *, n_boot: int = 2000, seed: int = 0,
                min_cell: int = 10, edges: list[float] | None = None,
                unosat: dict[str, pd.DataFrame] | None = None
                ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """The P2 analysis (pure: inputs in memory). Returns the summary and the tables."""
    from nakkhu import hazard

    edges = list(edges or [0, 3, 6, 9, 12])
    tests = [s for s in sites if s.role == "test"]
    controls = [s for s in sites if s.role != "test"]
    per_site: dict[str, Any] = {}
    pooled = []
    for s in tests:
        r = site_rows(s)
        per_site[s.site] = {"r1_rows": int(len(r)), "affected": int(r[R1_COL].sum()),
                            "kept": s.extra["r1_usable"], "ppv": s.extra["ppv"],
                            "npv": s.extra["npv"],
                            "reason": "" if s.extra["r1_usable"] else
                            "audit drop rule (PPV Wilson lower bound < 0.5)"}
        if s.extra["r1_usable"]:
            pooled.append(r)
    kept = bool(pooled)
    rows = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame(
        columns=[R1_COL, "site", "pool_block"])
    y = rows[R1_COL].to_numpy(dtype=int)
    # size check first (no predictor value is used: only which rows are finite)
    hand_ok = np.isfinite(rows[HAND].to_numpy(dtype=float)) if len(rows) else np.zeros(0, bool)
    p_ml = predictions(rows, model) if len(rows) else np.zeros(0)
    ml_ok = np.isfinite(p_ml)
    blocks = rows["pool_block"].to_numpy(dtype=object)
    strata = rows["site"].to_numpy(dtype=object)
    checks = {"pooled_all_rows": size_check(blocks, y, kept),
              "H2a": size_check(blocks[hand_ok], y[hand_ok], kept),
              "H2b": size_check(blocks[ml_ok], y[ml_ok], kept)}
    nan_share = {h: {"n_nan": int((~ok).sum()), "share_nan": float((~ok).mean()) if len(ok)
                     else float("nan")} for h, ok in (("H2a", hand_ok), ("H2b", ml_ok))}
    for s in per_site:
        sel = strata == s
        per_site[s]["nan_H2a"] = int((sel & ~hand_ok).sum())
        per_site[s]["nan_H2b"] = int((sel & ~ml_ok).sum())
        per_site[s]["share_nan_H2a"] = float((sel & ~hand_ok).sum() / max(1, sel.sum()))
        per_site[s]["share_nan_H2b"] = float((sel & ~ml_ok).sum() / max(1, sel.sum()))
    dist_score = -rows[DIST].to_numpy(dtype=float) if len(rows) else np.zeros(0)
    hand_score = -rows[HAND].to_numpy(dtype=float) if len(rows) else np.zeros(0)
    table: list[dict[str, Any]] = []
    hyp: dict[str, Any] = {}
    for name, ok, cand, label in (("H2a", hand_ok, hand_score, "frozen HAND rule"),
                                  ("H2b", ml_ok, p_ml, "frozen M-C")):
        row: dict[str, Any] = {"part": "hypothesis", "hypothesis": name, "candidate": label,
                               "comparator": "distance (same rows)", "rows": "pooled",
                               **{f"size_{k}": v for k, v in checks[name].items()}}
        yy = y[ok]
        if len(yy) and yy.min() != yy.max():
            row.update(compare(yy, cand[ok], dist_score[ok], blocks[ok], strata[ok], n_boot,
                               seed))
        else:
            row.update({"p_one_sided": 1.0, "note": "no scorable rows"})
        hyp[name] = row
        table.append(row)
    pvals = {h: (r["p_one_sided"] if r["size_mode"] == "confirmatory" else 1.0)
             for h, r in hyp.items()}
    rejected = holm(pvals)
    decision = {}
    for h, r in hyp.items():
        if r["size_mode"] != "confirmatory":
            decision[h] = {"mode": "descriptive", "decision": None,
                           "label": "underpowered / no valid reference, not confirmatory",
                           "p_one_sided": r.get("p_one_sided")}
        else:
            decision[h] = {"mode": "confirmatory", "holm_p": pvals[h],
                           "decision": "reject H0" if rejected[h] else "do not reject H0"}
    # secondary rows (no decision)
    sec = []
    if len(rows):
        v9 = -rows[HAND_V9].to_numpy(dtype=float)
        ok = np.isfinite(v9)
        if ok.any() and y[ok].min() != y[ok].max():
            sec.append({"part": "secondary", "comparison": "hand_v9 vs distance (C3 replication)",
                        "n": int(ok.sum()), "affected": int(y[ok].sum()),
                        **compare(y[ok], v9[ok], dist_score[ok], blocks[ok], strata[ok],
                                  n_boot, seed)})
        both = hand_ok & ml_ok
        if both.any() and y[both].min() != y[both].max():
            sec.append({"part": "secondary", "comparison": "M-C vs frozen HAND rule",
                        "n": int(both.sum()), "affected": int(y[both].sum()),
                        **compare(y[both], p_ml[both], hand_score[both], blocks[both],
                                  strata[both], n_boot, seed)})
        for s in sorted(set(strata)):
            for name, ok2, cand in (("H2a", hand_ok, hand_score), ("H2b", ml_ok, p_ml)):
                sel = (strata == s) & ok2
                yy = y[sel]
                base = {"part": "per_site", "site": s, "hypothesis": name,
                        "n": int(sel.sum()), "affected": int(yy.sum()),
                        "n_blocks": int(len(set(blocks[sel])))}
                if int(yy.sum()) >= min_cell and int((1 - yy).sum()) >= min_cell:
                    sec.append({**base, **compare(yy, cand[sel], dist_score[sel], blocks[sel],
                                                  strata[sel], n_boot, seed)})
                else:
                    sec.append({**base, "note": f"below min_cell ({min_cell}); counts only"})
    tiers = pd.DataFrame()
    if len(rows) and hand_ok.any():
        labels = hazard.tier_labels(edges)
        ti = hazard.tier_index(rows[HAND].to_numpy(dtype=float)[hand_ok], edges)
        hits = V.tier_hit_rates(ti, y[hand_ok], labels).assign(kind="hit_rate")
        try:
            ors = V.tier_odds_ratios(ti, y[hand_ok].astype(float), blocks[hand_ok], labels)
            ors = ors.assign(kind="odds_ratio", monotone=V.monotone(ors))
        except np.linalg.LinAlgError as error:
            ors = pd.DataFrame([{"kind": "odds_ratio", "error": str(error)}])
        tiers = pd.concat([hits, ors], ignore_index=True).assign(predictor=HAND, rows="pooled R1")
    if unosat:
        sec += unosat_rows(sites, unosat, model, n_boot, seed, min_cell)
    control = negative_control(controls, model, edges, min_cell)
    summary = {"sites": per_site, "pool": [s for s, v in per_site.items() if v["kept"]],
               "size_check": checks, "nan": nan_share, "hypotheses": decision,
               "holm": {"alpha_one_sided": ALPHA, "p_used": pvals, "rejected": rejected},
               "n_boot": n_boot, "seed": seed}
    return summary, {"p2_rows.csv": pd.DataFrame([*table, *sec]), "p2_tiers.csv": tiers,
                     "p2_negative_control.csv": control}


def negative_control(sites: list[Site], model: dict[str, Any], edges: list[float],
                     min_cell: int) -> pd.DataFrame:
    from nakkhu import hazard

    out = []
    for s in sites:
        r = site_rows(s)
        y = r[R1_COL].to_numpy(dtype=int)
        p = predictions(r, model)
        hand = r[HAND].to_numpy(dtype=float)
        ti = hazard.tier_index(hand, edges)
        shares = {f"share_tier_{lab}": float(np.mean(ti == k))
                  for k, lab in enumerate(hazard.tier_labels(edges))}
        row = {"site": s.site, "role": s.role, "r1_rows": int(len(y)), "affected": int(y.sum()),
               "mean_mc_probability": float(np.nanmean(p)) if np.isfinite(p).any() else None,
               "hand_nan": int(np.isnan(hand).sum()), **shares}
        if int(y.sum()) >= min_cell and int((1 - y).sum()) >= min_cell:
            for name, sc in (("hand", -hand), ("mc", p), ("distance", -r[DIST].to_numpy(float))):
                ok = np.isfinite(sc)
                row[f"auc_{name}"] = V.roc_auc(sc[ok], y[ok])
        else:
            row["note"] = f"below min_cell ({min_cell}); counts only"
        out.append(row)
    return pd.DataFrame(out)


def unosat_rows(sites: list[Site], unosat: dict[str, pd.DataFrame], model: dict[str, Any],
                n_boot: int, seed: int, min_cell: int) -> list[dict[str, Any]]:
    """H2a / H2b against the UNOSAT references (no decision): ``unosat[site]`` holds
    building_uid, inside (analysis extent) and the R-UN / R-UN-pts labels."""
    rows = []
    frames = []
    for s in sites:
        if s.role != "test" or s.site not in unosat:
            continue
        u = s.table.merge(unosat[s.site], on="building_uid", how="inner")
        u = u.loc[u["inside"].astype(bool)].copy()
        u["site"] = s.site
        u["pool_block"] = s.site + ":" + u["block_id"].astype(str)
        frames.append(u)
    if not frames:
        return rows
    u = pd.concat(frames, ignore_index=True)
    p = predictions(u, model)
    hand = -u[HAND].to_numpy(dtype=float)
    dist = -u[DIST].to_numpy(dtype=float)
    b, st = u["pool_block"].to_numpy(dtype=object), u["site"].to_numpy(dtype=object)
    for label in ("R-UN", "R-UN-pts"):
        y = u[label].to_numpy(dtype=int)
        for name, cand in (("H2a", hand), ("H2b", p)):
            ok = np.isfinite(cand)
            base = {"part": "unosat", "reference": label, "hypothesis": name,
                    "n": int(ok.sum()), "affected": int(y[ok].sum())}
            if int(y[ok].sum()) >= min_cell and int((1 - y[ok]).sum()) >= min_cell:
                rows.append({**base, **compare(y[ok], cand[ok], dist[ok], b[ok], st[ok], n_boot,
                                               seed)})
            else:
                rows.append({**base, "note": f"below min_cell ({min_cell}); counts only"})
    return rows


# --- inputs (pinned files) -----------------------------------------------------------------------


def load_inputs(project: Any, path: str = INPUTS) -> dict[str, Any]:
    from nakkhu.config import load_yaml

    p = project.paths.resolve(path)
    if not p.is_file():
        raise ConfirmError(f"{path} missing: P2 needs the pinned site change maps and the "
                           "P2-audit estimates first")
    doc = load_yaml(p)
    for key in ("addendum_a4", "model", "sites", "audit_estimates"):
        if key not in doc:
            raise ConfirmError(f"{path}: `{key}` is required")
    return doc


def input_files(project: Any, doc: dict[str, Any]) -> dict[str, str]:
    """relative path -> pinned sha256 of every file P2 reads."""
    out = {doc["model"]["file"]: doc["model"]["sha256"],
           doc["audit_estimates"]["file"]: doc["audit_estimates"]["sha256"]}
    for s in doc["sites"].values():
        for key in ("predictors", "change_map"):
            if key in s:
                out[s[key]["file"]] = s[key]["sha256"]
    for key in ("unosat_zip", "buildings"):
        for s in doc["sites"].values():
            if key in s:
                out[s[key]["file"]] = s[key]["sha256"]
        if key in doc:
            out[doc[key]["file"]] = doc[key]["sha256"]
    return out


def check_inputs(project: Any, doc: dict[str, Any]) -> list[str]:
    from nakkhu.followup.prep import sha256

    problems = []
    for rel, digest in input_files(project, doc).items():
        f = project.paths.resolve(rel)
        if not f.is_file():
            problems.append(f"P2 input {rel} missing")
        elif sha256(f) != digest:
            problems.append(f"P2 input {rel} sha256 differs from {INPUTS}")
    return problems


def prerequisites(project: Any, addendum_id: str) -> list[str]:
    """P2-specific refusals (label-free): a4 frozen and requiring this addendum, P2-audit
    journalled exactly once, P2 not run, every pinned input present and unaltered."""
    from nakkhu.evaluate.addendum import (
        addendum_path,
        frozen_addenda,
        load_addendum,
        prior_accesses,
    )

    try:
        doc = load_inputs(project)
    except ConfirmError as error:
        return [str(error)]
    problems = []
    a4 = str(doc["addendum_a4"])
    if a4 not in frozen_addenda(project):
        problems.append(f"{a4} is not frozen (P2 runs only after the Phase-2 freeze)")
    else:
        spec = load_addendum(addendum_path(project, a4))
        if addendum_id not in (spec.get("requires") or []):
            problems.append(f"{a4} does not require {addendum_id}")
    audits = prior_accesses(project, addendum_id, "P2-audit")
    if len(audits) != 1:
        problems.append(f"{addendum_id}:P2-audit has {len(audits)} journal entries; P2 needs "
                        "exactly one (the audit estimate) first")
    return problems + check_inputs(project, doc)


def load_sites(project: Any, doc: dict[str, Any]) -> tuple[list[Site], dict[str, Any]]:
    model = json.loads(project.paths.resolve(doc["model"]["file"]).read_text(encoding="utf-8"))
    est = json.loads(project.paths.resolve(doc["audit_estimates"]["file"]).read_text(
        encoding="utf-8"))
    sites = []
    for name, s in doc["sites"].items():
        table = pd.read_parquet(project.paths.resolve(s["predictors"]["file"]))
        change = pd.read_parquet(project.paths.resolve(s["change_map"]["file"]),
                                 columns=["building_id", "class", "observable"])
        role = str(s.get("role", "test"))
        e = (est.get("sites") or {}).get(name)
        if role == "test" and e is None:
            raise ConfirmError(f"site {name}: no P2-audit estimate")
        sites.append(Site(name, role, table, change, e))
    return sites, model


def score(project: Any, cfg: dict[str, Any], spec: dict[str, Any], *, n_boot: int | None = None,
          seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the pinned inputs (after the receipt check) and run :func:`score_sites` with the
    receipt's resample count and seed (the a3 values: 2000, 0)."""
    doc = load_inputs(project)
    problems = check_inputs(project, doc)
    if problems:
        raise ConfirmError("; ".join(problems))
    sites, model = load_sites(project, doc)
    boot, seed0 = S.bootstrap_settings(spec)
    boot = boot if n_boot is None else int(n_boot)
    seed = seed0 if seed is None else int(seed)
    summary, tables = score_sites(sites, model, n_boot=boot, seed=seed,
                                  min_cell=S.min_cell(spec), edges=S.tier_edges(spec))
    modes = {h: d["mode"] for h, d in summary["hypotheses"].items()}
    summary["reporting"] = ("confirmatory (prereg_v9_a3 P2)" if "confirmatory" in modes.values()
                            else "descriptive - underpowered / no valid reference, not "
                                 "confirmatory (prereg_v9_a3 P2)")
    summary["inputs"] = input_files(project, doc)
    out: dict[str, Any] = {"p2_confirm.json": summary}
    for name, frame in tables.items():
        out[name] = frame.assign(reporting=summary["reporting"])
    return out, {"modes": modes}


def write_output(path: Path, value: Any) -> None:
    from nakkhu.flood.validate import jsonable

    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text(json.dumps(jsonable(value), indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
