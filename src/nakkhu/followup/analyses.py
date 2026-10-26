"""The Phase-1 analyses of prereg_v9_a3 (plan 014), as pure functions of the prep table rows and
their references. They never open a file: :mod:`nakkhu.followup.run` loads the labels after
the journal entry and passes :class:`Ref` objects in.

Orientation (``common.orientation``): every HAND, distance and elevation predictor enters as
``score = -value``; model outputs are probabilities. Rows are the v9 primary rows of each
reference (finite hand_v9); a predictor's NaN rows are counted and the pair is scored on the
rows where both scores are finite (``common.rows``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nakkhu import hazard
from nakkhu.evaluate.ci import wilson
from nakkhu.flood import validate as V
from nakkhu.followup import stats as ST
from nakkhu.followup.spec import tlabel

DIST = "distance"
HAND_V9 = "hand_v9"
MR_D8 = "hand_mr_d8__glo30"


@dataclass
class Ref:
    """One inventory x reference: its rows of the prep table, labels and design weights."""

    inventory: str
    reference: str
    frame: pd.DataFrame
    y: np.ndarray
    w: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks(self) -> np.ndarray:
        return self.frame["block_id"].astype(str).to_numpy()

    def tag(self) -> dict[str, Any]:
        return {"inventory": self.inventory, "reference": self.reference,
                "weighted": self.w is not None}

    def subset(self, sel: np.ndarray, name: str) -> Ref:
        sel = np.asarray(sel, dtype=bool)
        return Ref(self.inventory, f"{self.reference}", self.frame.loc[sel].reset_index(
            drop=True), np.asarray(self.y)[sel], None if self.w is None
            else np.asarray(self.w)[sel], {**self.extra, "subset": name})


@dataclass
class Ctx:
    n_boot: int
    seed: int
    min_cell: int
    edges: list[float]
    level: float = V.LEVEL


def score(frame: pd.DataFrame, column: str) -> np.ndarray:
    """-value of a predictor column (lower = more exposed)."""
    return -frame[column].to_numpy(dtype=np.float64)


def compare(ref: Ref, cand: str, comp: str, ctx: Ctx, *, cand_score: np.ndarray | None = None,
            comp_score: np.ndarray | None = None, **tags: Any) -> dict[str, Any]:
    s1 = score(ref.frame, cand) if cand_score is None else cand_score
    s2 = score(ref.frame, comp) if comp_score is None else comp_score
    row = ST.pair(ref.y, s1, s2, ref.blocks, ref.w, ctx.n_boot, ctx.seed, ctx.min_cell)
    return {**ref.tag(), "subset": ref.extra.get("subset", "all"), **tags,
            "candidate": cand, "comparator": comp, **row}


def _missing(ref: Ref, columns: Sequence[str]) -> list[str]:
    return [c for c in columns if c not in ref.frame.columns]


# --- P1-1 -------------------------------------------------------------------------------------


def p1_1(refs: Sequence[Ref], spec: dict[str, Any], ctx: Ctx
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = spec["tests"]["P1-1"]
    rows, tiers = [], []
    for ref in refs:
        if ref.reference not in t["references"]:
            continue
        for pred in t["predictors"]:
            for dem in t["dems"]:
                col = f"{pred}__{dem}"
                for comp in (DIST, HAND_V9):
                    rows.append(compare(ref, col, comp, ctx, predictor=pred, dem=dem))
        labels = hazard.tier_labels(ctx.edges)
        v = ref.frame[MR_D8].to_numpy(dtype=float)
        ok = np.isfinite(v)
        sub = ref.subset(ok, "finite hand_mr_d8")
        ti = hazard.tier_index(v[ok], ctx.edges)
        hits = V.tier_hit_rates(ti, sub.y, labels, sub.w).assign(kind="hit_rate")
        try:
            ors = V.tier_odds_ratios(ti, sub.y.astype(float), sub.blocks, labels, w=sub.w)
            mono = V.monotone(ors)
            ors = ors.assign(kind="odds_ratio", monotone=mono)
        except np.linalg.LinAlgError as error:
            ors = pd.DataFrame([{"kind": "odds_ratio", "error": str(error)}])
        for part in (hits, ors):
            tiers.append(part.assign(**ref.tag(), predictor=MR_D8, nan_path_count=int((~ok).sum()),
                                     n_rows=len(v)))
    return pd.DataFrame(rows), pd.concat(tiers, ignore_index=True) if tiers else pd.DataFrame()


# --- P1-2 -------------------------------------------------------------------------------------


def p1_2(refs: Sequence[Ref], spec: dict[str, Any], ctx: Ctx) -> pd.DataFrame:
    t = spec["tests"]["P1-2"]
    dem = t["dem"]
    rows = []
    for ref in refs:
        for thr in [float(v) for v in t["thresholds_km2"]]:
            lab = tlabel(thr)
            hand, dist_s = f"hand_any__{dem}__{lab}", f"dist_stream__{dem}__{lab}"
            drains = ref.frame[f"drains_main__{dem}__{lab}"].to_numpy(dtype=float)
            share = float(np.nanmean(drains)) if np.isfinite(drains).any() else np.nan
            common: dict[str, Any] = {"threshold_km2": thr, "share_drains_main": share,
                      "n_drains_main": int(np.nansum(drains)),
                      "n_no_stream": int((~np.isfinite(drains)).sum())}
            for comp in (DIST, dist_s, MR_D8):
                rows.append(compare(ref, hand, comp, ctx, **common, comparison=f"vs {comp}"))
            for value, name in ((1.0, "drains_main True"), (0.0, "drains_main False")):
                sub = ref.subset(drains == value, name)
                rows.append(compare(sub, hand, DIST, ctx, **common,
                                    comparison="vs distance within drains_main"))
    return pd.DataFrame(rows)


# --- P1-3 -------------------------------------------------------------------------------------


def _design(cols: list[np.ndarray], knots: list[np.ndarray]) -> np.ndarray:
    parts = [np.ones((len(cols[0]), 1))]
    for x, k in zip(cols, knots, strict=True):
        parts.append(ST.ns_basis(x, k))
    return np.column_stack(parts)


MODELS = {"M0": (DIST,), "M1": (DIST, HAND_V9), "M1mr": (DIST, MR_D8)}


def _model_inputs(frame: pd.DataFrame, names: Sequence[str]) -> list[np.ndarray]:
    return [np.log1p(frame[n].to_numpy(dtype=float)) if n == DIST
            else frame[n].to_numpy(dtype=float) for n in names]


def lobo_predictions(frame: pd.DataFrame, y: np.ndarray, w: np.ndarray | None,
                     blocks: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, int]:
    """Leave-one-block-out predictions of the unpenalised spline logistic (knots per training
    fold); ``(p, n_failed_folds)``."""
    x = _model_inputs(frame, names)
    p = np.full(len(y), np.nan)
    failed = 0
    for b in np.unique(blocks):
        test = blocks == b
        train = ~test
        knots = [ST.ns_knots(v[train]) for v in x]
        try:
            beta = ST.logistic_fit(_design([v[train] for v in x], knots), y[train],
                                   None if w is None else w[train])
            if not np.all(np.isfinite(beta)):
                raise np.linalg.LinAlgError("non-finite coefficients")
        except np.linalg.LinAlgError:
            failed += 1
            continue
        p[test] = ST.predict(_design([v[test] for v in x], knots), beta)
    return p, failed


def p1_3_joint(ref: Ref, ctx: Ctx) -> list[dict[str, Any]]:
    cols = [DIST, HAND_V9, MR_D8]
    ok = np.all([np.isfinite(ref.frame[c].to_numpy(dtype=float)) for c in cols], axis=0)
    sub = ref.subset(ok, "finite distance, hand_v9, hand_mr_d8")
    base = {**sub.tag(), "n": int(len(sub.y)), "affected": int(sub.y.sum()),
            "n_excluded_nan": int((~ok).sum()), "n_blocks": int(len(set(sub.blocks)))}
    if not ST.scorable(sub.y, ctx.min_cell):
        return [{**base, "row": "below min_cell"}]
    preds, failed = {}, {}
    for name, names in MODELS.items():
        preds[name], failed[name] = lobo_predictions(sub.frame, sub.y, sub.w, sub.blocks, names)
    rows = []
    for name, p in preds.items():
        good = np.isfinite(p)
        ww = None if sub.w is None else sub.w[good]
        rows.append({**base, "row": "model", "model": name, "n_failed_folds": failed[name],
                     "n_oof": int(good.sum()),
                     "oof_auc": V.roc_auc(p[good], sub.y[good], ww),
                     "oof_pr_auc": V.average_precision(p[good], sub.y[good], ww),
                     "oof_log_loss": ST.log_loss(p[good], sub.y[good], ww),
                     "oof_brier": ST.brier(p[good], sub.y[good], ww)})
    for name in ("M1", "M1mr"):
        p1, p0 = preds[name], preds["M0"]
        good = np.isfinite(p1) & np.isfinite(p0)
        y, bl = sub.y[good], sub.blocks[good]
        ww = None if sub.w is None else sub.w[good]
        pr = ST.pair(y, p1[good], p0[good], bl, ww, ctx.n_boot, ctx.seed, ctx.min_cell)

        def dloss(idx: np.ndarray, fn: Any = ST.log_loss, a: np.ndarray = p1[good],
                  b: np.ndarray = p0[good], y: np.ndarray = y,
                  ww: np.ndarray | None = ww) -> float:
            wi = None if ww is None else ww[idx]
            return fn(a[idx], y[idx], wi) - fn(b[idx], y[idx], wi)

        ll = ST.block_bootstrap(dloss, bl, ctx.n_boot, ctx.seed)
        br = ST.block_bootstrap(lambda idx: dloss(idx, ST.brier), bl, ctx.n_boot, ctx.seed)
        all_idx = np.arange(len(y))
        rows.append({**base, "row": "delta", "model": f"{name} - M0", **pr,
                     "delta_log_loss": dloss(all_idx),
                     "delta_log_loss_lo": ST.percentile_ci(ll, ctx.level)[0],
                     "delta_log_loss_hi": ST.percentile_ci(ll, ctx.level)[1],
                     "delta_brier": dloss(all_idx, ST.brier),
                     "delta_brier_lo": ST.percentile_ci(br, ctx.level)[0],
                     "delta_brier_hi": ST.percentile_ci(br, ctx.level)[1]})
    for name in ("M1", "M1mr"):
        x = _model_inputs(sub.frame, MODELS[name])
        knots = [ST.ns_knots(v) for v in x]
        X = _design(x, knots)
        try:
            beta = ST.logistic_fit(X, sub.y, sub.w)
            cov = V.cluster_robust_cov(X, sub.y, beta, sub.blocks, sub.w)
            wald = ST.wald_chi2(beta, cov, range(5, 9))
        except np.linalg.LinAlgError as error:
            wald = {"error": str(error)}
        rows.append({**base, "row": "wald_cr1", "model": name,
                     "terms": f"ns({MODELS[name][1]}, df 4)", **wald,
                     "caveat": f"CR1 with {base['n_blocks']} clusters may be anti-conservative"})
    return rows


def band_of(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Band label of each value for half-open bands [e_i, e_{i+1}) and [e_last, inf)."""
    e = list(edges)
    names = [f"[{a:g},{b:g})" for a, b in zip(e[:-1], e[1:], strict=True)] + [f"[{e[-1]:g},inf)"]
    idx = np.searchsorted(np.asarray(e, dtype=float), np.asarray(values, dtype=float),
                          side="right") - 1
    out = np.array([names[i] if i >= 0 else "below" for i in idx], dtype=object)
    out[~np.isfinite(np.asarray(values, dtype=float))] = "nan"
    return out


def hand_band(values: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    """Mirror bands ``<= c0`` (incl. on drainage), ``(c0, c1]``, ``> c1``."""
    a, b = (float(c) for c in cuts)
    v = np.asarray(values, dtype=float)
    out = np.where(v <= a, f"<={a:g}", np.where(v <= b, f"({a:g},{b:g}]", f">{b:g}"))
    return np.where(np.isfinite(v), out, "nan").astype(object)


def p1_3_bands(ref: Ref, spec: dict[str, Any], ctx: Ctx) -> list[dict[str, Any]]:
    b = spec["tests"]["P1-3"]["bands"]
    rows = []
    dband = band_of(ref.frame[DIST].to_numpy(dtype=float), b["distance_bands_m"])
    names = [n for n in pd.unique(band_of(np.asarray(b["distance_bands_m"], dtype=float),
                                          b["distance_bands_m"]))]
    for name in names:
        sub = ref.subset(dband == name, f"distance band {name}")
        for pred in b["predictors"]:
            col = MR_D8 if pred == "hand_mr_d8" else pred
            rows.append(compare(sub, col, DIST, ctx, part="distance band", band=name))
    scorable = [n for n in names if ST.scorable(ref.y[dband == n], ctx.min_cell)]
    for pred in b["predictors"]:
        col = MR_D8 if pred == "hand_mr_d8" else pred
        h, d = score(ref.frame, col), score(ref.frame, DIST)
        ok = np.isfinite(h) & np.isfinite(d)
        y, bl, bd = ref.y[ok], ref.blocks[ok], dband[ok]
        ww = None if ref.w is None else ref.w[ok]

        def strat(idx: np.ndarray, h: np.ndarray = h[ok], d: np.ndarray = d[ok], y: np.ndarray = y,
                  bd: np.ndarray = bd, ww: np.ndarray | None = ww) -> float:
            wi = None if ww is None else ww[idx]
            return (ST.stratified_auc(h[idx], y[idx], bd[idx], scorable, wi)
                    - ST.stratified_auc(d[idx], y[idx], bd[idx], scorable, wi))

        boot = ST.block_bootstrap(strat, bl, ctx.n_boot, ctx.seed)
        lo, hi = ST.percentile_ci(boot, ctx.level)
        rows.append({**ref.tag(), "part": "stratified", "band": ";".join(scorable),
                     "candidate": col, "comparator": DIST, "n": int(ok.sum()),
                     "affected": int(y.sum()),
                     "auc_candidate": ST.stratified_auc(h[ok], y, bd, scorable, ww),
                     "auc_comparator": ST.stratified_auc(d[ok], y, bd, scorable, ww),
                     "delta_auc": strat(np.arange(len(y))), "boot_lo": lo, "boot_hi": hi,
                     "n_boot_valid": int(boot.size), "scored": bool(scorable)})
    hb = hand_band(ref.frame[HAND_V9].to_numpy(dtype=float), b["hand_bands_m"])
    for name in [n for n in pd.unique(hb) if n != "nan"]:
        sub = ref.subset(hb == name, f"hand_v9 band {name}")
        rows.append(compare(sub, HAND_V9, DIST, ctx, part="mirror (hand_v9 band)", band=name))
    return rows


# --- P1-4 -------------------------------------------------------------------------------------


def p1_4_rules(dem: str, drop_km2: float | None, thresholds: Sequence[float]
               ) -> list[tuple[str, str | None]]:
    """``(rule, column)`` of every P1-4 rule of a DEM (column None if it has no drop)."""
    out: list[tuple[str, str | None]] = [(f"threshold {t:g}", f"hand_any__{dem}__{tlabel(t)}")
                                         for t in thresholds]
    out.append(("drop threshold", None if drop_km2 is None
                else f"hand_any__{dem}__{tlabel(drop_km2)}"))
    out += [("hand_mr_d8", f"hand_mr_d8__{dem}"), ("hand_mr_rem", f"hand_mr_rem__{dem}")]
    return out


def p1_4(refs: Sequence[Ref], dems: Sequence[str], prepared: dict[str, dict[str, Any]],
         thresholds: Sequence[float], ctx: Ctx, surface_types: dict[str, str]
         ) -> pd.DataFrame:
    rows = []
    glo = prepared.get("glo30") or {}
    glo_rules = dict(p1_4_rules("glo30", glo.get("drop_threshold_km2"), thresholds))
    for dem in dems:
        meta = prepared.get(dem)
        if meta is None:
            rows.append({"dem": dem, "status": "not prepared (scored later as P1-4-" + dem
                         + " once its surface is ready)"})
            continue
        flags: dict[str, Any] = {"dem": dem, "surface_type": surface_types.get(dem, ""),
                 "drop_threshold_km2": meta.get("drop_threshold_km2"),
                 "routing_gate_pass": meta.get("routing_gate_pass"),
                 "in_ranges": bool(meta.get("routing_gate_pass")), "status": "scored"}
        for rule, col in p1_4_rules(dem, meta.get("drop_threshold_km2"), thresholds):
            for ref in refs:
                if col is None or col not in ref.frame.columns:
                    rows.append({**ref.tag(), **flags, "rule": rule, "status": "missing column"})
                    continue
                rows.append(compare(ref, col, DIST, ctx, **flags, rule=rule,
                                    comparison="vs distance"))
                glo_col = glo_rules.get(rule)
                if dem != "glo30" and glo_col:
                    rows.append(compare(ref, col, glo_col, ctx, **flags, rule=rule,
                                        comparison="vs glo30 same rule"))
    return pd.DataFrame(rows)


# --- P1-5 -------------------------------------------------------------------------------------


PREDICTORS_5 = (HAND_V9, MR_D8, DIST)


def _singles(y: np.ndarray, frame: pd.DataFrame, w: np.ndarray | None, ctx: Ctx,
             **tags: Any) -> list[dict[str, Any]]:
    rows = []
    for col in PREDICTORS_5:
        s = score(frame, col)
        ok = np.isfinite(s)
        row = {**tags, "predictor": col, "weighted": w is not None, "n": int(ok.sum()),
               "affected": int(np.asarray(y)[ok].sum())}
        if ST.scorable(np.asarray(y)[ok], ctx.min_cell):
            row.update(ST.single(np.asarray(y), s, w))
            row["scored"] = True
        else:
            row["scored"] = False
        rows.append(row)
    return rows


def p1_5(audit: Ref | None, r1_refs: Sequence[Ref], spec: dict[str, Any], ctx: Ctx
         ) -> pd.DataFrame:
    """``audit``: answered (yes/no) sitting-3 items merged with the prep rows, ``y`` = audit
    answer, ``w`` = design weight, ``extra['r1']`` = the R1 class of the same items and
    ``extra['stratum']``. ``r1_refs``: R1 rows with ``extra['score']`` (change-map score)."""
    parts = spec["tests"]["P1-5"]["parts"]
    edges = [0, 50, 100, 200, 300, 500, 1000]
    rows: list[dict[str, Any]] = []
    if audit is not None:
        r1 = np.asarray(audit.extra["r1"]).astype(int)
        for weighted in (False, True):
            w = audit.w if weighted else None
            rows += _singles(audit.y, audit.frame, w, ctx, part="label_source",
                             label="audit answer", inventory=audit.inventory)
            rows += _singles(r1, audit.frame, w, ctx, part="label_source",
                             label="R1 class of the same items", inventory=audit.inventory)
        neg = np.isin(np.asarray(audit.extra["stratum"]), ["negative_near", "negative_far"])
        miss = audit.subset(neg, "change-map negative items")
        scored = _singles(miss.y, miss.frame, None, ctx, part="misses",
                          label="audit yes vs no among change-map negatives",
                          inventory=audit.inventory)
        rows += scored
        if not all(r["scored"] for r in scored):
            band = band_of(miss.frame[DIST].to_numpy(dtype=float), edges)
            for name in pd.unique(band):
                sel = band == name
                k, n = int(miss.y[sel].sum()), int(sel.sum())
                lo, hi = wilson(k, n, ctx.level)
                rows.append({"part": "misses", "label": "audit yes share by distance band",
                             "inventory": audit.inventory, "band": name, "n": n, "affected": k,
                             "share": k / n if n else np.nan, "wilson_lo": lo, "wilson_hi": hi})
    for ref in r1_refs:
        labels = hazard.tier_labels(ctx.edges)
        groups = (("distance band", band_of(ref.frame[DIST].to_numpy(dtype=float), edges)),
                  ("hand_v9 tier", np.asarray(labels, dtype=object)[
                      hazard.tier_index(ref.frame[HAND_V9].to_numpy(dtype=float), ctx.edges)]))
        for kind, band in groups:
            for name in pd.unique(band):
                sel = band == name
                k, n = int(ref.y[sel].sum()), int(sel.sum())
                lo, hi = wilson(k, n, ctx.level)
                rows.append({"part": "gradient", "label": f"R1 share by {kind}",
                             **ref.tag(), "band": name, "n": n, "affected": k,
                             "share": k / n if n else np.nan, "wilson_lo": lo, "wilson_hi": hi})
        cs = np.asarray(ref.extra["score"], dtype=float)
        dband = band_of(ref.frame[DIST].to_numpy(dtype=float), edges)
        for name in ["overall", *pd.unique(dband)]:
            sel = np.ones(len(cs), bool) if name == "overall" else dband == name
            for col in (HAND_V9, DIST):
                v = ref.frame[col].to_numpy(dtype=float)
                ok = sel & np.isfinite(cs) & np.isfinite(v)
                a, b, bl = cs[ok], v[ok], ref.blocks[ok]
                rho = ST.spearman(a, b)
                boot = ST.block_bootstrap(lambda i, a=a, b=b: ST.spearman(a[i], b[i]), bl,
                                          ctx.n_boot, ctx.seed) if ok.sum() >= 3 else np.array([])
                lo, hi = ST.percentile_ci(boot, ctx.level)
                rows.append({"part": "change_score", "label": f"spearman(score, {col})",
                             **ref.tag(), "band": name, "n": int(ok.sum()), "rho": rho,
                             "boot_lo": lo, "boot_hi": hi, "n_boot_valid": int(boot.size)})
    out = pd.DataFrame(rows)
    out.attrs["parts"] = list(parts)
    return out


# --- P1-7 -------------------------------------------------------------------------------------


PREDICTORS_7 = (HAND_V9, MR_D8, DIST, "dist_cell")


def _boot_metric(ref: Ref, a: str, b: str, fn: Any, ctx: Ctx, level: float
                 ) -> dict[str, Any]:
    s1, s2 = score(ref.frame, a), score(ref.frame, b)
    ok = np.isfinite(s1) & np.isfinite(s2)
    y, bl = ref.y[ok], ref.blocks[ok]
    ww = None if ref.w is None else ref.w[ok]
    s1, s2 = s1[ok], s2[ok]

    def d(idx: np.ndarray) -> float:
        wi = None if ww is None else ww[idx]
        return fn(s1[idx], y[idx], wi) - fn(s2[idx], y[idx], wi)

    boot = ST.block_bootstrap(d, bl, ctx.n_boot, ctx.seed)
    lo, hi = ST.percentile_ci(boot, level)
    return {"n": int(ok.sum()), "affected": int(y.sum()), "value_candidate": fn(s1, y, ww),
            "value_comparator": fn(s2, y, ww), "delta": d(np.arange(len(y))), "boot_lo": lo,
            "boot_hi": hi, "level": level, "n_boot_valid": int(boot.size)}


def p1_7(refs: Sequence[Ref], ctx: Ctx, margin: float = 0.03) -> pd.DataFrame:
    rows = []
    for ref in refs:
        if not ST.scorable(ref.y, ctx.min_cell):
            rows.append({**ref.tag(), "metric": "all", "scored": False})
            continue
        for i, a in enumerate(PREDICTORS_7):
            for b in PREDICTORS_7[i + 1:]:
                rows.append({**ref.tag(), "metric": "spauc_fpr0.1", "candidate": a,
                             "comparator": b, **_boot_metric(
                                 ref, a, b, ST.standardised_pauc, ctx, ctx.level)})
                rows.append({**ref.tag(), "metric": "pr_auc", "candidate": a, "comparator": b,
                             **_boot_metric(ref, a, b, V.average_precision, ctx, ctx.level)})
        for a, b in ((HAND_V9, "dist_cell"), ("dist_cell", DIST)):
            rows.append({**compare(ref, a, b, ctx), "metric": "delta_auc (resolution)"})
        tost = _boot_metric(ref, HAND_V9, DIST, V.roc_auc, ctx, 0.90)
        tost["equivalent"] = bool(-margin < tost["boot_lo"] and tost["boot_hi"] < margin)
        rows.append({**ref.tag(), "metric": f"tost_auc_margin_{margin:g}",
                     "candidate": HAND_V9, "comparator": DIST, **tost})
    return pd.DataFrame(rows)


# --- P1-8 / P1-5-ext (external polygons) ------------------------------------------------------


def external_reference(rows_ref: Ref, label: np.ndarray, name: str, rows_name: str,
                       ctx: Ctx, predictors: Sequence[str] = (HAND_V9, MR_D8, DIST)
                       ) -> list[dict[str, Any]]:
    """AUCs and paired delta AUCs (each HAND predictor vs distance) against an external label
    on the given rows."""
    ref = Ref(rows_ref.inventory, name, rows_ref.frame, np.asarray(label).astype(int),
              rows_ref.w, {"subset": rows_name})
    out = []
    for col in predictors:
        if col == DIST:
            continue
        out.append(compare(ref, col, DIST, ctx, label=name, rows=rows_name))
    if MR_D8 in predictors and HAND_V9 in predictors:
        out.append(compare(ref, MR_D8, HAND_V9, ctx, label=name, rows=rows_name))
    return out


def agreement(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict[str, Any]:
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    return {"part": "agreement", "label": f"{name_a} vs {name_b}", "n": int(len(a)),
            "kappa": ST.cohen_kappa(a, b), "n11": int((a & b).sum()), "n10": int((a & ~b).sum()),
            "n01": int((~a & b).sum()), "n00": int((~a & ~b).sum())}


def within_label(ref: Ref, label: np.ndarray, name: str, ctx: Ctx) -> list[dict[str, Any]]:
    """AUCs of the predictors against R1 within label = 1 and within label = 0."""
    rows = []
    for value in (1, 0):
        sub = ref.subset(np.asarray(label).astype(int) == value, f"{name} = {value}")
        rows += _singles(sub.y, sub.frame, sub.w, ctx, part="within", label=f"R1 within "
                         f"{name} = {value}", inventory=ref.inventory)
    return rows


# --- P1-9 / P1-10 ----------------------------------------------------------------------------


def covered(ref: Ref) -> Ref:
    return ref.subset(np.isfinite(ref.frame["hand_team"].to_numpy(dtype=float)),
                      "covered by hand_team")


def p1_9(refs: Sequence[Ref], ctx: Ctx, cuts: dict[str, float]) -> pd.DataFrame:
    rows = []
    for ref in refs:
        cov = covered(ref)
        rows.append({**ref.tag(), "part": "coverage", "n_full": len(ref.y),
                     "affected_full": int(ref.y.sum()),
                     "blocks_full": int(len(set(ref.blocks))), "n": len(cov.y),
                     "affected": int(cov.y.sum()), "n_blocks": int(len(set(cov.blocks)))})
        for comp in (DIST, HAND_V9, MR_D8):
            rows.append({"part": "delta_auc", **compare(cov, "hand_team", comp, ctx)})
        for key, cut in cuts.items():
            mask = cov.frame[f"team_mask_{key}"].to_numpy(dtype=float)
            ok = np.isfinite(mask)
            rows.append({**ref.tag(), "part": "cumulative_skill", "rule": f"team mask {key}",
                         "cut_m": cut, "n": int(ok.sum()),
                         **ST.binary_skill(mask[ok] > 0, cov.y[ok],
                                           None if cov.w is None else cov.w[ok])})
            for col in ("hand_team", HAND_V9, MR_D8):
                v = cov.frame[col].to_numpy(dtype=float)
                ok = np.isfinite(v)
                rows.append({**ref.tag(), "part": "cumulative_skill", "rule": f"{col} < {cut:g}",
                             "cut_m": cut, "n": int(ok.sum()),
                             **ST.binary_skill(v[ok] < cut, cov.y[ok],
                                               None if cov.w is None else cov.w[ok])})
    return pd.DataFrame(rows)


def p1_10(refs: Sequence[Ref], ctx: Ctx, dems: Sequence[str], conditionings: Sequence[str]
          ) -> pd.DataFrame:
    rows = []
    for ref in refs:
        cov = covered(ref)
        for kind in ("acc", "dig"):
            for dem in dems:
                for cnd in conditionings:
                    col = f"hand_team_{kind}__{dem}__{cnd}"
                    comps = [DIST, "hand_team"]
                    if f"hand_mr_d8__{dem}" in cov.frame.columns:
                        comps.append(f"hand_mr_d8__{dem}")
                    for comp in comps:
                        rows.append(compare(cov, col, comp, ctx, member=col, recipe=kind,
                                            dem=dem, conditioning=cnd))
    return pd.DataFrame(rows)
