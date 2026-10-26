"""P1-6 of prereg_v9_a3: learned building-level exposure models (plan 014).

Families (YAML ``tests.P1-6.families``): M-C penalised logistic GAM (natural cubic spline df 4
per feature, L2 penalty, lambda grid), M-D ``HistGradientBoostingClassifier`` (early stopping
off, random_state 0, 60 random configurations drawn with seed 0), M-E = M-D with monotone
constraints (non-increasing in every hand and distance feature and in flowdist_main).

CV: outer leave-one-block-out over the Microsoft R1 blocks; inner 5-fold ``GroupKFold`` by
block over the outer-training rows; the configuration with the lowest mean inner log-loss is
refit on all outer-training rows. detector_final rows of a held-out block are predicted by that
fold's model. Rows: every row with all 28 features finite (the NaN count is reported). Grouped
permutation importance: each group's columns permuted jointly within the held-out block, 20
repeats, seed 0, drop in pooled OOF AUC (descriptive).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.flood import validate as V
from nakkhu.followup import stats as ST

# (feature, prep column, transform, group); order = the YAML `features` order.
FEATURES: tuple[tuple[str, str, str, str], ...] = (
    ("hand_v9", "hand_v9", "id", "hand"),
    ("hand_any_0.05", "hand_any__glo30__t0p05", "id", "hand"),
    ("hand_any_0.1", "hand_any__glo30__t0p1", "id", "hand"),
    ("hand_any_0.25", "hand_any__glo30__t0p25", "id", "hand"),
    ("hand_any_0.5", "hand_any__glo30__t0p5", "id", "hand"),
    ("hand_any_1", "hand_any__glo30__t1", "id", "hand"),
    ("hand_any_2", "hand_any__glo30__t2", "id", "hand"),
    ("hand_mr_d8", "hand_mr_d8__glo30", "id", "hand"),
    ("hand_mr_rem", "hand_mr_rem__glo30", "id", "hand"),
    ("hand_mr_d8_fabdem", "hand_mr_d8__fabdem", "id", "hand"),
    ("log1p_distance", "distance", "log1p", "distance"),
    ("log1p_dist_stream_0.2", "dist_stream__glo30__t0p2", "log1p", "distance"),
    ("log1p_dist_stream_2", "dist_stream__glo30__t2", "log1p", "distance"),
    ("flowdist_main", "flowdist_main__glo30", "id", "distance"),
    ("elev_glo30", "elev_glo30", "id", "elevation"),
    ("elev_fabdem", "elev_fabdem", "id", "elevation"),
    ("dsm_excess", "dsm_excess", "id", "elevation"),
    ("slope_deg", "slope_deg", "id", "terrain"),
    ("curv_plan", "curv_plan", "id", "terrain"),
    ("curv_profile", "curv_profile", "id", "terrain"),
    ("tpi_150", "tpi_150", "id", "terrain"),
    ("tpi_500", "tpi_500", "id", "terrain"),
    ("twi", "twi", "id", "terrain"),
    ("log10_area_cell", "log10_area_cell_km2", "id", "flow"),
    ("log10_area_first_drop", "log10_area_first_drop_km2", "id", "flow"),
    ("log10_area_main", "log10_area_main_km2", "id", "flow"),
    ("footprint_area_m2", "area_m2", "id", "building"),
    ("density_100m", "density_100m", "id", "building"),
)
MONOTONE_GROUPS = ("hand", "distance")      # flowdist_main is in the distance group
INNER_FOLDS = 5


def feature_names() -> list[str]:
    return [f for f, *_ in FEATURES]


def groups() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, (_, _, _, g) in enumerate(FEATURES):
        out.setdefault(g, []).append(i)
    return out


def check_spec(spec: dict[str, Any]) -> None:
    """The feature table must be the YAML's: 28 features in its six groups."""
    feats = spec["tests"]["P1-6"]["features"]
    yaml_groups = {g: len(v) for g, v in feats.items() if g != "excluded"}
    code_groups = {g: len(v) for g, v in groups().items()}
    if yaml_groups != code_groups or len(FEATURES) != 28:
        raise ValueError(f"P1-6 feature groups differ: YAML {yaml_groups}, code {code_groups}")


def matrix(frame: pd.DataFrame) -> np.ndarray:
    cols = []
    for _, col, transform, _ in FEATURES:
        v = frame[col].to_numpy(dtype=np.float64)
        cols.append(np.log1p(v) if transform == "log1p" else v)
    return np.column_stack(cols)


def monotone_cst() -> list[int]:
    return [-1 if g in MONOTONE_GROUPS else 0 for *_, g in FEATURES]


def hgb_configs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The 60 M-D configurations: ParameterSampler(grid, 60, random_state=0) on the YAML grid
    (learning_rate x max_iter x max_leaf_nodes x min_samples_leaf x l2_regularization x
    max_features)."""
    import re

    from sklearn.model_selection import ParameterSampler

    text = str(spec["tests"]["P1-6"]["families"]["M-D"]["search"])
    grid = {}
    for name in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                 "l2_regularization", "max_features"):
        m = re.search(rf"{name} \[([^\]]+)\]", text)
        if not m:
            raise ValueError(f"P1-6 M-D search has no {name} list")
        vals = [float(v) for v in m.group(1).split(",")]
        grid[name] = [int(v) if name in ("max_iter", "max_leaf_nodes", "min_samples_leaf")
                      else v for v in vals]
    n, seed = int(_find(r"(\d+) configs", text)), int(_find(r"seed (\d+)", text))
    return list(ParameterSampler(grid, n_iter=n, random_state=seed))


def _find(pattern: str, text: str) -> str:
    import re

    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"P1-6 search text has no {pattern!r}")
    return match.group(1)


def gam_lambdas(spec: dict[str, Any]) -> list[float]:

    text = str(spec["tests"]["P1-6"]["families"]["M-C"]["search"])
    return [float(v) for v in _find(r"\[([^\]]+)\]", text).split(",")]


# --- model wrappers ---------------------------------------------------------------------------


class Gam:
    def __init__(self, lam: float):
        self.lam = float(lam)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Gam:
        self.knots = [ST.ns_knots(X[:, j]) for j in range(X.shape[1])]
        self.beta = ST.logistic_fit(self._design(X), y, lam=self.lam)
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X))] + [ST.ns_basis(X[:, j], k)
                                                    for j, k in enumerate(self.knots)])

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        return ST.predict(self._design(X), self.beta)


class Hgb:
    def __init__(self, monotone: bool, **params: Any):
        from sklearn.ensemble import HistGradientBoostingClassifier

        extra = {"monotonic_cst": monotone_cst()} if monotone else {}
        self.model = HistGradientBoostingClassifier(early_stopping=False, random_state=0,
                                                    **params, **extra)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Hgb:
        self.model.fit(X, y)
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def make(family: str, cfg: Any) -> Any:
    if family == "M-C":
        return Gam(cfg)
    return Hgb(family == "M-E", **cfg)


def candidates(family: str, spec: dict[str, Any]) -> list[Any]:
    return gam_lambdas(spec) if family == "M-C" else hgb_configs(spec)


def inner_select(family: str, cands: Sequence[Any], X: np.ndarray, y: np.ndarray,
                 blocks: np.ndarray) -> tuple[int, list[float]]:
    from sklearn.model_selection import GroupKFold

    folds = list(GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups=blocks))
    losses = []
    for cfg in cands:
        fold_loss = []
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                fold_loss.append(np.nan)
                continue
            try:
                p = make(family, cfg).fit(X[tr], y[tr]).predict_proba1(X[te])
                fold_loss.append(ST.log_loss(p, y[te]))
            except np.linalg.LinAlgError:
                fold_loss.append(np.nan)
        losses.append(float(np.nanmean(fold_loss)) if np.isfinite(fold_loss).any() else np.inf)
    return int(np.argmin(losses)), losses


def _outer_fold(args: tuple[Any, ...]) -> dict[str, Any]:
    family, cands, X, y, blocks, held, Xd, held_d = args
    train = blocks != held
    best, losses = inner_select(family, cands, X[train], y[train], blocks[train])
    model = make(family, cands[best]).fit(X[train], y[train])
    test = ~train
    out = {"block": held, "best": best, "inner_log_loss": losses[best],
           "p": model.predict_proba1(X[test]), "model": model}
    out["p_det"] = model.predict_proba1(Xd[held_d]) if held_d.any() else np.array([])
    return out


def cross_validate(family: str, spec: dict[str, Any], X: np.ndarray, y: np.ndarray,
                   blocks: np.ndarray, Xd: np.ndarray, blocks_d: np.ndarray, jobs: int = 1
                   ) -> dict[str, Any]:
    cands = candidates(family, spec)
    ids = list(np.unique(blocks))
    tasks = [(family, cands, X, y, blocks, b, Xd, blocks_d == b) for b in ids]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            folds = list(pool.map(_outer_fold, tasks))
    else:
        folds = [_outer_fold(t) for t in tasks]
    p = np.full(len(y), np.nan)
    pd_ = np.full(len(Xd), np.nan)
    for f in folds:
        p[blocks == f["block"]] = f["p"]
        pd_[blocks_d == f["block"]] = f["p_det"]
    return {"family": family, "p": p, "p_det": pd_, "folds": folds, "candidates": cands}


def permutation_importance(cv: dict[str, Any], X: np.ndarray, y: np.ndarray,
                           blocks: np.ndarray, repeats: int = 20, seed: int = 0
                           ) -> list[dict[str, Any]]:
    base = V.roc_auc(cv["p"], y)
    rng = np.random.default_rng(seed)
    models = {f["block"]: f["model"] for f in cv["folds"]}
    rows = []
    for g, cols in groups().items():
        drops = []
        for _ in range(repeats):
            p = np.full(len(y), np.nan)
            for b, model in models.items():
                sel = np.flatnonzero(blocks == b)
                Xp = X[sel].copy()
                perm = rng.permutation(len(sel))
                Xp[:, cols] = Xp[perm][:, cols]
                p[sel] = model.predict_proba1(Xp)
            drops.append(base - V.roc_auc(p, y))
        rows.append({"family": cv["family"], "group": g, "n_features": len(cols),
                     "repeats": repeats, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops, ddof=1)) if repeats > 1 else np.nan})
    return rows


# --- shardable units: one (family, held-out block) per unit ---------------------------------------

FAMILIES = ("M-C", "M-D", "M-E")
REPEATS = 20


def prepare(ms: Any, det: Any | None) -> dict[str, Any]:
    """Rows with all 28 features finite (Microsoft R1 and detector_final R1) and the block ids
    in fold order (``np.unique``)."""
    X_all = matrix(ms.frame)
    ok = np.all(np.isfinite(X_all), axis=1)
    sub = ms.subset(ok, "all 28 features finite")
    X, y, blocks = X_all[ok], sub.y.astype(int), sub.blocks
    if det is not None:
        Xd_all = matrix(det.frame)
        okd = np.all(np.isfinite(Xd_all), axis=1)
        dsub = det.subset(okd, "all 28 features finite")
        Xd, blocks_d = Xd_all[okd], dsub.blocks
    else:
        dsub, Xd, blocks_d = None, np.empty((0, X.shape[1])), np.array([], dtype=object)
    return {"ms": ms, "ok": ok, "sub": sub, "X": X, "y": y, "blocks": blocks, "dsub": dsub,
            "Xd": Xd, "blocks_d": blocks_d, "ids": [str(b) for b in np.unique(blocks)]}


def unit_key(family: str, block: str) -> str:
    return f"{family}|{block}"


def unit_keys(data: dict[str, Any], families: Sequence[str] = FAMILIES) -> list[str]:
    """Every unit in a fixed order: family (YAML order), then block (fold order)."""
    return [unit_key(f, b) for f in families for b in data["ids"]]


def perm_orders(blocks: np.ndarray, ids: Sequence[str], block: str, repeats: int, seed: int
                ) -> list[list[np.ndarray]]:
    """The row permutations of ``block`` that :func:`permutation_importance` draws, as
    ``[group][repeat]``: one generator per family, consumed group -> repeat -> block (fold
    order), so a unit reproduces them without the other blocks' models."""
    rng = np.random.default_rng(seed)
    sizes = {b: int((blocks == b).sum()) for b in ids}
    out: list[list[np.ndarray]] = []
    for _ in groups():
        per_repeat = []
        for _ in range(repeats):
            for b in ids:
                perm = rng.permutation(sizes[b])
                if b == block:
                    per_repeat.append(perm)
        out.append(per_repeat)
    return out


def run_unit(family: str, cands: Sequence[Any], data: dict[str, Any], block: str,
             repeats: int = REPEATS, seed: int = 0) -> dict[str, Any]:
    """One outer fold of one family: inner selection, refit, held-out (and detector_final)
    probabilities, and the held-out probabilities under each group permutation."""
    X, y, blocks = data["X"], data["y"], data["blocks"]
    train = blocks != block
    best, losses = inner_select(family, cands, X[train], y[train], blocks[train])
    model = make(family, cands[best]).fit(X[train], y[train])
    sel = np.flatnonzero(~train)
    held_d = data["blocks_d"] == block
    p_det = model.predict_proba1(data["Xd"][held_d]) if held_d.any() else np.array([])
    orders = perm_orders(blocks, data["ids"], block, repeats, seed)
    perm = np.empty((len(orders), repeats, len(sel)))
    for gi, cols in enumerate(groups().values()):
        for r in range(repeats):
            Xp = X[sel].copy()
            Xp[:, cols] = Xp[orders[gi][r]][:, cols]
            perm[gi, r] = model.predict_proba1(Xp)
    return {"key": unit_key(family, block), "family": family, "block": block, "best": best,
            "config": cands[best], "inner_log_loss": losses[best], "inner_losses": losses,
            "p": model.predict_proba1(X[sel]), "p_det": p_det, "perm": perm,
            "uids": data["sub"].frame["building_uid"].to_numpy()[sel].tolist(),
            "uids_det": ([] if data["dsub"] is None else
                         data["dsub"].frame["building_uid"].to_numpy()[held_d].tolist())}


def _unit_task(args: tuple[Any, ...]) -> dict[str, Any]:
    import time

    family, cands, data, block, repeats, seed = args
    c0, w0 = time.process_time(), time.perf_counter()
    out = run_unit(family, cands, data, block, repeats, seed)
    return {**out, "cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - w0}


def run_units(keys: Sequence[str], spec: dict[str, Any], data: dict[str, Any], *,
              repeats: int = REPEATS, seed: int = 0, jobs: int = 1) -> Any:
    """Yield finished units of ``keys`` (in completion order when ``jobs > 1``)."""
    from concurrent.futures import as_completed

    cands = {f: candidates(f, spec) for f in {k.split("|", 1)[0] for k in keys}}
    tasks = [(k.split("|", 1)[0], cands[k.split("|", 1)[0]], data, k.split("|", 1)[1],
              repeats, seed) for k in keys]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for fut in as_completed([pool.submit(_unit_task, t) for t in tasks]):
                yield fut.result()
    else:
        for t in tasks:
            yield _unit_task(t)


def importance_from_units(family: str, units: dict[str, dict[str, Any]], data: dict[str, Any],
                          p: np.ndarray, repeats: int) -> list[dict[str, Any]]:
    """:func:`permutation_importance` from the units' permuted held-out probabilities."""
    y, blocks = data["y"], data["blocks"]
    base = V.roc_auc(p, y)
    rows = []
    for gi, (g, cols) in enumerate(groups().items()):
        drops = []
        for r in range(repeats):
            pp = np.full(len(y), np.nan)
            for b in data["ids"]:
                pp[blocks == b] = units[unit_key(family, b)]["perm"][gi][r]
            drops.append(base - V.roc_auc(pp, y))
        rows.append({"family": family, "group": g, "n_features": len(cols),
                     "repeats": repeats, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_sd": float(np.std(drops, ddof=1)) if repeats > 1 else np.nan})
    return rows


def p1_6(ms: Any, det: Any | None, spec: dict[str, Any], ctx: Any, *, jobs: int = 1,
         families: Sequence[str] = FAMILIES, repeats: int = REPEATS
         ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """``ms`` / ``det``: :class:`~nakkhu.followup.analyses.Ref` of Microsoft R1 and
    detector_final R1. Returns (models table, importance table, fold details). In-process
    form of the sharded run (the same units, then :func:`summarize`)."""
    check_spec(spec)
    data = prepare(ms, det)
    units = {u["key"]: u for u in run_units(unit_keys(data, families), spec, data,
                                            repeats=repeats, seed=ctx.seed, jobs=jobs)}
    return summarize(units, data, spec, ctx, families=families, repeats=repeats)


def summarize(units: dict[str, dict[str, Any]], data: dict[str, Any], spec: dict[str, Any],
              ctx: Any, *, families: Sequence[str] = FAMILIES, repeats: int = REPEATS
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The declared P1-6 report from every unit (refuses a missing or misaligned unit)."""
    from nakkhu.followup.analyses import DIST, HAND_V9, MR_D8, Ref, compare

    check_spec(spec)
    ms, ok, sub, y, blocks = data["ms"], data["ok"], data["sub"], data["y"], data["blocks"]
    dsub, blocks_d = data["dsub"], data["blocks_d"]
    uids = sub.frame["building_uid"].to_numpy()
    uids_d = None if dsub is None else dsub.frame["building_uid"].to_numpy()
    missing = [k for k in unit_keys(data, families) if k not in units]
    if missing:
        raise ValueError(f"P1-6: {len(missing)} unit(s) missing, e.g. {missing[:3]}")
    rows, importance, details = [], [], {}
    for family in families:
        p = np.full(len(y), np.nan)
        pdet_all = np.full(0 if uids_d is None else len(uids_d), np.nan)
        folds = []
        for b in data["ids"]:
            u = units[unit_key(family, b)]
            sel = blocks == b
            if list(u["uids"]) != uids[sel].tolist():
                raise ValueError(f"P1-6 unit {u['key']}: held-out rows differ from the data")
            p[sel] = np.asarray(u["p"], dtype=float)
            if uids_d is not None:
                seld = blocks_d == b
                if list(u["uids_det"]) != uids_d[seld].tolist():
                    raise ValueError(f"P1-6 unit {u['key']}: detector rows differ")
                pdet_all[seld] = np.asarray(u["p_det"], dtype=float)
            folds.append({"block": b, "selected": int(u["best"]), "config": u["config"],
                          "inner_log_loss": float(u["inner_log_loss"])})
        cv = {"p": p, "p_det": pdet_all, "folds": folds}
        good = np.isfinite(p)
        base = {"family": family, "inventory": ms.inventory, "reference": ms.reference,
                "n_rows": len(ms.y), "n_excluded_nan_feature": int((~ok).sum()),
                "n": int(good.sum()), "affected": int(y[good].sum()),
                "n_folds": len(cv["folds"])}
        rows.append({**base, "row": "oof", "auc": V.roc_auc(p[good], y[good]),
                     "pr_auc": V.average_precision(p[good], y[good]),
                     "log_loss": ST.log_loss(p[good], y[good]),
                     "brier": ST.brier(p[good], y[good]), "ece": ST.ece(p[good], y[good])})
        model_ref = Ref(sub.inventory, sub.reference, sub.frame.assign(_p=p), sub.y, None,
                        {"subset": "all 28 features finite"})
        for comp in (DIST, HAND_V9, MR_D8):
            rows.append({**base, "row": "delta", **compare(model_ref, "_p", comp, ctx,
                                                           cand_score=p)})
        if dsub is not None and len(dsub.y):
            pdet = cv["p_det"]
            gd = np.isfinite(pdet)
            dref = Ref(dsub.inventory, dsub.reference, dsub.frame, dsub.y, None,
                       {"subset": "all 28 features finite"})
            rows.append({**base, "row": "oof_detector_final", "inventory": dsub.inventory,
                         "n": int(gd.sum()), "affected": int(dsub.y[gd].sum()),
                         "auc": V.roc_auc(pdet[gd], dsub.y[gd]),
                         "pr_auc": V.average_precision(pdet[gd], dsub.y[gd]),
                         "log_loss": ST.log_loss(pdet[gd], dsub.y[gd]),
                         "brier": ST.brier(pdet[gd], dsub.y[gd]),
                         "ece": ST.ece(pdet[gd], dsub.y[gd])})
            for comp in (DIST, HAND_V9, MR_D8):
                rows.append({**base, "row": "delta_detector_final", **compare(
                    dref, "_p", comp, ctx, cand_score=pdet)})
        importance += importance_from_units(family, units, data, p, repeats)
        details[family] = folds
    return pd.DataFrame(rows), pd.DataFrame(importance), details
