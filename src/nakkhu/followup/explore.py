"""The one Phase-2 preparation explore run (prereg_v9_a3 P1-explore; user decision 2026-10-17).

Journal note ``prereg_v9_a3:explore 1 ...`` (:data:`EXPLORE_P2`), reserved on the laptop and
executed remotely through :mod:`nakkhu.followup.remote` like a declared test. It computes only:

1. the AUCs of the six members of the a3 Phase-2 HAND rule (``phase2.candidates.hand_rule``)
   on Microsoft R1, on ONE row set: the P1-9 covered rows (finite ``hand_team``) on which all
   six members are finite; then the declared rule is applied mechanically (best AUC; members
   within 0.005 of it are tied; ties prefer glo30 over fabdem, then hand_mr_d8, then
   hand_team_acc, then hand_mr_rem);
2. the M-C refit of ``phase2.candidates.ml_model``: all Nakkhu Microsoft R1 rows with every
   P1-6 feature finite, lambda chosen by 5-fold GroupKFold by block (the P1-6 inner search),
   refit on all rows, no recalibration; written as a JSON model file (lambda, knots,
   coefficients, feature order) whose sha256 prereg_v9_a4 records.

No other quantity is computed: no comparison with distance, no CI, no model score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPLORE_P2 = ("explore 1 phase-2 selection: six HAND members on one common covered row set "
              "and the M-C refit")
OUTPUTS = ("p2_select_hand_members.csv", "p2_select_rule.json", "p2_mc_model.json",
           "p2_mc_feature_spec.json")
TIE_MARGIN = 0.005
DEM_ORDER = ("glo30", "fabdem")
PREDICTOR_ORDER = ("hand_mr_d8", "hand_team_acc", "hand_mr_rem")
MEMBERS = (("hand_mr_d8", "glo30", "hand_mr_d8__glo30"),
           ("hand_mr_d8", "fabdem", "hand_mr_d8__fabdem"),
           ("hand_mr_rem", "glo30", "hand_mr_rem__glo30"),
           ("hand_mr_rem", "fabdem", "hand_mr_rem__fabdem"),
           ("hand_team_acc", "glo30", "hand_team_acc__glo30__breach_burn"),
           ("hand_team_acc", "fabdem", "hand_team_acc__fabdem__breach_burn"))
MODEL_SCHEMA = 1


def apply_rule(table: pd.DataFrame, margin: float = TIE_MARGIN) -> dict[str, Any]:
    """The a3 hand_rule on per-member AUCs (columns predictor, dem, auc)."""
    best = float(table["auc"].max())
    tied = table[table["auc"] >= best - margin - 1e-12].copy()
    tied["dem_rank"] = tied["dem"].map({d: i for i, d in enumerate(DEM_ORDER)})
    tied["pred_rank"] = tied["predictor"].map({p: i for i, p in enumerate(PREDICTOR_ORDER)})
    pick = tied.sort_values(["dem_rank", "pred_rank"], kind="stable").iloc[0]
    top = table.loc[table["auc"].idxmax()]
    return {"best_member": f"{top['predictor']} x {top['dem']}", "best_auc": best,
            "tie_margin": margin,
            "tied_members": [f"{p} x {d}" for p, d in zip(tied["predictor"], tied["dem"],
                                                         strict=True)],
            "pick": {"predictor": str(pick["predictor"]), "dem": str(pick["dem"]),
                     "column": str(pick["column"]), "auc": float(pick["auc"])},
            "rule": "highest AUC; members within 0.005 of it are tied; ties prefer glo30 over "
                    "fabdem, then hand_mr_d8, then hand_team_acc, then hand_mr_rem "
                    "(prereg_v9_a3 phase2.candidates.hand_rule)"}


def hand_members(ms: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    from nakkhu.flood.validate import roc_auc

    frame = ms.frame
    covered = np.isfinite(frame["hand_team"].to_numpy(dtype=float))
    finite = np.column_stack([np.isfinite(frame[c].to_numpy(dtype=float))
                              for *_, c in MEMBERS]).all(axis=1)
    common = covered & finite
    y = np.asarray(ms.y).astype(int)[common]
    blocks = frame["block_id"].astype(str).to_numpy()[common]
    rows = []
    for pred, dem, col in MEMBERS:
        v = frame[col].to_numpy(dtype=float)
        rows.append({"predictor": pred, "dem": dem, "column": col,
                     "conditioning": "breach_burn", "inventory": ms.inventory,
                     "reference": ms.reference, "rows": "P1-9 covered rows finite for all six",
                     "n_covered": int(covered.sum()),
                     "n_nan_on_covered": int((covered & ~np.isfinite(v)).sum()),
                     "n": int(common.sum()), "affected": int(y.sum()),
                     "n_blocks": int(len(set(blocks))),
                     "auc": float(roc_auc(-v[common], y))})
    table = pd.DataFrame(rows)
    return table, apply_rule(table)


def refit_mc(ms: Any, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from nakkhu.followup import ml

    ml.check_spec(spec)
    data = ml.prepare(ms, None)
    X, y, blocks = data["X"], data["y"], data["blocks"]
    cands = ml.candidates("M-C", spec)
    best, losses = ml.inner_select("M-C", cands, X, y, blocks)
    model = ml.make("M-C", cands[best]).fit(X, y)
    feature_spec = {"schema": MODEL_SCHEMA, "family": "M-C",
                    "features": [{"name": f, "prep_column": c, "transform": t, "group": g}
                                 for f, c, t, g in ml.FEATURES],
                    "basis": "natural cubic spline df 4 per feature (nakkhu.followup.stats."
                             "ns_basis), knots = 1/25/50/75/99th percentiles of the training "
                             "values, values clamped to the boundary knots; design = intercept "
                             "then the 4 basis columns of each feature in this order",
                    "link": "logistic", "orientation": "probability (higher = more exposed)"}
    doc = {"schema": MODEL_SCHEMA, "family": "M-C", "lambda": float(cands[best]),
           "lambda_grid": [float(c) for c in cands],
           "inner_mean_log_loss": [float(v) for v in losses],
           "selection": "5-fold GroupKFold by plan-005 block (sklearn GroupKFold, deterministic), "
                        "lowest mean inner log-loss, refit on all rows (prereg_v9_a3 "
                        "phase2.candidates.ml_model); never recalibrated",
           "training": {"inventory": ms.inventory, "reference": ms.reference,
                        "rows": "Microsoft R1 v9 primary rows with all 28 features finite",
                        "n": int(len(y)), "affected": int(y.sum()),
                        "n_blocks": int(len(set(blocks))),
                        "n_excluded_nan_feature": int((~data["ok"]).sum())},
           "features": [f for f, *_ in ml.FEATURES],
           "knots": [[float(v) for v in k] for k in model.knots],
           "coefficients": [float(v) for v in model.beta]}
    p = model.predict_proba1(X)
    q = predict(doc, _frame_from_matrix(X))
    if not np.allclose(p, q, rtol=0, atol=1e-12):
        raise ValueError("the serialised M-C model does not reproduce the fitted model")
    return doc, feature_spec


def _frame_from_matrix(X: np.ndarray) -> pd.DataFrame:
    """Prep-column frame whose ``ml.matrix`` is ``X`` (log1p columns inverted)."""
    from nakkhu.followup import ml

    cols = {}
    for j, (_, col, transform, _) in enumerate(ml.FEATURES):
        cols[col] = np.expm1(X[:, j]) if transform == "log1p" else X[:, j]
    return pd.DataFrame(cols)


def predict(doc: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Probabilities of a serialised M-C model for a prep-column frame (NaN where a feature
    is not finite)."""
    from nakkhu.followup import ml
    from nakkhu.followup import stats as ST

    if doc.get("family") != "M-C" or doc.get("features") != [f for f, *_ in ml.FEATURES]:
        raise ValueError("model file is not the P1-6 M-C feature set")
    X = ml.matrix(frame)
    ok = np.all(np.isfinite(X), axis=1)
    design = np.column_stack([np.ones(int(ok.sum()))] + [
        ST.ns_basis(X[ok, j], np.asarray(k)) for j, k in enumerate(doc["knots"])])
    out = np.full(len(frame), np.nan)
    out[ok] = ST.predict(design, np.asarray(doc["coefficients"], dtype=float))
    return out


def score(project: Any, cfg: dict[str, Any], spec: dict[str, Any],
          prep: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the labels (after the receipt check), compute both parts; returns
    ``{file name: DataFrame | dict}`` and an info dict."""
    from nakkhu.followup import run as R

    labels = R.load_labels(project, list(prep))
    refs = R.references(prep, labels)
    ms = R.pick(refs, ("microsoft", "R1"))[0]
    table, rule = hand_members(ms)
    model, fspec = refit_mc(ms, spec)
    info = {"hand_rule": rule, "mc_lambda": model["lambda"], "mc_training": model["training"]}
    return {"p2_select_hand_members.csv": table, "p2_select_rule.json": rule,
            "p2_mc_model.json": model, "p2_mc_feature_spec.json": fspec}, info


def write_output(path: Path, value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False, lineterminator="\n")
    else:
        path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
