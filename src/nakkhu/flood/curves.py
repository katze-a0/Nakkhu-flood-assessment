"""Thinned ROC / PR curves of a finished ``flood validate`` run (plan 009 figures).

The validate run writes every distinct-score point of each curve (``roc_curves.csv``,
``pr_curves.csv``), which is over the 5 MB artifact cap. :func:`thin_run` derives thinned
copies deterministically from those files alone: no building, answer or HAND value is read,
so it is not a test access and never touches the test journal.

Rule (per curve = one inventory x reference): the first and last points are kept; a point is
dropped only if it lies within ``tol`` of the segment between the kept points around it
(Ramer-Douglas-Peucker, Euclidean distance in the plotted unit square: (fpr, tpr) for ROC,
(recall, precision) for PR). ``tol`` starts at :data:`TOL0` and is doubled until the curve has
at most :data:`CAP` points. The new run records the source run, the SHA-256 of both source
files, the rule and, per curve, the tolerance used, the point counts and the ROC-AUC
(trapezoid from (0, 0)) / average precision before and after thinning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project, ResolvedConfig

TOL0 = 1e-4
CAP = 1000
GROUP = ["inventory", "reference"]
CURVES = {"roc_curves": ("fpr", "tpr"), "pr_curves": ("recall", "precision")}
STAGE = "validation-curves"


def rdp_keep(xy: np.ndarray, tol: float) -> np.ndarray:
    """Boolean mask of the points kept by Ramer-Douglas-Peucker at tolerance ``tol``: the
    endpoints always; every dropped point lies within ``tol`` of the segment joining the
    nearest kept points on either side of it."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = xy[a], xy[b]
        mid = xy[a + 1:b]
        d = q - p
        length = float(np.hypot(*d))
        if length == 0.0:
            dist = np.hypot(*(mid - p).T)
        else:     # distance to the segment (projection clipped to its ends)
            t = np.clip(((mid - p) @ d) / (length * length), 0.0, 1.0)
            dist = np.hypot(*(mid - (p + t[:, None] * d)).T)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def thin_curve(xy: np.ndarray, tol0: float = TOL0, cap: int = CAP) -> tuple[np.ndarray, float]:
    """(kept mask, tolerance used): ``tol0`` doubled until at most ``cap`` points are kept."""
    tol = float(tol0)
    keep = rdp_keep(xy, tol)
    while keep.sum() > cap:
        tol *= 2.0
        keep = rdp_keep(xy, tol)
    return keep, tol


def roc_auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under (0, 0) + the curve points (trapezoids; equals the Mann-Whitney AUC with
    ties counted one half on the full distinct-score curve)."""
    x = np.r_[0.0, np.asarray(fpr, dtype=np.float64)]
    y = np.r_[0.0, np.asarray(tpr, dtype=np.float64)]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def average_precision_steps(recall: np.ndarray, precision: np.ndarray) -> float:
    """Step sum as ``nakkhu.flood.validate.average_precision`` (from recall 0)."""
    r = np.r_[0.0, np.asarray(recall, dtype=np.float64)]
    return float(np.sum(np.diff(r) * np.asarray(precision, dtype=np.float64)))


def thin_table(frame: pd.DataFrame, x: str, y: str, tol0: float = TOL0, cap: int = CAP,
               group: list[str] | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Thinned rows (original order and columns) and one record per curve (one curve per
    ``group`` key, default :data:`GROUP`)."""
    group = list(group or GROUP)
    keep_all = np.zeros(len(frame), dtype=bool)
    records = []
    for key, idx in frame.groupby(group, sort=False).indices.items():
        g = frame.iloc[idx]
        xy = g[[x, y]].to_numpy(dtype=np.float64)
        finite = np.isfinite(xy).all(axis=1)
        keep = np.zeros(len(g), dtype=bool)
        sub, tol = thin_curve(xy[finite], tol0, cap)
        keep[np.flatnonzero(finite)[sub]] = True
        keep_all[idx[keep]] = True
        names = key if isinstance(key, tuple) else (key,)
        rec: dict[str, Any] = {**dict(zip(group, names, strict=True)), "points": len(g),
                               "kept": int(keep.sum()), "non_finite_dropped": int((~finite).sum()),
                               "tolerance": tol}
        kept = g.iloc[np.flatnonzero(keep)]
        if x == "fpr":
            rec["auc_full"] = roc_auc_trapezoid(g.loc[finite, x], g.loc[finite, y])
            rec["auc_thin"] = roc_auc_trapezoid(kept[x], kept[y])
        else:
            rec["ap_full"] = average_precision_steps(g.loc[finite, x], g.loc[finite, y])
            rec["ap_thin"] = average_precision_steps(kept[x], kept[y])
        records.append(rec)
    return frame.loc[keep_all].reset_index(drop=True), records


def thin_run(project: Project, source: str, tol0: float = TOL0, cap: int = CAP,
             promote_outputs: bool = True) -> tuple[Any, dict[str, Any]]:
    """Thinned copies of a validate run's curves in a new ``validation-curves`` run;
    promoted as ``validation/<name>_thin.csv``."""
    from nakkhu.artifacts import promote
    from nakkhu.evaluate.guard import sha256_file
    from nakkhu.runs import atomic_write_json, create_run

    src = project.paths.runs / source
    files = {name: src / f"{name}.csv" for name in CURVES}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("source curve file(s) missing: " + ", ".join(missing))
    hashes = {f"{name}.csv": sha256_file(p) for name, p in files.items()}
    rule = {"method": "Ramer-Douglas-Peucker, endpoints kept", "tolerance_start": tol0,
            "tolerance_growth": "doubled until points <= cap", "cap_points": cap,
            "space": {"roc_curves": "(fpr, tpr)", "pr_curves": "(recall, precision)"},
            "group": GROUP, "test_access": "none (reads only the source run's curve files)"}
    run = create_run(project, STAGE, ResolvedConfig(
        values={"command": "flood-thin-curves", "source_run": source, "source_sha256": hashes,
                "rule": rule}, source="cli"))
    summary: dict[str, Any] = {"source_run": source, "source_sha256": hashes, "rule": rule,
                               "curves": {}}
    with run.step("thin-curves"):
        for name, (x, y) in CURVES.items():
            frame = pd.read_csv(files[name])
            thin, records = thin_table(frame, x, y, tol0, cap)
            thin.to_csv(run.path / f"{name}_thin.csv", index=False)
            summary["curves"][name] = records
            summary[f"{name}_rows"] = {"full": len(frame), "thin": len(thin)}
        atomic_write_json(run.path / "thin_curves_summary.json", summary)
        run.write_summary({k: v for k, v in summary.items() if k != "curves"})
    if promote_outputs:
        summary["promoted"] = []
        for name in [*(f"{n}_thin.csv" for n in CURVES), "thin_curves_summary.json"]:
            promote(run.path / name, f"validation/{name}", project.paths, replace=True)
            summary["promoted"].append(f"artifacts/validation/{name}")
    return run, summary
