"""Reserve on the laptop, execute on a remote host, promote on the laptop (offload-experiments
2.8), for the two label-reading analyses of the main-river recompute (``exposure``, ``event``).

1. :func:`reserve` (laptop, holds the live journal): prereg verify clean, the hold off
   (``runs/eval/test_hold.yaml``, read only), never reserved before (journal note
   ``main_river_v1:<analysis>``, one disclosed ``--retry`` allowed after a failure before any
   output), a clean code commit (``configs/paper/`` excepted: the manuscript numbers file is
   not read by any analysis), the promoted label-free ``hand-mr`` artifacts, then the run
   directory, one journal entry (:func:`nakkhu.evaluate.guard.record_test_access`) and a
   receipt with the sha256 of the journal line, the code commit, the configs and every input.
2. :func:`execute` (remote host, from a bundle): refuses on a host holding the live journal,
   verifies the receipt against the shipped read-only journal copy, the bundle manifest and the
   inputs, and only then reads labels. Writes into the reserved run; never journals.
3. :func:`promote_analysis` (laptop): the pulled run against the receipt, the live journal and
   the execution record; outputs promoted to ``artifacts/mainriver/<dir>/`` (new names; nothing
   existing is replaced).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nakkhu.config import Project
from nakkhu.evaluate.guard import sha256_file
from nakkhu.followup.remote import (
    BUNDLE_MANIFEST,
    EXECUTION_FILE,
    JOURNAL_COPY,
    RECEIPT_FILE,
    ReceiptError,
    bundle_problems,
    host_type,
    load_receipt,
    sha256_bytes,
)
from nakkhu.mainriver import spec as S

ANALYSES = ("exposure", "event", "roc")
CODE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock", ".python-version")
# Paths that must equal HEAD on the laptop: this package and its configs. Other uncommitted
# paths (another agent's work in progress) do not reach the remote host, which executes the
# bundle of HEAD (manifest-checked); they are listed in the receipt for disclosure.
OWN_PATHS = ("src/nakkhu/mainriver", "configs/hand/main_river_v1.yaml", "configs/mainriver",
             "src/nakkhu/uncertainty", "src/nakkhu/flood", "src/nakkhu/hazard.py",
             "src/nakkhu/evaluate", "uv.lock", "pyproject.toml")
HAND_FILES = ("summary.json", "surface_grid.csv", "count_stability.csv",
              "building_hand_mr_microsoft.parquet", "building_hand_mr_detector_final.parquet")
UNCERTAINTY_OUTPUTS = ("calibration_curve.csv", "ece.csv", "tier_counts_factorial.parquet",
                       "variance_decomposition.csv", "ppi_tier_counts.csv",
                       "coverage_check.json", "inventory_comparison.csv", "falsifiers.json")
EVENT_OUTPUTS = ("tier_hit_rates.csv", "cumulative_skill.csv", "odds_ratios.csv",
                 "competitors_auc.csv", "config_skill_grid.csv", "config_skill_range.csv",
                 "calibration_h6.csv", "h6_reliability.csv", "by_inventory.csv",
                 "roc_curves_thin.csv", "pr_curves_thin.csv", "h6_skill.csv",
                 "unosat_competitors.csv", "team_own_coverage.csv", "damage_by_tier.csv",
                 "validation_summary.json")
# roc: ROC curves of main-river HAND and distance to the channel for the event-skill figure
ROC_OUTPUTS = ("roc_predictor_curves_thin.csv", "roc_predictor_summary.json")
ROC_HAND_FILES = ("summary.json", "building_hand_mr_microsoft.parquet",
                  "building_hand_mr_detector_final.parquet")
LOCAL_ONLY = ("roc_curves.csv", "pr_curves.csv",      # full curves stay in the run (> 5 MB)
              "roc_predictor_curves.csv")


def outputs_of(analysis: str) -> tuple[str, ...]:
    return {"exposure": UNCERTAINTY_OUTPUTS, "event": EVENT_OUTPUTS,
            "roc": ROC_OUTPUTS}[analysis]


def note_of(rc: dict[str, Any], analysis: str, retry: bool = False) -> str:
    return f"{rc['version']}:{analysis}" + (" retry" if retry else "")


def code_state(project: Project) -> dict[str, Any]:
    root = project.paths.root

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                 check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    own = git("status", "--porcelain", "--", *OWN_PATHS)
    other = git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS)
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if own is None else bool(own),
            "other_uncommitted": [] if not other else sorted(
                ln[3:] for ln in other.splitlines() if ln.strip())}


def hand_run(project: Project, rc: dict[str, Any]) -> Path:
    """The ``hand-mr`` run whose outputs are promoted under ``artifacts/mainriver/hand/``."""
    from nakkhu.artifacts import load_index

    rel = f"{rc['artifact_dir']}/hand/surface_grid.csv"
    hit = [e for e in load_index(project.paths)["artifacts"] if e["path"] == rel]
    if not hit:
        raise S.MainRiverError(f"artifacts/{rel} is not promoted (run `mainriver surfaces` "
                               "remotely, pull, `mainriver promote-surfaces`)")
    return project.paths.runs / hit[0]["source_run"]


def hand_inputs(project: Project, run_dir: Path, analysis: str = "event") -> dict[str, str]:
    if analysis == "roc":     # building tables only (no raster is read)
        files = [run_dir / f for f in ROC_HAND_FILES]
        files += [project.paths.artifacts / "followup" / f"prep_core_{inv}.parquet"
                  for inv in ("microsoft", "detector_final")]
        return {project.paths.relative(f): sha256_file(f) for f in files}
    files = [run_dir / f for f in HAND_FILES]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    interim = project.paths.resolve(summary["interim"])
    files.append(interim / "grid" / "cell_index.tif")
    for s in summary["surfaces"].values():
        if s.get("raster"):
            files.append(project.paths.resolve(s["raster"]))
    return {project.paths.relative(f): sha256_file(f) for f in files}


def label_inputs(project: Project, rc: dict[str, Any], analysis: str) -> dict[str, str]:
    from nakkhu.audit import core as audit_core
    from nakkhu.flood import validate as V

    files: list[Path] = []
    if analysis == "roc":
        for inv in rc["analyses"]["roc"]["inventories"]:
            cm = V.change_map_path(project, inv)
            if cm is None:
                raise S.MainRiverError(f"change map of {inv} missing")
            files.append(cm)
        files.append(project.paths.artifacts / V.ESTIMATES)
    elif analysis == "exposure":
        for task in ("precision_pool", "precision_microsoft", "recall_points",
                     "recall_nearmiss"):
            files += [audit_core.items_path(project, task),
                      audit_core.responses_path(project, task)]
        files.append(audit_core.task_dir(project, "precision_pool") / "members.parquet")
    else:
        from nakkhu.followup import run as FR
        from nakkhu.followup import spec as FS

        for inv in rc["analyses"]["event"]["inventories"]:
            cm = V.change_map_path(project, inv)
            if cm is None:
                raise S.MainRiverError(f"change map of {inv} missing")
            files.append(cm)
        files += [audit_core.items_path(project, "flood_trace"),
                  audit_core.responses_path(project, "flood_trace"),
                  project.paths.artifacts / V.ESTIMATES,
                  FR.unosat_zip(project, FS.load_prep_config(project, rc["prep_config"]))]
    out = {}
    for f in files:
        if not Path(f).is_file():
            raise S.MainRiverError(f"input {f} missing")
        out[project.paths.relative(f)] = sha256_file(Path(f))
    return out


def prior(project: Project, rc: dict[str, Any], analysis: str) -> list[dict[str, Any]]:
    from nakkhu.evaluate.guard import read_test_journal

    base = note_of(rc, analysis)
    return [e for e in read_test_journal(project) if e.get("note") in (base, f"{base} retry")]


def reserve(project: Project, analysis: str, *, retry: bool = False, dry_run: bool = False,
            config: str = S.RECOMPUTE_CONFIG) -> dict[str, Any]:
    from nakkhu.config import ResolvedConfig
    from nakkhu.evaluate.guard import journal_path, record_test_access, require_frozen, verify
    from nakkhu.runs import CONFIG_FILE, atomic_write_json, create_run, utc_now
    from nakkhu.uncertainty.run import hold_state

    if analysis not in ANALYSES:
        raise S.MainRiverError(f"unknown analysis {analysis!r}; one of {ANALYSES}")
    rc = S.load_recompute(project, config)
    problems = [f"prereg: {p}" for p in verify(project)["problems"]]
    held, why = hold_state(project)
    if held:
        problems.append(f"hold: {why}")
    done = prior(project, rc, analysis)
    if retry and len(done) != 1:
        problems.append(f"--retry needs exactly one earlier entry, found {len(done)}")
    elif not retry and done:
        problems.append(f"{note_of(rc, analysis)} was reserved before ({done[0].get('utc')})")
    if retry and done:
        failed = _failed_without_outputs(project, done[0]["run"], analysis)
        if not failed:
            problems.append("the earlier run did not fail before writing an output")
    code = code_state(project)
    if not code["commit"]:
        problems.append("code: no git commit")
    elif code["dirty"] is not False:
        problems.append(f"code: {', '.join(OWN_PATHS)} must equal HEAD (commit first)")
    inputs: dict[str, str] = {}
    hand: dict[str, str] = {}
    hrun = None
    if not problems:
        try:
            hrun = hand_run(project, rc)
            hand = hand_inputs(project, hrun, analysis)
            inputs = label_inputs(project, rc, analysis)
        except S.MainRiverError as error:
            problems.append(str(error))
    if analysis == "exposure" and not problems:
        from nakkhu.mainriver.analyses import uncertainty_spec
        from nakkhu.uncertainty.run import FileInputs

        problems += [p for p in FileInputs().problems(project, uncertainty_spec(project, rc))
                     if not p.startswith("hold")]
    if problems or hrun is None:
        return {"ready": False, "analysis": analysis, "problems": problems}
    if dry_run:
        return {"ready": True, "analysis": analysis, "dry_run": True,
                "hand_run": project.paths.relative(hrun), "code_commit": code["commit"]}
    ens_path = str(rc["ensemble_config"])
    values = {"command": "mainriver-run", "version": rc["version"], "analysis": analysis,
              "retry": retry, "code_commit": code["commit"], "execution": "remote (receipt)",
              "recompute_config_sha256": sha256_file(project.paths.resolve(config)),
              "ensemble_config_sha256": sha256_file(project.paths.resolve(ens_path)),
              "hand_run": project.paths.relative(hrun)}
    resolved = ResolvedConfig(values=values, source=config)
    guard = require_frozen(project)
    run = create_run(project, str(rc["run_stage"]), resolved)
    access = record_test_access(project, run.ref, resolved.cfghash, guard["prereg_sha256"],
                                guard["split_sha256"], note=note_of(rc, analysis, retry),
                                extra={"recompute": rc["version"], "analysis": analysis})
    lines = journal_path(project).read_text(encoding="utf-8").splitlines()
    index = next(i + 1 for i in range(len(lines) - 1, -1, -1)
                 if lines[i] and json.loads(lines[i]) == access["entry"])
    receipt = {"schema": 1, "kind": "mainriver", "analysis": analysis, "run": run.ref,
               "run_cfghash": resolved.cfghash, "retry": retry,
               "config_sha256": sha256_file(run.path / CONFIG_FILE),
               "journal": {"index": index, "line_sha256": sha256_bytes(lines[index - 1].encode(
                   "utf-8")), "entry": access["entry"]},
               "code_commit": code["commit"], "recompute_config": config,
               "laptop_other_uncommitted": code["other_uncommitted"],
               "inputs": {**hand, **inputs}, "values": values, "reserved_utc": utc_now(),
               "outputs": list(outputs_of(analysis))}
    atomic_write_json(run.path / RECEIPT_FILE, receipt)
    return {"ready": True, "analysis": analysis, "run": run.ref,
            "receipt": project.paths.relative(run.path / RECEIPT_FILE), "journal_index": index,
            "journal_note": access["entry"]["note"]}


def _failed_without_outputs(project: Project, run_ref: str, analysis: str) -> bool:
    stage, _, rid = run_ref.partition("/")
    d = project.paths.runs / stage / rid
    return (d / "failure.json").is_file() and not any((d / n).exists()
                                                      for n in outputs_of(analysis))


def _line_problems(lines: list[str], receipt: dict[str, Any], where: str) -> list[str]:
    j = receipt["journal"]
    idx = int(j["index"])
    if not 1 <= idx <= len(lines):
        return [f"{where}: {len(lines)} lines, receipt names line {idx}"]
    line = lines[idx - 1]
    out = []
    if sha256_bytes(line.encode("utf-8")) != j["line_sha256"]:
        out.append(f"{where}: line {idx} sha256 differs from the receipt")
    entry = json.loads(line)
    if entry.get("run") != receipt["run"] or entry.get("eval_cfghash") != receipt["run_cfghash"]:
        out.append(f"{where}: line {idx} is not this run")
    want = j["entry"]["note"]
    base = want.removesuffix(" retry")
    same = [i + 1 for i, ln in enumerate(lines) if ln and
            str(json.loads(ln).get("note") or "") in (base, f"{base} retry")]
    if len(same) != (2 if receipt["retry"] else 1) or same[-1] != idx:
        out.append(f"{where}: {base} has journal lines {same}; line {idx} must be the last")
    return out


def _run_dir(project: Project, receipt: dict[str, Any]) -> Path:
    stage, _, rid = receipt["run"].partition("/")
    return project.paths.runs / stage / rid


def verify_receipt(project: Project, path: str | Path, analysis: str
                   ) -> tuple[dict[str, Any], str]:
    from nakkhu.evaluate.guard import journal_path
    from nakkhu.runs import CONFIG_FILE

    receipt, rsha = load_receipt(path)
    if receipt.get("kind") != "mainriver" or receipt["analysis"] != analysis:
        raise ReceiptError(f"receipt is not a mainriver {analysis} receipt")
    problems = []
    if journal_path(project).exists():
        problems.append("this host holds the live test journal: execute on a remote host")
    problems += bundle_problems(project, receipt["code_commit"])
    copy = project.paths.root / JOURNAL_COPY
    if not copy.is_file():
        problems.append(f"{JOURNAL_COPY} missing")
    else:
        problems += _line_problems(copy.read_text(encoding="utf-8").splitlines(), receipt,
                                   JOURNAL_COPY)
    for rel, digest in receipt["inputs"].items():
        f = project.paths.resolve(rel)
        if not f.is_file() or sha256_file(f) != digest:
            problems.append(f"input {rel} missing or altered")
    cf = _run_dir(project, receipt) / CONFIG_FILE
    if not cf.is_file() or sha256_file(cf) != receipt["config_sha256"]:
        problems.append(f"{receipt['run']}: {CONFIG_FILE} is not the reserved one")
    if problems:
        raise ReceiptError("receipt refused: " + "; ".join(problems))
    return receipt, rsha


def execute(project: Project, receipt_path: str | Path, analysis: str) -> dict[str, Any]:
    from nakkhu.flood.curves import thin_table
    from nakkhu.mainriver import analyses as A
    from nakkhu.runs import Run, atomic_write_json, collect_env, utc_now
    from nakkhu.uncertainty.run import _jsonable

    receipt, rsha = verify_receipt(project, receipt_path, analysis)
    run_dir = _run_dir(project, receipt)
    if (run_dir / EXECUTION_FILE).exists() or (run_dir / "summary.json").exists():
        raise ReceiptError(f"{receipt['run']} was already executed")
    rc = S.load_recompute(project, receipt["recompute_config"])
    ens = S.load_ensemble(project, rc["ensemble_config"])
    hrun = project.paths.resolve(receipt["values"]["hand_run"])
    stage, _, rid = receipt["run"].partition("/")
    run = Run(stage=stage, run_id=rid, path=run_dir)
    started = utc_now()
    written: dict[str, str] = {}
    try:
        with run.step(analysis):
            if analysis == "exposure":
                out = A.exposure(project, rc, ens, hrun)
                for name in UNCERTAINTY_OUTPUTS:
                    obj, path = out[name], run_dir / name
                    if name.endswith(".json"):
                        atomic_write_json(path, _jsonable(obj))
                    elif name.endswith(".parquet"):
                        obj.to_parquet(path, index=False)
                    else:
                        obj.to_csv(path, index=False, lineterminator="\n")
            elif analysis == "roc":
                full, summary = A.roc_predictor_curves(project, rc, ens, hrun)
                full.to_csv(run_dir / "roc_predictor_curves.csv", index=False,
                            lineterminator="\n")
                thin, recs = thin_table(full, "fpr", "tpr",
                                        group=["inventory", "reference", "predictor"])
                thin.to_csv(run_dir / "roc_predictor_curves_thin.csv", index=False,
                            lineterminator="\n")
                summary["thinning"] = recs
                summary["rows"] = {"full": len(full), "thin": len(thin)}
                atomic_write_json(run_dir / "roc_predictor_summary.json", summary)
            else:
                tables, summary = A.event(project, rc, ens, hrun)
                for name, frame in tables.items():
                    frame.to_csv(run_dir / name, index=False, lineterminator="\n")
                for name, (x, y) in (("roc_curves", ("fpr", "tpr")),
                                     ("pr_curves", ("recall", "precision"))):
                    thin, recs = thin_table(tables[f"{name}.csv"], x, y)
                    thin.to_csv(run_dir / f"{name}_thin.csv", index=False, lineterminator="\n")
                    summary[f"{name}_thinning"] = recs
                atomic_write_json(run_dir / "validation_summary.json", summary)
            for name in outputs_of(analysis):
                written[name] = sha256_file(run_dir / name)
            run.write_summary({"analysis": analysis, "outputs": list(written),
                               "journal_note": receipt["journal"]["entry"]["note"],
                               "executed_on": host_type(),
                               "reporting": "post hoc (main-river HAND, plan 014 P1-1)"
                               if analysis != "roc" else summary["reporting"]})
    except Exception as error:
        atomic_write_json(run_dir / "failure.json", {"error": f"{type(error).__name__}: "
                                                              f"{error}", "utc": utc_now(),
                                                     "outputs_written": sorted(written)})
        raise
    env = collect_env(Path.cwd())
    env.pop("git", None)
    atomic_write_json(run_dir / EXECUTION_FILE, {
        "analysis": analysis, "run": run.ref, "receipt_sha256": rsha,
        "code_commit": receipt["code_commit"], "host_type": host_type(),
        "started_utc": started, "finished_utc": utc_now(), "outputs": written, "env": env})
    return {"ready": True, "analysis": analysis, "run": run.ref, "outputs": written,
            "host_type": host_type()}


def find_receipt(project: Project, analysis: str, rc: dict[str, Any]) -> Path:
    base = project.paths.runs / str(rc["run_stage"])
    found = []
    for p in sorted(base.glob(f"*/{RECEIPT_FILE}")) if base.is_dir() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("kind") == "mainriver" and r.get("analysis") == analysis:
            found.append((int(r["journal"]["index"]), p))
    if not found:
        raise ReceiptError(f"no {analysis} receipt under runs/{rc['run_stage']}/")
    return max(found)[1]


def promote_analysis(project: Project, analysis: str,
                     config: str = S.RECOMPUTE_CONFIG) -> dict[str, Any]:
    from nakkhu.artifacts import promote
    from nakkhu.evaluate.guard import journal_path
    from nakkhu.runs import CONFIG_FILE, journal_status

    rc = S.load_recompute(project, config)
    receipt, rsha = load_receipt(find_receipt(project, analysis, rc))
    jp = journal_path(project)
    if not jp.is_file():
        raise ReceiptError("live journal missing: promotion happens on the laptop")
    problems = _line_problems(jp.read_text(encoding="utf-8").splitlines(), receipt,
                              "live journal")
    run_dir = _run_dir(project, receipt)
    if sha256_file(run_dir / CONFIG_FILE) != receipt["config_sha256"]:
        problems.append("run config is not the reserved one")
    ex_path = run_dir / EXECUTION_FILE
    if not ex_path.is_file():
        raise ReceiptError(f"{receipt['run']}: {EXECUTION_FILE} missing (not pulled?)")
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    if ex.get("receipt_sha256") != rsha or ex.get("code_commit") != receipt["code_commit"]:
        problems.append("execution was not made from this receipt / commit")
    if sorted(ex.get("outputs") or {}) != sorted(receipt["outputs"]):
        problems.append("outputs differ from the declared ones")
    for name, digest in (ex.get("outputs") or {}).items():
        if sha256_file(run_dir / name) != digest:
            problems.append(f"{name} altered since execution")
    if journal_status(run_dir) != "complete":
        problems.append(f"run journal is {journal_status(run_dir)}")
    if problems:
        raise ReceiptError("promotion refused: " + "; ".join(problems))
    sub = rc["analyses"][analysis]["outputs_dir"]
    promoted = [promote(run_dir / n, f"{rc['artifact_dir']}/{sub}/{n}", project.paths)["path"]
                for n in receipt["outputs"]]
    return {"ready": True, "analysis": analysis, "run": receipt["run"],
            "journal_index": receipt["journal"]["index"], "host_type": ex.get("host_type"),
            "promoted": promoted}


SURFACE_OUTPUTS = ("surface_grid.csv", "count_stability.csv", "tier_range_summary.csv",
                   "building_hand_mr_microsoft.parquet", "building_hand_mr_detector_final.parquet",
                   "building_tiers_microsoft.csv", "building_tiers_detector_final.csv",
                   "building_tiers.json", "surface_checks.json")


def promote_surfaces(project: Project, run_ref: str,
                     config: str = S.RECOMPUTE_CONFIG) -> dict[str, Any]:
    """Laptop: promote a pulled label-free ``hand-mr`` run (prep check must be identical)."""
    from nakkhu.artifacts import promote
    from nakkhu.runs import journal_status

    rc = S.load_recompute(project, config)
    stage, _, rid = run_ref.partition("/")
    run_dir = project.paths.runs / stage / rid
    if journal_status(run_dir) != "complete":
        raise S.MainRiverError(f"{run_ref} is {journal_status(run_dir)}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("prep_check_identical"):
        raise S.MainRiverError(f"prep check failed: {summary.get('prep_check_failures')}")
    for s in summary["surfaces"].values():
        if s.get("raster"):
            f = project.paths.resolve(s["raster"])
            if not f.is_file() or sha256_file(f) != s["raster_sha256"]:
                raise S.MainRiverError(f"raster {s['raster']} missing or altered (pull it)")
    promoted = [promote(run_dir / n, f"{rc['artifact_dir']}/hand/{n}", project.paths)["path"]
                for n in SURFACE_OUTPUTS]
    return {"ready": True, "run": run_ref, "promoted": promoted}


__all__ = ["reserve", "execute", "promote_analysis", "promote_surfaces", "verify_receipt",
           "BUNDLE_MANIFEST"]
