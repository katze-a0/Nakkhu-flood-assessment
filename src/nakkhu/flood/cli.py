"""``nakkhu flood ...`` subcommands (plan 009). Heavy imports happen inside the handlers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project

CHANGE_CONFIG = "configs/flood/change.yaml"
REFERENCE_INVENTORIES = ("google_v3", "google_v3_uncut", "microsoft", "osm")


def _change(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.flood.pipeline import FloodError, run_change

    try:
        run, summary = run_change(project, args.flood_config, inventories=args.inventory,
                                  promote_outputs=not args.no_promote)
    except FloodError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, info in summary["by_inventory"].items():
        print(f"{name}: {info['n_buildings']} buildings, {info['n_observable']} observable, "
              f"{info['n_positive']} positive, {info['n_not_observable']} not observable")
    for name in summary.get("pending_inventories", []):
        print(f"PEND  {name}: inventory file missing")
    print(json.dumps({"coregistration_verify": summary["coregistration_verify"],
                      "primary_strips": summary["primary_strips"],
                      "timings_s": summary["timings_s"]}))
    print(f"run {run.ref}")
    return 0


def _validate(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.flood.validate import prerequisites, rerun_competitors, run_validate

    if args.rerun_competitors:
        result = rerun_competitors(project, args.rerun_competitors, reason=args.reason or "",
                                   dry_run=args.dry_run)
        keys = ("ready", "problems", "dry_run", "source_run", "inventories", "n_boot", "seed",
                "source_journal_entry", "journal_ref", "reason", "run", "journal_entry",
                "reproduction", "predictors")
        print(json.dumps({k: result[k] for k in keys if k in result}, indent=1, default=str))
        if not result["ready"]:
            return 1
        return 0 if result.get("dry_run") or result["reproduction"].get("reproduced") else 2
    if args.dry_run or args.reason:
        print("error: --dry-run and --reason go with --rerun-competitors", file=sys.stderr)
        return 1
    names: list[str] = []
    for item in args.inventory:
        if item == "final":
            names.append("detector_final")
        elif item == "all":
            names += ["detector_final", *REFERENCE_INVENTORIES]
        else:
            names.append(item)
    names = list(dict.fromkeys(names))
    if args.check:
        from nakkhu.config import load_config

        sweep = load_config(project.paths.resolve("configs/hand/sweep.yaml")).values
        final = project.paths.resolve(sweep["inventories"]["detector_final"]["path"])
        problems = prerequisites(project, final if "detector_final" in names else None,
                                 names)
        for p in problems:
            print(f"PEND  {p}")
        print("ready" if not problems else f"{len(problems)} prerequisite(s) missing")
        return 0 if not problems else 1
    try:
        run, summary = run_validate(project, names, n_boot=args.n_boot)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"run {run.ref}; test access flagged: {summary['test_access']['flagged']}")
    return 0


def _thin_curves(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.flood.curves import thin_run

    try:
        run, summary = thin_run(project, args.run, promote_outputs=not args.no_promote)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, records in summary["curves"].items():
        for r in records:
            metric = ("auc_full", "auc_thin") if "auc_full" in r else ("ap_full", "ap_thin")
            print(f"{name} {r['inventory']}/{r['reference']}: {r['points']} -> {r['kept']} "
                  f"(tol {r['tolerance']:g}); {metric[0]} {r[metric[0]]:.6f} "
                  f"{metric[1]} {r[metric[1]]:.6f}")
    print(f"run {run.ref}")
    return 0


def register(commands: Any) -> None:
    flood = commands.add_parser("flood", help="change map and event validation (plan 009)")
    sub = flood.add_subparsers(dest="flood_command", required=True, metavar="ACTION")
    change = sub.add_parser("change", help="build the flood-trace / damage change map")
    change.add_argument("--config", dest="flood_config", default=CHANGE_CONFIG,
                        help=f"change-map config (default: {CHANGE_CONFIG})")
    change.add_argument("--inventory", nargs="+",
                        help="inventory ids to score (default: buildings.inventories)")
    change.add_argument("--no-promote", action="store_true",
                        help="keep the outputs in the run only")
    change.set_defaults(func=_change)
    val = sub.add_parser("validate", help="V1-V4 and H2-H6 against the change map and audit "
                         "(journalled test access; refuses until prereg v2 and sitting 3)")
    val.add_argument("--inventory", nargs="+", default=["final"],
                     help="final, all, or inventory ids from configs/hand/sweep.yaml")
    val.add_argument("--configs", default="all", choices=("all",),
                     help="HAND surfaces scored (every column of the building HAND table)")
    val.add_argument("--n-boot", type=int, default=2000)
    val.add_argument("--check", action="store_true",
                     help="list missing prerequisites only (no test access)")
    val.add_argument("--rerun-competitors", metavar="RUN",
                     help="recompute only the V4 competitor table of validate run RUN "
                          "(validation/<run_id>) with its inventories, references, n_boot and "
                          "seed (plans/competitor_auc_bug.md); one journalled test access, "
                          "once per source run; writes a new validation-competitors run and "
                          "promotes nothing")
    val.add_argument("--reason", help="why the competitors are re-run (journalled; required "
                                      "with --rerun-competitors)")
    val.add_argument("--dry-run", action="store_true",
                     help="with --rerun-competitors: check and load predictors only (no "
                          "reference, no answer, no journal entry)")
    val.set_defaults(func=_validate)
    thin = sub.add_parser("thin-curves",
                          help="thinned ROC / PR curves of a validate run (reads only that "
                               "run's curve files; no test access)")
    thin.add_argument("--run", required=True, help="the validate run, e.g. validation/<run_id>")
    thin.add_argument("--no-promote", action="store_true",
                      help="keep the outputs in the run only")
    thin.set_defaults(func=_thin_curves)
