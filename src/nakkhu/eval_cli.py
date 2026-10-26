"""``nakkhu split``, ``nakkhu prereg`` and ``nakkhu eval`` subcommands (plan 005)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project


def _split_build(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.splits import SplitError, build

    try:
        result = build(project, check=args.check, replace=args.replace)
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    if args.check and not result.get("identical_to_frozen"):
        print("rebuild differs from configs/splits/split_v1.yaml", file=sys.stderr)
        return 1
    return 0


def _prereg_freeze(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.evaluate.guard import GuardError, freeze

    try:
        if args.amend and args.replace:
            print("error: --amend and --replace are exclusive", file=sys.stderr)
            return 2
        result = freeze(project, replace=args.replace, changelog=args.changelog,
                        amend=args.amend, previous_file=args.previous_file,
                        allow=args.allow or ())
    except GuardError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


def _prereg_addendum(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.evaluate.addendum import AddendumError, freeze_addendum

    try:
        result = freeze_addendum(project, args.file, plan=args.plan, changelog=args.changelog)
    except AddendumError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


def _prereg_verify(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.evaluate.guard import read_test_journal, verify

    state = verify(project)
    state["test_journal_entries"] = len(read_test_journal(project))
    print(json.dumps(state, indent=1))
    return 1 if state["problems"] else 0


def _eval(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.evaluate.guard import GuardError
    from nakkhu.evaluate.score import EvalError, compare_runs, evaluate_run

    try:
        if args.compare:
            result: dict[str, Any] = compare_runs(project, *args.compare)
        else:
            if not args.run or not args.split:
                print("error: --run and --split are required (or --compare A B)",
                      file=sys.stderr)
                return 2
            result = evaluate_run(project, args.run, args.split,
                                  threshold_from=args.threshold_from, fixed=args.fixed,
                                  ap_ci=not args.no_ap_ci)
    except (GuardError, EvalError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str))
    if result.get("flagged"):
        print("FLAG: this run was scored on test before with a different eval config",
              file=sys.stderr)
    return 0


def register(commands: Any) -> None:
    split = commands.add_parser("split", help="spatial-block train/val/test split (plan 005)")
    sub = split.add_subparsers(dest="split_command", required=True, metavar="ACTION")
    build = sub.add_parser("build", help="build configs/splits/split_v1.yaml from "
                           "configs/splits/split.yaml and promote its summary and blocks")
    build.add_argument("--check", action="store_true",
                       help="rebuild into a run and compare byte for byte with the frozen file")
    build.add_argument("--replace", action="store_true",
                       help="overwrite a different split file (only before any test scoring)")
    build.set_defaults(func=_split_build)

    prereg = commands.add_parser("prereg", help="pre-registration freeze (plan 005)")
    psub = prereg.add_subparsers(dest="prereg_command", required=True, metavar="ACTION")
    freeze = psub.add_parser("freeze", help="hash configs/prereg.yaml into "
                             "artifacts/prereg/<version>.sha256")
    freeze.add_argument("--replace", action="store_true",
                        help="re-freeze a changed prereg (refused once test was accessed)")
    freeze.add_argument("--changelog", help="markdown file promoted as prereg/changelog.md "
                        "(appended to it with --amend)")
    freeze.add_argument("--amend", action="store_true",
                        help="one audited amendment after test access (plan 013): needs "
                        "--previous-file, --allow and --changelog")
    freeze.add_argument("--previous-file",
                        help="copy of the frozen previous configs/prereg.yaml (hash-checked; "
                        "promoted as prereg/<previous>.yaml)")
    freeze.add_argument("--allow", nargs="+", metavar="PREFIX",
                        help="dotted prefixes the amendment may change (locked sections are "
                        "refused regardless)")
    freeze.set_defaults(func=_prereg_freeze)
    addendum = psub.add_parser("addendum", help="freeze a follow-up addendum "
                               "configs/prereg_addenda/<id>.yaml (plan 014; allowed after test "
                               "access, never changes configs/prereg.yaml)")
    addendum.add_argument("--file", required=True,
                          help="configs/prereg_addenda/<id>.yaml (id like prereg_v9_a1)")
    addendum.add_argument("--plan", help="plan markdown frozen as prereg/addenda/<id>.plan.md")
    addendum.add_argument("--changelog", help="markdown appended to prereg/changelog.md "
                          "(must name the addendum id)")
    addendum.set_defaults(func=_prereg_addendum)
    verify = psub.add_parser("verify", help="check prereg hash, split hash, mirrored configs "
                             "and frozen addenda")
    verify.set_defaults(func=_prereg_verify)

    ev = commands.add_parser("eval", help="score a run against aligned OSM on val or test "
                             "blocks (test only with a frozen prereg; journalled)")
    ev.add_argument("--run", help="run with detections.gpkg, <stage>/<run_id> (default stage "
                    "infer)")
    ev.add_argument("--split", choices=("val", "test"))
    ev.add_argument("--threshold-from", help="val eval run whose F1-max threshold is applied")
    ev.add_argument("--fixed", action="store_true",
                    help="score every prediction (fixed footprint inventories)")
    ev.add_argument("--no-ap-ci", action="store_true", help="skip the AP50 bootstrap")
    ev.add_argument("--compare", nargs=2, metavar=("EVAL_A", "EVAL_B"),
                    help="paired block bootstrap of two eval runs (A - B)")
    ev.set_defaults(func=_eval)
