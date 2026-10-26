"""``nakkhu uncertainty ...`` (plan 010). ``check``, ``spec`` and ``run --dry-run`` never touch
test data; ``run`` is the one journalled test access for U1-U3 and D4."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project


def _check(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.uncertainty.spec import prerequisites

    problems = prerequisites(project)
    for p in problems:
        print(f"PEND  {p}")
    print("ready" if not problems else f"{len(problems)} prerequisite(s) missing")
    return 0 if not problems else 1


def _spec(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.uncertainty.spec import load_spec

    print(json.dumps(load_spec(project), indent=1, default=str))
    return 0


def _run(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.uncertainty.run import run_uncertainty

    result = run_uncertainty(project, dry_run=args.dry_run)
    for p in result["problems"]:
        print(f"REFUSE  {p}")
    if not result["ran"]:
        print("ready (dry run: nothing read, nothing journalled)" if result["ready"]
              else f"refused: {len(result['problems'])} precondition(s) not met")
        return 0 if result["ready"] else 1
    print(json.dumps({k: result[k] for k in ("run", "artifacts")}, indent=1))
    if result["test_access"]["flagged"]:
        print("FLAG: journalled before with a different config", file=sys.stderr)
    return 0


def register(commands: Any) -> None:
    unc = commands.add_parser("uncertainty", help="detection-to-exposure uncertainty U1-U3, "
                              "D4 (plan 010; spec frozen in prereg v5)")
    sub = unc.add_subparsers(dest="uncertainty_command", required=True, metavar="ACTION")
    check = sub.add_parser("check", help="list missing prerequisites (no test access)")
    check.set_defaults(func=_check)
    spec = sub.add_parser("spec", help="print configs/uncertainty/uncertainty.yaml")
    spec.set_defaults(func=_spec)
    run = sub.add_parser("run", help="U1-U3 + D4: ONE journalled test access; refuses unless "
                         "prereg verify passes, the hold (runs/eval/test_hold.yaml) is off, "
                         "it never ran, and every input exists")
    run.add_argument("--dry-run", action="store_true",
                     help="check every precondition only (reads no answer, journals nothing)")
    run.set_defaults(func=_run)
