"""``nakkhu mainriver``: the main-river HAND recompute (plans/recompute_main_river.md).

- ``surfaces`` (remote host; label-free): the main-river ensemble rasters, building tables,
  tier counts / ranges and tier export, in a ``hand-mr`` run.
- ``promote-surfaces --run hand-mr/<id>`` (laptop): promote them to artifacts/mainriver/hand/.
- ``check-inputs`` (remote host; label-free): load every input with the answers withheld.
- ``reserve --analysis exposure|event|roc`` (laptop): journal entry + receipt.
- ``run --analysis A --receipt R`` (remote host): execute from the receipt.
- ``promote --analysis A`` (laptop): verify the pulled run and promote."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project


def _emit(out: dict[str, Any]) -> int:
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get("ready", out.get("ok", True)) else 1


def _guarded(fn: Any) -> Any:
    def wrapper(project: Project, args: argparse.Namespace) -> int:
        from nakkhu.followup.spec import FollowupError
        from nakkhu.mainriver.spec import MainRiverError

        try:
            return fn(project, args)
        except (MainRiverError, FollowupError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return wrapper


@_guarded
def _surfaces(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver.surfaces import run_surfaces

    run, summary = run_surfaces(project, args.config)
    return _emit({"ok": summary["prep_check_identical"], "run": run.ref,
                  "prep_check_failures": summary["prep_check_failures"]})


@_guarded
def _promote_surfaces(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver.remote import promote_surfaces

    return _emit(promote_surfaces(project, args.run))


@_guarded
def _check(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver import spec as S
    from nakkhu.mainriver.analyses import check_inputs

    rc = S.load_recompute(project)
    ens = S.load_ensemble(project, rc["ensemble_config"])
    stage, _, rid = args.hand_run.partition("/")
    return _emit(check_inputs(project, rc, ens, project.paths.runs / stage / rid))


@_guarded
def _reserve(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver.remote import reserve

    return _emit(reserve(project, args.analysis, retry=args.retry, dry_run=args.dry_run))


@_guarded
def _run(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver import remote

    if args.dry_run:
        receipt, rsha = remote.verify_receipt(project, args.receipt, args.analysis)
        return _emit({"ready": True, "run": receipt["run"], "receipt_sha256": rsha})
    return _emit(remote.execute(project, args.receipt, args.analysis))


@_guarded
def _promote(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.mainriver.remote import promote_analysis

    return _emit(promote_analysis(project, args.analysis))


def register(commands: Any) -> None:
    from nakkhu.mainriver.spec import ENSEMBLE_CONFIG

    p = commands.add_parser("mainriver", help="main-river HAND recompute (label-free surfaces; "
                                              "reserved exposure / event analyses)")
    sub = p.add_subparsers(dest="mainriver_command", required=True)
    s = sub.add_parser("surfaces", help="remote, label-free: main-river ensemble surfaces")
    s.add_argument("--config", default=ENSEMBLE_CONFIG)
    s.set_defaults(func=_surfaces)
    ps = sub.add_parser("promote-surfaces", help="laptop: promote a pulled hand-mr run")
    ps.add_argument("--run", required=True)
    ps.set_defaults(func=_promote_surfaces)
    c = sub.add_parser("check-inputs", help="remote, label-free: load inputs, answers withheld")
    c.add_argument("--hand-run", dest="hand_run", required=True)
    c.set_defaults(func=_check)
    r = sub.add_parser("reserve", help="laptop: journal entry + receipt")
    r.add_argument("--analysis", required=True, choices=["exposure", "event", "roc"])
    r.add_argument("--retry", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=_reserve)
    x = sub.add_parser("run", help="remote: execute a reserved analysis from its receipt")
    x.add_argument("--analysis", required=True, choices=["exposure", "event", "roc"])
    x.add_argument("--receipt", required=True)
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=_run)
    pr = sub.add_parser("promote", help="laptop: verify a pulled analysis run and promote")
    pr.add_argument("--analysis", required=True, choices=["exposure", "event", "roc"])
    pr.set_defaults(func=_promote)
