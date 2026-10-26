"""``nakkhu followup`` (plan 014): ``prep`` (label-free), ``reserve`` (laptop: journal entry +
receipt), ``run --receipt`` (remote host only), ``promote`` / ``merge`` (laptop), and the
label-free ``check-env`` / ``calibrate`` of a remote environment."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project

ALIASES = {"explore-p2": "nakkhu.followup.explore:EXPLORE_P2"}


def _test(name: str) -> str:
    """CLI alias of a test id (the Phase-2 explore note contains spaces)."""
    if name in ALIASES:
        from nakkhu.followup.explore import EXPLORE_P2

        return EXPLORE_P2
    return name


def _emit(out: dict[str, Any]) -> int:
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get("ready", out.get("ok", True)) else 1


def _guarded(fn: Any) -> Any:
    def wrapper(project: Project, args: argparse.Namespace) -> int:
        from nakkhu.followup.spec import FollowupError

        try:
            return fn(project, args)
        except FollowupError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return wrapper


@_guarded
def _prep(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.prep import run_prep

    run, summary = run_prep(project, args.prep_config)
    print(json.dumps({"run": run.ref, "files": len(summary["manifest"]["files"]),
                      "stored_surfaces_identical": summary["checks"][
                          "stored_surfaces_identical"]}, indent=1))
    return 0


@_guarded
def _reserve(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.remote import reserve

    return _emit(reserve(project, _test(args.test), retry=args.retry, dry_run=args.dry_run,
                         config=args.prep_config))


@_guarded
def _run(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup import remote

    args.test = _test(args.test)
    if args.receipt is None:
        if args.dry_run:
            return _emit(remote.reserve(project, args.test, retry=args.retry, dry_run=True,
                                        config=args.prep_config))
        print("error: refused without --receipt. Declared tests are reserved on the laptop "
              "(`nakkhu followup reserve --test <id>`) and executed only on a remote host "
              "from a bundle with that receipt.", file=sys.stderr)
        return 1
    if args.dry_run:
        receipt, rsha = remote.verify_receipt(project, args.receipt, args.test)
        return _emit({"ready": True, "test": args.test, "dry_run": True,
                      "run": receipt["run"], "receipt_sha256": rsha,
                      "host_type": remote.host_type()})
    return _emit(remote.execute(project, args.receipt, args.test, jobs=args.jobs,
                                shard=args.shard))


@_guarded
def _promote(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.remote import promote_test

    return _emit(promote_test(project, _test(args.test), run=args.run))


@_guarded
def _merge(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.remote import ReceiptError, merge_p1_6

    if args.test != "P1-6":
        raise ReceiptError("only P1-6 is sharded")
    return _emit(merge_p1_6(project, run=args.run))


@_guarded
def _check_env(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.remote import check_env

    return _emit(check_env(project, args.commit))


@_guarded
def _calibrate(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup.remote import calibrate

    return _emit(calibrate(project, fits=args.fits))


@_guarded
def _p2(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.followup import phase2 as P2

    action = args.p2_action
    if action == "sites":
        run, summary = P2.build_sites(project, args.sites_config)
        return _emit({"run": run.ref, "sites": summary["sites"]})
    if action == "osm":
        return _emit(P2.osm_footprints(project, args.sites_config))
    sites = args.site or list(P2.load_sites(project, args.sites_config)["sites"])
    if action == "predictors":
        run, summary = P2.site_predictors(project, sites, args.sites_config)
        return _emit({"run": run.ref, "sites": summary["sites"]})
    out = {}
    for site in sites:
        if action == "mosaic":
            out[site] = P2.build_mosaics(project, site, args.sites_config, workers=args.jobs)
        elif action == "register":
            out[site] = P2.register_site(project, site, args.sites_config, workers=args.jobs)
        elif action == "change":
            run, summary = P2.change_site(project, site, args.sites_config)
            out[site] = {"run": run.ref, "coregistration_verify": summary["coregistration_verify"]}
        elif action == "coreg":
            run, summary = P2.coreg_gate(project, site, args.sites_config)
            out[site] = {"run": run.ref, "coregistration": summary["coregistration"],
                         "blocks": summary["blocks"]}
    return _emit(out)


def register(commands: Any) -> None:
    from nakkhu.followup.spec import PREP_CONFIG

    parser = commands.add_parser("followup", help="plan 014 follow-up tests (prereg_v9_a3)")
    sub = parser.add_subparsers(dest="followup_command", required=True)
    prep = sub.add_parser("prep", help="label-free prep tables (no test access)")
    prep.add_argument("--prep-config", dest="prep_config", default=PREP_CONFIG)
    prep.set_defaults(func=_prep)

    res = sub.add_parser("reserve", help="laptop: refusal checks, journal entry and receipt")
    res.add_argument("--test", required=True, help="declared test id, e.g. P1-3")
    res.add_argument("--retry", action="store_true",
                     help="the one disclosed retry of a run that failed before any metric")
    res.add_argument("--dry-run", action="store_true",
                     help="check every prerequisite and stop before the journal entry")
    res.add_argument("--prep-config", dest="prep_config", default=PREP_CONFIG)
    res.set_defaults(func=_reserve)

    run = sub.add_parser("run", help="remote host: execute a reserved test from its receipt")
    run.add_argument("--test", required=True, help="declared test id, e.g. P1-3")
    run.add_argument("--receipt", default=None, help="receipt JSON written by `reserve`")
    run.add_argument("--shard", default=None, help="P1-6 only: i/n, a-b/n or i,j/n")
    run.add_argument("--retry", action="store_true", help="(dry run without receipt only)")
    run.add_argument("--dry-run", action="store_true",
                     help="without --receipt: label-free prerequisites; with it: verify only")
    run.add_argument("--jobs", type=int, default=1, help="worker processes (P1-6 units)")
    run.add_argument("--prep-config", dest="prep_config", default=PREP_CONFIG)
    run.set_defaults(func=_run)

    pro = sub.add_parser("promote", help="laptop: verify a pulled run and promote its outputs")
    pro.add_argument("--test", required=True)
    pro.add_argument("--run", default=None, help="run ref (default: the test's latest receipt)")
    pro.set_defaults(func=_promote)

    mer = sub.add_parser("merge", help="laptop: verify every P1-6 unit, summarise, promote")
    mer.add_argument("--test", required=True)
    mer.add_argument("--run", default=None)
    mer.set_defaults(func=_merge)

    env = sub.add_parser("check-env", help="label-free check of a bundle environment")
    env.add_argument("--commit", default=None, help="expected code commit")
    env.set_defaults(func=_check_env)

    p2 = sub.add_parser("p2", help="Phase-2 label-free site preparation (sites.yaml)")
    p2.add_argument("p2_action", choices=("sites", "osm", "mosaic", "register", "coreg",
                                          "predictors", "change"),
                    help="change builds Phase-2 labels and refuses until prereg_v9_a4 is frozen")
    p2.add_argument("--site", nargs="+", default=None, help="site letters (default: all)")
    p2.add_argument("--jobs", type=int, default=2, help="worker processes")
    p2.add_argument("--sites-config", dest="sites_config",
                    default="configs/followup/phase2/sites.yaml")
    p2.set_defaults(func=_p2)

    cal = sub.add_parser("calibrate", help="label-free timing of P1-6 fits (random target)")
    cal.add_argument("--fits", type=int, default=6)
    cal.set_defaults(func=_calibrate)
