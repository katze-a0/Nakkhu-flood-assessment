"""``nakkhu hand ...`` subcommands (plan 008). Heavy imports happen inside the handlers."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from nakkhu.config import Project

SWEEP_CONFIG = "configs/hand/sweep.yaml"


def _print_summary(run: Any, summary: dict[str, Any]) -> None:
    print(f"run {run.ref}: {summary.get('surfaces_done', 0)} surfaces done, "
          f"{summary.get('surfaces_skipped', 0)} skipped; primary {summary.get('primary')}")
    for dem, info in (summary.get("dems") or {}).items():
        print(f"  {dem}: {info['status']}" + (f" ({info['reason']})" if info["reason"] else ""))


def _hand_sweep(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.hydro.sweep import SweepError, run_sweep

    try:
        run, summary = run_sweep(project, args.hand_config)
    except SweepError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print_summary(run, summary)
    return 0


def _hand_attribute(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.hydro.sweep import SweepError, attribute, load_settings

    settings, _ = load_settings(project, args.hand_config)
    names = (list(settings.v["inventories"]) if args.inventory == ["all"]
             else list(dict.fromkeys(args.inventory)))
    unknown = [n for n in names if n not in settings.v["inventories"]]
    if unknown:
        print(f"error: unknown inventory {unknown}", file=sys.stderr)
        return 2
    available, pending = [], []
    for name in names:
        entry = settings.v["inventories"][name]
        (available if settings.resolve(entry["path"]).is_file() else pending).append(name)
    for name in pending:
        print(f"PEND  {name}: no file at {settings.v['inventories'][name]['path']}")
    if not available:
        return 1
    try:
        run, summary = attribute(project, available, args.sweep, args.hand_config)
    except SweepError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, info in summary["inventories"].items():
        print(f"DONE  {name}: {info['n_buildings']} buildings")
    print(f"run {run.ref} (sweep {summary['sweep_run']})")
    return 0


def _hand_inventories(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.hydro.detections import DetectionInventoryError, export

    try:
        written, pending = export(project, args.inventory or None, args.hand_config)
    except DetectionInventoryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for name, meta in written.items():
        print(f"DONE  {name}: {meta['n_boxes_kept']} boxes at conf >= {meta['threshold']:.4f} "
              f"from {meta['infer_test']}")
    for reason in pending.values():
        print(f"PEND  {reason}")
    return 0 if written or not pending else 1


def _hand_tier_export(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.hydro.sweep import SweepError
    from nakkhu.hydro.tier_export import TierExportError, export_run

    try:
        run, summary = export_run(project, list(dict.fromkeys(args.inventory)),
                                  args.attribute, args.hand_config)
    except (TierExportError, SweepError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, info in summary["inventories"].items():
        match = ", ".join(f"{k} {v}" for k, v in info["counts_match"].items())
        print(f"DONE  {name}: {info['rows']} buildings from {info['attribute_run']} "
              f"(counts match: {match})")
    print(f"run {run.ref} ({summary['hand_column']})")
    return 0


def _hand_dem_diagnostics(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.hydro.dem_diagnostics import DiagnosticsError, run_diagnostics

    try:
        run, summary = run_diagnostics(project, args.hand_config, args.sweep)
    except DiagnosticsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"run {run.ref} (sweep {summary['sweep_run']}; label-free; DEMs "
          f"{', '.join(summary['dems'])})")
    return 0


def register(commands: Any) -> None:
    hand = commands.add_parser("hand", help="Python HAND and the sensitivity sweep (plan 008)")
    sub = hand.add_subparsers(dest="hand_command", required=True, metavar="ACTION")
    run = sub.add_parser("run", help="one configuration, e.g. configs/hand/primary.yaml")
    run.add_argument("--config", dest="hand_config", required=True,
                     help="single-configuration YAML (with `base:`)")
    run.set_defaults(func=_hand_sweep)
    sweep = sub.add_parser("sweep", help="every available DEM x conditioning x threshold")
    sweep.add_argument("--config", dest="hand_config", default=SWEEP_CONFIG,
                       help=f"sweep config (default: {SWEEP_CONFIG})")
    sweep.set_defaults(func=_hand_sweep)
    attr = sub.add_parser("attribute", help="building HAND per inventory, tier stability")
    attr.add_argument("--inventory", required=True, nargs="+",
                      help="inventory ids from the sweep config (one run), or all")
    attr.add_argument("--sweep", help="hand/<run_id> to use (default: latest complete sweep)")
    attr.add_argument("--config", dest="hand_config", default=SWEEP_CONFIG,
                      help=f"sweep config (default: {SWEEP_CONFIG})")
    attr.set_defaults(func=_hand_attribute)
    inv = sub.add_parser("inventories", help="write single-seed detector inventories (sweep "
                         "entries with `detections`) from their georeferenced test runs")
    inv.add_argument("--inventory", nargs="+", help="ids (default: every detections entry)")
    inv.add_argument("--config", dest="hand_config", default=SWEEP_CONFIG,
                     help=f"sweep config (default: {SWEEP_CONFIG})")
    inv.set_defaults(func=_hand_inventories)
    tiers = sub.add_parser("tier-export", help="per-building HAND tier table (id, centroid x/y, "
                           "primary HAND, tier) from a hand-attribute run; descriptive, no "
                           "test access; counts checked against the promoted tier counts")
    tiers.add_argument("--inventory", required=True, nargs="+",
                       help="inventory ids from the sweep config, e.g. microsoft")
    tiers.add_argument("--attribute", nargs="+",
                       help="hand-attribute/<run_id> runs to read (default: the newest "
                            "complete run holding each inventory's HAND table)")
    tiers.add_argument("--config", dest="hand_config", default=SWEEP_CONFIG,
                       help=f"sweep config (default: {SWEEP_CONFIG})")
    tiers.set_defaults(func=_hand_tier_export)
    diag = sub.add_parser("dem-diagnostics", help="label-free DEM diagnostics of a derived sweep "
                          "(elevation / HAND differences at building centroids, drains-to-main "
                          "shares, voids and artefacts); reads no label")
    diag.add_argument("--config", dest="hand_config", default="configs/hand/sweep_dems.yaml",
                      help="sweep config with a `diagnostics` section")
    diag.add_argument("--sweep", help="<stage>/<run_id> (default: latest complete run of the "
                      "config's stage)")
    diag.set_defaults(func=_hand_dem_diagnostics)
