"""``nakkhu aoi ...`` subcommands (plan 002 provenance). Heavy imports happen in the handlers."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from nakkhu.aoi import BUILD_CONFIG, PRODUCTS
from nakkhu.config import Project


def _metric(cmp: dict[str, Any]) -> str:
    if "iou" in cmp:
        return (f"area {cmp['area_km2_rebuilt']:.4f} vs {cmp['area_km2_registered']:.4f} km2, "
                f"IoU {cmp['iou']:.8f}, Hausdorff {cmp['hausdorff_m']:.3g} m, "
                f"coordinates identical {cmp['coordinates_identical']}")
    if "elements_identical" in cmp:
        return (f"{cmp['n_rebuilt']} vs {cmp['n_registered']} ways, elements identical "
                f"{cmp['elements_identical']}, only rebuilt {cmp['only_rebuilt']}, only "
                f"registered {cmp['only_registered']}, tags/geometry differ "
                f"{len(cmp['tags_differ'])}/{len(cmp['geometry_differs'])}")
    if "ways" in cmp:
        same = sum(w.get("coordinates_identical", False) for w in cmp["ways"])
        return (f"{len(cmp['ways'])} ways ({same} coordinate-identical), length "
                f"{cmp['total_length_km_rebuilt']} vs {cmp['total_length_km_registered']} km, "
                f"max Hausdorff {cmp['max_hausdorff_m']:.3g} m")
    return (f"{cmp['n_rebuilt']} vs {cmp['n_registered']} points, max shift "
            f"{cmp['max_shift_m']:.3g} m")


def _aoi_build(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.aoi.build import AoiBuildError, build

    try:
        run, summary = build(project, args.only or PRODUCTS, args.aoi_config, args.upstream)
    except AoiBuildError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for product, result in summary["products"].items():
        state = "MATCH" if result["comparison"]["match"] else "DIFF "
        print(f"{state} {product}: {_metric(result['comparison'])}")
        for way in (result["build"].get("unlisted_same_name") or []):
            print(f"      unlisted way {way['id']} {way['name']!r} (same name, not configured)")
    print(f"run {run.ref} (upstream {summary['upstream']}); registered files unchanged")
    return 0 if summary["all_match"] else 1


def _aoi_reach(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.aoi.reach import ARTIFACT, ReachError, run_reach

    try:
        run, summary = run_reach(project, args.reach_config)
    except ReachError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for key, reach in summary["reaches"].items():
        extra = ""
        if "share_range" in reach:
            low, high = reach["share_range"]
            extra = f" (end +-{reach['end_uncertainty_m']:g} m: {low:.4f}-{high:.4f})"
        print(f"{key}: {reach['inside_m'] / 1000:.3f} of {reach['length_m'] / 1000:.3f} km "
              f"inside, share {reach['share']:.4f}{extra}")
    print(f"run {run.ref}; promoted artifacts/{ARTIFACT}")
    return 0


def register(commands: Any) -> None:
    aoi = commands.add_parser("aoi", help="derived AOI files (plan 002 provenance)")
    sub = aoi.add_subparsers(dest="aoi_command", required=True, metavar="ACTION")
    build = sub.add_parser("build", help="rebuild AOI files into runs/aoi/ and compare them with "
                           "the registered files in configs/aoi/")
    build.add_argument("--only", nargs="+", choices=PRODUCTS, help="products (default: all; "
                       "upstream products are rebuilt too)")
    build.add_argument("--upstream", choices=("rebuilt", "registered"), default="rebuilt",
                       help="inputs of downstream products: the rebuilt chain (default) or the "
                       "registered files (isolates one step)")
    build.add_argument("--config", dest="aoi_config", default=BUILD_CONFIG,
                       help=f"build config (default: {BUILD_CONFIG})")
    build.set_defaults(func=_aoi_build)
    reach = sub.add_parser("reach", help="share of the 2024 flooded reach inside the analysis "
                           "extent; promotes artifacts/aoi/reach_share.json")
    reach.add_argument("--config", dest="reach_config", default="configs/aoi/reach.yaml",
                       help="reach config (default: configs/aoi/reach.yaml)")
    reach.set_defaults(func=_aoi_reach)
