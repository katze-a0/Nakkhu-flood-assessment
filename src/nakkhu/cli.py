"""``nakkhu`` command line. Later plans register their own subcommands in :func:`build_parser`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from nakkhu.artifacts import PromotionError, load_index, promote
from nakkhu.config import PROJECT_CONFIG, ConfigError, Project, load_project
from nakkhu.manifest import ManifestError, verify_manifest
from nakkhu.paths import PathError
from nakkhu.runs import list_runs

DEFAULT_MANIFEST = "configs/data/manifest.yaml"


def _data_verify(project: Project, args: argparse.Namespace) -> int:
    paths_section = project.config.values.get("paths") or {}
    manifest = args.manifest or paths_section.get("manifest", DEFAULT_MANIFEST)
    results = verify_manifest(project.paths.resolve(manifest), project.paths)
    for result in results:
        if not result.ok:
            print(f"FAIL  {result.id}: " + "; ".join(result.problems))
        elif result.pending:
            print(f"PEND  {result.id}")
        else:
            print(f"OK    {result.id}")
    failed = sum(not result.ok for result in results)
    pending = sum(result.ok and result.pending for result in results)
    print(f"{len(results)} entries, {failed} failed, {pending} pending")
    return 1 if failed else 0


def _data_fetch(project: Project, args: argparse.Namespace) -> int:
    from nakkhu import acquire

    sources = acquire.load_sources(project)
    ids = acquire.source_ids(sources) if args.all else args.ids
    unknown = [sid for sid in ids if sid not in sources or sid.startswith("_")]
    if unknown or not ids:
        print(f"error: unknown or missing source ids: {unknown or '(none given)'}",
              file=sys.stderr)
        return 2
    if args.dry_run:
        rows, total = acquire.dry_run(project, ids)
        for sid, row in rows:
            size = "unknown" if row.size is None else f"{row.size:,}"
            licence = sources[sid].get("licence", "")
            kind = "partial" if row.partial else "full"
            print(f"{sid}\t{kind}\t{size}\t{licence}\t{row.url}\t{row.note}")
        full = [row for _, row in rows if not row.partial]
        unknown_sizes = sum(row.size is None for row in full)
        print(f"total {total:,} bytes ({total / 1e9:.2f} GB) in {len(full)} full downloads"
              + (f" ({unknown_sizes} of unknown size)" if unknown_sizes else "")
              + f"; plus {len(rows) - len(full)} windowed/range reads (only the clip is"
              " transferred)")
        return 0
    failed = 0
    for sid in ids:
        try:
            entry = acquire.fetch(project, sid, force=args.force)
        except Exception as error:  # report and continue with the next source
            failed += 1
            print(f"FAIL  {sid}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        state = "pending" if entry.get("status") == "pending" else f"{entry['bytes']:,} bytes"
        print(f"DONE  {sid}: {state}")
    return 1 if failed else 0


def _data_freeze(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.acquire.freeze import FreezeError, freeze

    try:
        promoted = freeze(project)
    except FreezeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for entry in promoted:
        print(f"promoted artifacts/{entry['path']} ({entry['bytes']} bytes)")
    return 0


def _runs_list(project: Project, args: argparse.Namespace) -> int:
    runs = list_runs(project.paths, args.stage)
    for run in runs:
        print(f"{run.ref}\t{run.status}")
    print(f"{len(runs)} runs")
    return 0


def _artifacts_promote(project: Project, args: argparse.Namespace) -> int:
    entry = promote(args.source, args.dest, project.paths, replace=args.replace)
    print(f"promoted {entry['source_run']}/{entry['source_file']} -> artifacts/{entry['path']}")
    print(f"sha256 {entry['sha256']}")
    return 0


def _artifacts_list(project: Project, args: argparse.Namespace) -> int:
    entries = load_index(project.paths)["artifacts"]
    for entry in entries:
        print(f"{entry['path']}\t{entry['sha256'][:12]}\t{entry['bytes']}\t{entry['source_run']}")
    print(f"{len(entries)} artifacts")
    return 0


def _tile(project: Project, args: argparse.Namespace) -> int:
    import json

    from nakkhu import tiling

    try:
        if args.write_grid:
            grid = tiling.write_grid(project)
            print(f"grid written to {tiling.TILING_CONFIG}: {grid}")
            return 0
        if args.summary:
            summary = tiling.run_summary(project)
            print(json.dumps({k: summary[k] for k in ("grid_sha256", "mosaics_share_grid",
                                                      "cross_scene_ids")}, indent=1))
            print("promoted artifacts/tiling/tile_summary.json")
            return 0
        if not args.scene or not args.purpose:
            print("error: --scene and --purpose are required", file=sys.stderr)
            return 2
        stats = tiling.run_tile(project, args.scene, args.purpose,
                                rebuild_mosaic=args.rebuild_mosaic)
    except tiling.TilingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=1))
    return 0


def _georef(project: Project, args: argparse.Namespace) -> int:
    import json

    from nakkhu.georef import GeorefError, georef_run

    try:
        result = georef_run(project, args.run)
    except GeorefError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nakkhu", description="Nakkhu River building-level flood-exposure pipeline"
    )
    parser.add_argument("--root", help="nakkhu project root (default: NAKKHU_ROOT or checkout)")
    parser.add_argument(
        "--config", default=PROJECT_CONFIG, help=f"project config (default: {PROJECT_CONFIG})"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override a dotted config path, e.g. --set paths.data_root=E:/nakkhu-data",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    data = commands.add_parser("data", help="data manifest")
    data_commands = data.add_subparsers(dest="data_command", required=True, metavar="ACTION")
    verify = data_commands.add_parser(
        "verify", help="check manifest fields and recompute SHA-256 of every local file"
    )
    verify.add_argument("--manifest", help=f"manifest path (default: {DEFAULT_MANIFEST})")
    verify.set_defaults(func=_data_verify)
    fetch = data_commands.add_parser(
        "fetch", help="fetch sources from configs/data/sources.yaml and record them"
    )
    fetch.add_argument("ids", nargs="*", help="source ids")
    fetch.add_argument("--all", action="store_true", help="every source in the register")
    fetch.add_argument(
        "--dry-run", action="store_true", help="print URL, Content-Length and licence only"
    )
    fetch.add_argument("--force", action="store_true", help="refetch even if up to date")
    fetch.set_defaults(func=_data_fetch)
    freeze = data_commands.add_parser(
        "freeze",
        help="verify, then write and promote the manifest snapshot, licences and coverage",
    )
    freeze.set_defaults(func=_data_freeze)

    runs = commands.add_parser("runs", help="run directories under runs/")
    runs_commands = runs.add_subparsers(dest="runs_command", required=True, metavar="ACTION")
    runs_list = runs_commands.add_parser("list", help="list runs and their journal status")
    runs_list.add_argument("--stage", help="only this stage")
    runs_list.set_defaults(func=_runs_list)

    from nakkhu.aoi.cli import register as register_aoi  # plan 002 provenance

    register_aoi(commands)

    from nakkhu.hydro.cli import register as register_hand  # plan 008

    register_hand(commands)

    from nakkhu.labels_cli import register as register_labels  # plan 004

    register_labels(commands)

    from nakkhu.eval_cli import register as register_eval  # plan 005

    register_eval(commands)

    from nakkhu.audit_cli import register as register_audit  # plan 006

    register_audit(commands)

    from nakkhu.flood.cli import register as register_flood  # plan 009

    register_flood(commands)

    from nakkhu.detector_cli import register as register_detector  # plan 007

    register_detector(commands)

    from nakkhu.uncertainty.cli import register as register_uncertainty  # plan 010

    register_uncertainty(commands)

    from nakkhu.paper import register as register_paper  # plan 012

    register_paper(commands)

    from nakkhu.followup.cli import register as register_followup  # plan 014

    register_followup(commands)

    from nakkhu.mainriver.cli import register as register_mainriver  # main-river recompute

    register_mainriver(commands)

    artifacts =commands.add_parser("artifacts", help="approved artifacts under artifacts/")
    artifact_commands = artifacts.add_subparsers(
        dest="artifacts_command", required=True, metavar="ACTION"
    )
    promote_cmd = artifact_commands.add_parser(
        "promote", help="copy a file from a complete run into artifacts/ and index it"
    )
    promote_cmd.add_argument("source", help="<stage>/<run_id>/<file> under runs/")
    promote_cmd.add_argument("dest", help="artifact path relative to artifacts/")
    promote_cmd.add_argument("--replace", action="store_true", help="overwrite an artifact")
    promote_cmd.set_defaults(func=_artifacts_promote)
    artifacts_list = artifact_commands.add_parser("list", help="list artifacts/INDEX.yaml")
    artifacts_list.set_defaults(func=_artifacts_list)

    tile = commands.add_parser("tile", help="0.5 m mosaic and tiles of a scene (plan 003)")
    tile.add_argument("--scene", help="scene id from configs/tiling.yaml, e.g. maxar_wv02_20240113")
    tile.add_argument("--purpose", choices=("train", "infer"),
                      help="train: stride 640 over the training domain; infer: stride 512 over "
                      "the analysis extent")
    tile.add_argument("--rebuild-mosaic", action="store_true", help="rebuild a cached mosaic")
    tile.add_argument("--write-grid", action="store_true",
                      help="compute the grid and write it into configs/tiling.yaml")
    tile.add_argument("--summary", action="store_true",
                      help="compute and promote artifacts/tiling/tile_summary.json")
    tile.set_defaults(func=_tile)

    georef = commands.add_parser("georef", help="georeference and de-duplicate an infer run")
    georef.add_argument("--run", required=True, help="infer run id or <stage>/<run_id>")
    georef.set_defaults(func=_georef)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = load_project(args.root, args.config, args.overrides)
        return args.func(project, args)
    except (ConfigError, ManifestError, PathError, PromotionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
