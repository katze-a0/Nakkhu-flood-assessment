"""``nakkhu dataset``, ``nakkhu train``, ``nakkhu infer`` and ``nakkhu detector`` (plan 007)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from nakkhu.config import Project


def _dataset_build(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.dataset import DatasetError, build

    try:
        out = [build(project, arm, args.task, force=args.force) for arm in args.arm]
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps([{k: o[k] for k in ("counts", "reused")} | {"arm": o["inputs"]["arm"]}
                      for o in out], indent=1))
    return 0


def _train(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.dataset import DatasetError
    from nakkhu.train import TrainError, load_train_config, train

    config = load_train_config(project, args.config, tuple(args.cfg_set))
    try:
        record = train(project, config, args.arm, args.seed, device=args.device,
                       force=args.force)
    except (TrainError, DatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({k: record[k] for k in ("run", "train_wall_s", "ultralytics_val")},
                     indent=1))
    return 0


def _infer(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.infer import InferError, cut_val_windows, infer_run

    try:
        if args.cut_val_windows:
            result: dict[str, Any] = cut_val_windows(project)
        else:
            if not args.run or not args.region:
                print("error: --run and --region are required", file=sys.stderr)
                return 2
            result = infer_run(project, args.run, args.region, scene=args.scene,
                               device=args.device)
    except InferError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str))
    return 0


def _final(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.ensemble import EnsembleError, cli_final

    try:
        return cli_final(project, args)
    except EnsembleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _zeroshot(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.zeroshot import cli_zeroshot

    return cli_zeroshot(project, args)


def _efficiency(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.efficiency import cli_efficiency

    return cli_efficiency(project, args)


def _detector(project: Project, args: argparse.Namespace) -> int:
    from nakkhu import detector

    func = {"ingest": detector.cli_ingest, "selftrain": detector.cli_selftrain,
            "summarise": detector.cli_summarise,
            "inventory": detector.cli_inventory,
            "score-test": detector.cli_score_test,
            "final": _final, "zeroshot": _zeroshot,
            "efficiency": _efficiency}[args.detector_command]
    return func(project, args)


def register(commands: Any) -> None:
    ds = commands.add_parser("dataset", help="Ultralytics training sets from label arms "
                             "(plan 007; train + val tiles only)")
    dsub = ds.add_subparsers(dest="dataset_command", required=True, metavar="ACTION")
    b = dsub.add_parser("build", help="materialise arms into "
                        "{data_root}/interim/detector/datasets/<arm>-<task>/")
    b.add_argument("--arm", nargs="+", required=True)
    b.add_argument("--task", choices=("obb", "detect"), default="obb")
    b.add_argument("--force", action="store_true")
    b.set_defaults(func=_dataset_build)

    tr = commands.add_parser("train", help="train one (arm, model, seed) unit into "
                             "runs/train/<arm>/<model>/seed<k>/ (offloaded GPU)")
    tr.add_argument("--config", required=True, help="configs/train/<model>.yaml")
    tr.add_argument("--arm", required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--device", help="Ultralytics device (e.g. 0, cpu)")
    tr.add_argument("--force", action="store_true", help="retrain a complete unit")
    tr.add_argument("--cfg", dest="cfg_set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a train-config key (pilot: --cfg epochs=N)")
    tr.set_defaults(func=_train)

    inf = commands.add_parser("infer", help="predict with a trained unit over a region into "
                              "runs/infer/<id>/detections_px.parquet (offloaded GPU)")
    inf.add_argument("--run", help="train/<arm>/<model>/seed<k>")
    inf.add_argument("--region", choices=("val", "train", "test"))
    inf.add_argument("--scene", default="maxar_wv02_20240113")
    inf.add_argument("--device")
    inf.add_argument("--cut-val-windows", action="store_true",
                     help="cut the stride-512 val windows from the WV02 mosaic (CPU)")
    inf.set_defaults(func=_infer)

    det = commands.add_parser("detector", help="plan 007 bookkeeping: ingest offloaded "
                              "units, L8 self-training round, artifact tables")
    dsub2 = det.add_subparsers(dest="detector_command", required=True, metavar="ACTION")
    ing = dsub2.add_parser("ingest", help="copy pulled unit outputs into runs/ (SHA-256 "
                           "checked), georef the val predictions and score them on val")
    ing.add_argument("source", nargs="+", help="pulled kernel/session output directories")
    ing.add_argument("--no-eval", action="store_true")
    ing.set_defaults(func=_detector)
    st = dsub2.add_parser("selftrain", help="build L8 from the L7 seed-0 model (plan 004 "
                          "Step 7): add_conf at 90 %% precision vs OSM on val")
    st.add_argument("--train-preds", required=True, help="infer run of L7 seed 0 on train")
    st.add_argument("--val-eval", required=True, help="val eval run of L7 seed 0")
    st.set_defaults(func=_detector)
    iv = dsub2.add_parser("inventory", help="D3: Google v3 / Microsoft / Overture as fixed "
                          "OBB inventories (infer runs), scored on val with --fixed")
    iv.add_argument("--no-eval", action="store_true")
    iv.set_defaults(func=_detector)
    stt = dsub2.add_parser("score-test", help="score every evaluated unit and inventory on "
                           "test ONCE (prereg guard, journalled); --dry-run lists them")
    stt.add_argument("--dry-run", action="store_true")
    stt.set_defaults(func=_detector)
    fi = dsub2.add_parser("final", help="detector_final: 5-seed clustered inventory "
                          "(nakkhu.ensemble); val eval only, test-region geometry, no test "
                          "metric")
    fi.add_argument("--arm", default="L5")
    fi.add_argument("--model", default="yolo11n-obb")
    fi.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    fi.add_argument("--promote", action="store_true")
    fi.set_defaults(func=_detector)
    zs = dsub2.add_parser("zeroshot", help="D3: import SAM3 zero-shot outputs (offload/sam3), "
                          "georef, score val only")
    zs.add_argument("source", nargs="+", help="pulled kernel output directories")
    zs.set_defaults(func=_detector)
    ef = dsub2.add_parser("efficiency", help="model_efficiency.csv: params / GFLOPs at 640 "
                          "(seed-k best.pt on CPU) and the logged ms per 640 px window on val "
                          "and the test region (mean +- SD over seeds; nothing re-run)")
    ef.add_argument("--arm", default="L5")
    ef.add_argument("--models", nargs="+",
                    default=["yolo11n-obb", "yolov8n-obb", "yolo11s-obb", "yolo11s-hbb"])
    ef.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="seeds averaged for the timings (default: the Phase B seeds 0-2)")
    ef.add_argument("--size-seed", type=int, default=0, help="seed whose best.pt is sized")
    ef.add_argument("--no-promote", action="store_true")
    ef.set_defaults(func=_detector)
    sm = dsub2.add_parser("summarise", help="label_arms.csv / baselines.csv / cost.csv from "
                          "val evals (no test)")
    sm.add_argument("--promote", action="store_true")
    sm.set_defaults(func=_detector)


def remote_main(argv: list[str] | None = None) -> int:
    """Entry for offloaded units: only ``dataset``, ``train`` and ``infer`` (no other plan's
    CLI is imported, so a work-in-progress module elsewhere cannot break a remote run)."""
    from nakkhu.config import load_project

    parser = argparse.ArgumentParser(prog="nakkhu-remote")
    parser.add_argument("--root")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    commands = parser.add_subparsers(dest="command", required=True)
    register(commands)
    args = parser.parse_args(argv)
    project = load_project(args.root, overrides=args.overrides)
    return args.func(project, args)
