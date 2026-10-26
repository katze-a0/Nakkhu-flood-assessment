"""``nakkhu labels ...`` subcommands (plan 004). OSM-facing commands call :mod:`nakkhu.labelqa`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project, ResolvedConfig

ARTIFACT_DIR = "labels"
SAM2_DIR = "_sam2"


def _arm_cfg(project: Project, arm: str) -> dict[str, Any]:
    from nakkhu.labels.arms import arm_config

    return arm_config(project.paths.root, arm).values


def _run(project: Project, name: str, values: dict[str, Any]):
    from nakkhu.runs import create_run

    return create_run(project, "labels", ResolvedConfig(values={"command": name, **values},
                                                         source="configs/labels"))


def _promote(project: Project, run: Any, files: list[str]) -> list[str]:
    from nakkhu.artifacts import promote

    done = []
    for name in files:
        promote(run.path / name, f"{ARTIFACT_DIR}/{name}", project.paths, replace=True)
        done.append(f"artifacts/{ARTIFACT_DIR}/{name}")
    return done


def _sam2_dir(project: Project) -> Path:
    from nakkhu.labels.build import labels_root

    return labels_root(project) / SAM2_DIR


def _sam2_results(project: Project) -> pd.DataFrame | None:
    path = _sam2_dir(project) / "results.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _tile_px(project: Project) -> int:
    from nakkhu.tiling import load_tiling

    return int(load_tiling(project).values["grid"]["tile_px"])


# --- build ----------------------------------------------------------------------------------


def _build(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.labels.arms import ARMS
    from nakkhu.labels.build import (
        LabelBuildError,
        arm_stats_rows,
        build_arm,
        prepare,
        split_ids,
    )
    from nakkhu.labels.gate import admit

    arms = list(ARMS) if args.arm == ["all"] else args.arm
    cfgs = {arm: _arm_cfg(project, arm) for arm in arms}
    common = cfgs[arms[0]]
    run = _run(project, "build", {"arms": {a: cfgs[a] for a in arms}, "workers": args.workers})
    with run.step("prepare"):
        t0 = time.perf_counter()
        prep = prepare(project, common, workers=args.workers, use_cache=not args.no_cache)
        prep.stats["prepare_s"] = round(time.perf_counter() - t0, 1)
        reg = prep.registration.merge(prep.index[["tile_id", "domain", "low_valid"]],
                                      on="tile_id")
        reg.to_csv(run.path / "registration_shifts.csv", index=False, float_format="%.4f")
        from nakkhu.labels.match import class_table

        class_table(prep.matches).to_csv(run.path / "match_classes.csv", index=False)
    sam2 = _sam2_results(project)
    gate_path = _sam2_dir(project).parent / "_gate" / "gate_scores.parquet"
    rows: list[dict[str, Any]] = []
    built: dict[str, Any] = {}
    for arm in arms:
        cfg = cfgs[arm]
        with run.step(f"arm-{arm.lower()}"):
            admitted = None
            status = "built"
            reason = ""
            st = cfg.get("selftrain")
            if st is not None:
                l7_dir = _sam2_dir(project).parent / "L7"
                if not st.get("predictions") or st.get("add_conf") is None:
                    status, reason = "pending", "needs the plan 007 L7 model (self-training)"
                elif not (l7_dir / "tiles.txt").is_file():
                    status, reason = "pending", "L7 not built"
                else:
                    from nakkhu.labels.selftrain import apply_to_arm

                    det = pd.read_parquet(project.paths.resolve(st["predictions"]))
                    st_stats = apply_to_arm(l7_dir, l7_dir.parent / arm, det,
                                            add_conf=float(st["add_conf"]),
                                            support_conf=float(st["support_conf"]),
                                            overlap_iou=float(st["overlap_iou"]),
                                            tile_px=_tile_px(project))
                    rows.append({"arm": arm, "status": "built", "split": "train",
                                 "tiles_admitted": len(st_stats),
                                 "kept_admitted": int(st_stats["labels"].sum()),
                                 "note": f"added {int(st_stats['added'].sum())}, dropped "
                                         f"{int(st_stats['dropped'].sum())}"})
                    continue
                rows.append({"arm": arm, "status": status, "split": "all", "note": reason})
                continue
            if cfg.get("refine") == "sam2" and sam2 is None:
                status, reason = "pending", "needs merged SAM2 outputs (offload/sam2_refine)"
            elif cfg.get("use_gate"):
                fraction = cfg["gate"].get("admit_fraction")
                if fraction is None:
                    status, reason = "pending", "gate.admit_fraction not chosen (val blocks)"
                elif not gate_path.is_file():
                    status, reason = "pending", "gate scores missing (nakkhu labels gate)"
                else:
                    scores = pd.read_parquet(gate_path)
                    train_ids = split_ids(project, cfg, cfg["split"]["train"])
                    admitted = admit(scores, train_ids, float(fraction))
            if status == "built":
                try:
                    result = build_arm(project, prep, cfg, workers=args.workers,
                                       sam2_results=sam2, admitted_by_gate=admitted)
                except LabelBuildError as error:
                    status, reason = "pending", str(error)
                    result = None
                if result is not None:
                    stats = result["tile_stats"]
                    if cfg.get("refine") == "sam2" and not stats["sam2_complete"].all():
                        # Report only tiles that SAM2 actually refined (e.g. the timed sample).
                        refined = set(sam2["tile_id"]) if sam2 is not None else set()
                        stats = stats.loc[stats["tile_id"].isin(refined)]
                        status = "partial"
                        reason = (f"stats over the {len(stats)} tiles with SAM2 outputs only "
                                  f"(of {len(result['tile_stats'])})")
                    built[arm] = stats
                    for r in arm_stats_rows(arm, status, stats):
                        r["note"] = reason
                        rows.append(r)
                    continue
            rows.append({"arm": arm, "status": status, "split": "all", "note": reason})
    pd.DataFrame(rows).to_csv(run.path / "arm_stats.csv", index=False, float_format="%.4f")
    summary = {"arms": {r["arm"]: r["status"] for r in rows}, "prepare": prep.stats}
    run.write_summary(json.loads(json.dumps(summary, default=str)))
    promoted = []
    if not args.no_promote:
        files = ["registration_shifts.csv", "match_classes.csv"]
        if set(arms) == set(ARMS):
            files.append("arm_stats.csv")
        promoted = _promote(project, run, files)
    print(pd.DataFrame(rows).loc[lambda f: f["split"] == "all"].to_string(index=False))
    print(json.dumps({"run": run.ref, "registration": prep.stats["registration"],
                      "promoted": promoted}, indent=1, default=str))
    return 0


# --- SAM2 -----------------------------------------------------------------------------------


def _sam2_prompts(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.labels.build import prepare
    from nakkhu.labels.sam2_refine import build_prompts

    cfg = _arm_cfg(project, "L6")
    prep = prepare(project, cfg, workers=args.workers)
    prompts = build_prompts(prep, cfg, _tile_px(project))
    out = _sam2_dir(project)
    out.mkdir(parents=True, exist_ok=True)
    prompts.to_parquet(out / "prompts.parquet")
    per_tile = prompts.groupby("tile_id").size().to_numpy()
    info = {"prompts": len(prompts), "tiles": int(per_tile.size),
            "prompts_per_tile_median": float(np.median(per_tile)),
            "prompts_per_tile_max": int(per_tile.max()),
            "path": project.paths.relative(out / "prompts.parquet"),
            "cache_key": prep.stats["cache_key"]}
    (out / "prompts.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return 0


def _sam2_run(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.labels.sam2_refine import run
    from nakkhu.labels.tiles import tile_dir

    cfg = _arm_cfg(project, "L6")["sam2"]
    out = _sam2_dir(project) / "results" / f"{args.name}_{args.part.replace('/', 'of')}.jsonl"
    tile_ids = None
    if args.tiles_file:
        tile_ids = [t.strip() for t in Path(args.tiles_file).read_text().split() if t.strip()]
    stats = run(_sam2_dir(project) / "prompts.parquet", tile_dir(project), out, args.part,
                cfg["model"], args.device, float(cfg["mask_threshold"]),
                int(cfg["batch_prompts"]), args.limit, tile_ids)
    stats["out"] = project.paths.relative(out)
    print(json.dumps(stats, indent=1))
    return 0


def _sam2_merge(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.labels.sam2_refine import agreement_table, merge

    cfg = _arm_cfg(project, "L6")["sam2"]
    files = sorted((_sam2_dir(project) / "results").glob("*.jsonl"))
    # Delta override: every `_sam2/delta/<name>/delta.json` lists tiles whose prompts changed
    # after the base bundle; for those tiles only `_sam2/delta/<name>/results/*.jsonl` counts
    # (sam2_refine.merge). Deltas apply in name order (names are UTC-stamped).
    deltas = []
    for meta in sorted((_sam2_dir(project) / "delta").glob("*/delta.json")):
        changed = set(json.loads(meta.read_text(encoding="utf-8"))["changed_tiles"])
        deltas.append((changed, sorted((meta.parent / "results").glob("*.jsonl"))))
    results, tiles = merge(files, deltas)
    prompts = pd.read_parquet(_sam2_dir(project) / "prompts.parquet")
    keep = set(zip(prompts["tile_id"], prompts["source"], prompts["index"].astype(int),
                   strict=True))
    current = [k in keep for k in zip(results["tile_id"], results["source"],
                                      results["index"].astype(int), strict=True)]
    results = results.loc[current].reset_index(drop=True)
    run = _run(project, "sam2-merge", {"sam2": cfg, "files": [f.name for f in files],
                                       "deltas": [str(len(c)) for c, _ in deltas]})
    with run.step("merge"):
        results.to_parquet(_sam2_dir(project) / "results.parquet")
        tiles.to_csv(run.path / "sam2_tiles.csv", index=False)
        table = agreement_table(results, float(cfg["min_mask_iou"]))
        table.to_csv(run.path / "sam2_agreement.csv", index=False, float_format="%.4f")
    done = int((tiles["status"] == "ok").sum()) if len(tiles) else 0
    info: dict[str, Any] = {"files": len(files), "deltas": len(deltas), "tiles_done": done,
                            "tiles_expected": int(prompts["tile_id"].nunique()),
                            "prompts_done": len(results), "prompts_expected": len(prompts)}
    run.write_summary(info)
    if not args.no_promote and info["tiles_done"] == info["tiles_expected"]:
        info["promoted"] = _promote(project, run, ["sam2_agreement.csv"])
    print(json.dumps({"run": run.ref, **info}, indent=1))
    return 0


def _sam2_delta(project: Project, args: argparse.Namespace) -> int:
    """Tiles whose current prompts differ from a base bundle's prompts, and their delta bundle."""
    import subprocess

    from nakkhu.labels.sam2_refine import changed_tiles

    base = _sam2_dir(project) / "bundle" / args.base / "data" / "prompts.parquet"
    current = _sam2_dir(project) / "prompts.parquet"
    changed = changed_tiles(pd.read_parquet(base), pd.read_parquet(current))
    cmd = [sys.executable, str(project.paths.root / "offload" / "sam2_refine" / "make_bundle.py"),
           "--kind", "delta", "--base", args.base, "--shards", str(args.shards),
           "--data-root", str(project.paths.data_root)]
    listing = _sam2_dir(project) / "delta_tiles.tmp"
    listing.write_text("".join(f"{t}\n" for t in changed), encoding="utf-8")
    cmd += ["--tiles-file", str(listing)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    listing.unlink(missing_ok=True)
    print(out.stdout + out.stderr)
    return out.returncode


# --- gate -----------------------------------------------------------------------------------


def _gate(project: Project, args: argparse.Namespace) -> int:
    from nakkhu.labels.build import labels_root, prepare
    from nakkhu.labels.gate import gate_scores, match_features
    from nakkhu.labels.sam2_refine import agreement_table

    cfg = _arm_cfg(project, "L7")
    prep = prepare(project, cfg, workers=args.workers)
    # Excluded footprints (greenhouses) are not label instances: leave them out of the
    # cross-source features too.
    keep_g = [i[~prep.excluded["google"][i]] for i in prep.tile_google]
    keep_m = [i[~prep.excluded["microsoft"][i]] for i in prep.tile_microsoft]
    feats = match_features(prep.index["tile_id"].tolist(), keep_g, keep_m,
                           prep.matches.google, prep.matches.microsoft)
    feats = feats.merge(prep.registration[["tile_id", "peak_ratio", "reg_ok"]], on="tile_id")
    feats = feats.merge(prep.index[["tile_id", "domain", "low_valid", "valid_frac"]],
                        on="tile_id")
    sam2 = _sam2_results(project)
    spec = dict(cfg["gate"]["features"])
    status = "complete"
    feats["sam2_mean_iou"] = np.nan
    if sam2 is not None:
        agree = agreement_table(sam2, float(cfg["sam2"]["min_mask_iou"]))
        prompts = pd.read_parquet(_sam2_dir(project) / "prompts.parquet")
        if set(prompts["tile_id"]) <= set(agree["tile_id"]):
            feats = feats.drop(columns="sam2_mean_iou").merge(
                agree[["tile_id", "sam2_mean_iou"]], on="tile_id", how="left")
        else:   # a partial SAM2 feature would distort every rank: leave it out
            status = (f"provisional_no_sam2 (SAM2 outputs for {agree['tile_id'].nunique()} "
                      f"of {prompts['tile_id'].nunique()} tiles)")
    if feats["sam2_mean_iou"].isna().all():
        spec.pop("sam2_mean_iou", None)
        if status == "complete":
            status = "provisional_no_sam2"
    scores = gate_scores(feats, spec)
    scores.insert(1, "gate_status", status)
    from nakkhu.labels.build import _split_lookup

    split_of, _ = _split_lookup(project, cfg)
    scores.insert(2, "split", scores["tile_id"].map(split_of))
    run = _run(project, "gate", {"gate": cfg["gate"], "features_used": list(spec),
                                 "cache_key": prep.stats["cache_key"]})
    with run.step("gate"):
        scores.to_csv(run.path / "gate_scores.csv", index=False, float_format="%.6f")
        out = labels_root(project) / "_gate"
        out.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(out / "gate_scores.parquet")
        again = gate_scores(feats, spec)
        deterministic = bool(np.array_equal(again["gate_score"].to_numpy(),
                                            scores["gate_score"].to_numpy()))
    info: dict[str, Any] = {"status": status, "features": list(spec), "tiles": len(scores),
                            "deterministic_rerun": deterministic}
    run.write_summary(info)
    if not args.no_promote:
        info["promoted"] = _promote(project, run, ["gate_scores.csv"])
    print(json.dumps({"run": run.ref, **info}, indent=1))
    return 0


# --- OSM reference checks (labelqa) ---------------------------------------------------------


def _split_tiles(project: Project, cfg: dict[str, Any], splits: list[str]) -> set[str]:
    """Usable tiles of plan 005 splits; this module never scores test tiles outside G1."""
    from nakkhu.labels.build import split_ids

    if "test" in splits:
        raise SystemExit("error: test tiles are scored only by `labels g1 --splits test`")
    return {t for split in splits for t in split_ids(project, cfg, split)}


def _convention(project: Project, args: argparse.Namespace) -> int:
    import geopandas as gpd
    import shapely

    from nakkhu import labelqa
    from nakkhu.labels.build import prepare
    from nakkhu.labels.tiles import to_map

    cfg = _arm_cfg(project, "L5")
    prep = prepare(project, cfg, workers=args.workers)
    tile_ids = _split_tiles(project, cfg, args.splits)
    bounds = np.array([b for t, b in zip(prep.index["tile_id"], prep.index["bounds"],
                                         strict=True) if t in tile_ids])
    clip = shapely.box(bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(),
                       bounds[:, 3].max())
    osm = labelqa.aligned_osm(project, prep.index, clip)  # registered per tile
    sam2_units = None
    sam2 = _sam2_results(project)
    if sam2 is not None and tile_ids <= set(sam2["tile_id"]):   # only with full coverage
        idx = prep.index.set_index("tile_id")
        part = sam2.loc[sam2["tile_id"].isin(tile_ids) & (sam2["mask_wkt"] != "")]
        geoms = [to_map(np.array([shapely.from_wkt(w)], dtype=object), idx.loc[t, "affine"])[0]
                 for t, w in zip(part["tile_id"], part["mask_wkt"], strict=True)]
        sam2_units = gpd.GeoDataFrame(geometry=geoms, crs=prep.footprints.google.crs)
    table, summary = labelqa.convention_table(prep, osm, tile_ids,
                                              float(cfg["match"]["contain_min"]), sam2_units)
    run = _run(project, "convention", {"splits": args.splits, "match": cfg["match"],
                                       "cache_key": prep.stats["cache_key"]})
    with run.step("convention"):
        summary.insert(0, "region", "+".join(args.splits))
        summary.to_csv(run.path / "convention_1n.csv", index=False, float_format="%.4f")
        table.to_csv(run.path / "convention_components.csv", index=False)
    f1s = summary.loc[summary["iou"] == 0.5].set_index("convention")["f1"]
    choice = ("split" if f1s.get("google_split", 0) >= f1s.get("microsoft_merged", 0)
              else "merged")
    info: dict[str, Any] = {
        "region_tiles": len(tile_ids), "components": len(table), "osm_in_region": len(osm),
        "f1_iou50": f1s.to_dict(), "convention_by_f1_iou50": choice,
        "caveat": "OSM aligned per tile, greenhouses removed; region = plan 005 split tiles; "
                  "OSM completeness unknown"}
    run.write_summary(json.loads(json.dumps(info, default=str)))
    if not args.no_promote:
        info["promoted"] = _promote(project, run, ["convention_1n.csv"])
    print(summary.to_string(index=False))
    print(json.dumps({"run": run.ref, **info}, indent=1, default=str))
    return 0


def _osm_register(project: Project, args: argparse.Namespace) -> int:
    from nakkhu import labelqa
    from nakkhu.labels.tiles import load_index, tile_dir

    cfg = _arm_cfg(project, "L5")
    directory = tile_dir(project, cfg["tiles"]["scene"], cfg["tiles"]["purpose"])
    index = load_index(directory)
    index = index.loc[index["image"].notna()].reset_index(drop=True)
    run = _run(project, "osm-register", {"registration": cfg["registration"],
                                         "tiles": cfg["tiles"]})
    with run.step("osm-register"):
        shifts, summary = labelqa.register_osm(project, index, directory, cfg, args.workers)
        path = labelqa.osm_shift_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        shifts.to_parquet(path)
        shifts.to_csv(run.path / "osm_registration_shifts.csv", index=False,
                      float_format="%.4f")
        (run.path / "osm_registration_summary.json").write_text(
            json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    run.write_summary(json.loads(json.dumps(summary, default=str)))
    info: dict[str, Any] = {"run": run.ref, "shifts": project.paths.relative(path)}
    if not args.no_promote:
        info["promoted"] = _promote(project, run, ["osm_registration_shifts.csv",
                                                   "osm_registration_summary.json"])
    print(json.dumps({**info, **summary}, indent=1, default=str))
    return 0


def _setpoint(project: Project, args: argparse.Namespace) -> int:
    """L7 admit fraction on val-split tiles only (pooled L6 label F1 vs aligned OSM)."""
    import shapely

    from nakkhu import labelqa
    from nakkhu.labels.build import labels_root
    from nakkhu.labels.gate import choose_admit_fraction
    from nakkhu.labels.tiles import load_index, tile_dir

    cfg = _arm_cfg(project, "L7")
    val = _split_tiles(project, cfg, [cfg["split"]["val"]])
    test = {t for t in _all_split_ids(project, cfg, "test")}
    index = load_index(tile_dir(project))
    part = index.loc[index["tile_id"].isin(val)]
    b = np.array(part["bounds"].tolist())
    # Unshifted OSM: tile_label_quality applies each tile's OSM shift itself (passing
    # aligned_osm here shifted OSM twice; fixed 2026-10-06).
    osm = labelqa.load_osm(project, shapely.box(b[:, 0].min() - 20, b[:, 1].min() - 20,
                                                b[:, 2].max() + 20, b[:, 3].max() + 20),
                           drop_greenhouses=True)
    quality = labelqa.tile_label_quality(project, labels_root(project) / "L6", part, osm,
                                         labelqa.load_osm_shifts(project),
                                         set(part["domain"]), _tile_px(project))
    scores = pd.read_parquet(labels_root(project) / "_gate" / "gate_scores.parquet")
    if not str(scores["gate_status"].iloc[0]) == "complete":
        print("error: the gate is provisional; rebuild it with SAM2 first", file=sys.stderr)
        return 1
    key = f"{int(round(float(cfg['gate']['admit_iou']) * 100)):02d}"
    merged = quality.merge(scores[["tile_id", "gate_score"]], on="tile_id")
    merged = merged.sort_values(["gate_score", "tile_id"], ascending=[False, True])
    rows = []
    for frac in cfg["gate"]["admit_candidates"]:
        n = int(np.ceil(float(frac) * len(merged)))
        took, rest = merged.iloc[:n], merged.iloc[n:]
        tp = int(took[f"tp_{key}"].sum())
        fp = int(took[f"fp_{key}"].sum())
        fn = int(took[f"fn_{key}"].sum()) + int(rest["n_osm"].sum())
        rows.append({"fraction": float(frac), "val_tiles_admitted": n, "tp": tp, "fp": fp,
                     "fn": fn, "f1": labelqa.f1(tp, fp, fn),
                     "tile_ids": merged["tile_id"].tolist()})
    curve = pd.DataFrame(rows)
    chosen = choose_admit_fraction(curve, val, test)
    run = _run(project, "setpoint", {"gate": cfg["gate"], "split": cfg["split"]})
    with run.step("setpoint"):
        curve.drop(columns="tile_ids").to_csv(run.path / "gate_setpoint_val.csv", index=False,
                                              float_format="%.4f")
    info = {"admit_fraction": chosen, "val_tiles": len(merged), "iou": key,
            "written": False}
    if args.write:
        _write_admit_fraction(project, chosen)
        info["written"] = True
    run.write_summary(info)
    if not args.no_promote:
        info["promoted"] = _promote(project, run, ["gate_setpoint_val.csv"])
    print(curve.drop(columns="tile_ids").to_string(index=False))
    print(json.dumps({"run": run.ref, **info}, indent=1))
    return 0


def _all_split_ids(project: Project, cfg: dict[str, Any], split: str) -> list[str]:
    from nakkhu.splits import load_split

    table, _ = load_split(project, cfg["split"]["file"])
    return sorted(table.loc[table["split"] == split, "tile_id"])


def _write_admit_fraction(project: Project, fraction: float) -> None:
    import re

    path = project.paths.resolve("configs/labels/common.yaml")
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(r"(?m)^(  admit_fraction: ).*$", rf"\g<1>{fraction:g}", text)
    if n != 1:
        raise SystemExit("error: gate.admit_fraction line not found in common.yaml")
    path.write_text(new, encoding="utf-8")


def _g1(project: Project, args: argparse.Namespace) -> int:
    import shapely

    from nakkhu import labelqa
    from nakkhu.labels.build import labels_root
    from nakkhu.labels.tiles import load_index, tile_dir

    index = load_index(tile_dir(project))
    index = index.loc[index["image"].notna() & ~index["low_valid"]]
    cfg = _arm_cfg(project, "L5")
    tag = "dev" if "test" not in args.splits else "test"
    wanted = {t for sp in args.splits for t in _all_split_ids(project, cfg, sp)}
    index = index.loc[index["tile_id"].isin(wanted)]
    domains = set(index["domain"])
    # The run exists before scoring: test access is journalled under its ref (plan 005 guard).
    run = _run(project, f"g1-{tag}", {"arm": args.arm, "splits": sorted(args.splits)})
    part = index.loc[index["domain"].isin(domains)]
    b = np.array(part["bounds"].tolist())
    osm = labelqa.load_osm(project, shapely.box(b[:, 0].min() - 20, b[:, 1].min() - 20,
                                                b[:, 2].max() + 20, b[:, 3].max() + 20),
                           drop_greenhouses=True)
    shifts = labelqa.load_osm_shifts(project)
    arm_dir = labels_root(project) / args.arm
    try:
        errors = labelqa.tile_label_quality(project, arm_dir, index, osm, shifts, domains,
                                            _tile_px(project),
                                            access=(run.ref, run.run_id.rsplit("-", 1)[-1]))
    except labelqa.LabelQAError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    scores = pd.read_parquet(labels_root(project) / "_gate" / "gate_scores.parquet")
    fractions = np.round(np.arange(0, 0.95, 0.1), 2)
    rows, curves = [], []
    for key in ("30", "50"):
        stat, curve = labelqa.g1(scores, errors, fractions, key)
        for flag in (True, False):   # OSM registered on the tile vs flagged (shift 0)
            part = errors.loc[errors["osm_reg_ok"] == flag]
            stat[f"spearman_rho_osm_reg_ok_{flag}"] = labelqa.g1(scores, part, fractions,
                                                                 key)[0]["spearman_rho"]
            stat[f"tiles_osm_reg_ok_{flag}"] = labelqa.g1(scores, part, fractions,
                                                          key)[0]["tiles"]
        rows.append(stat)
        curve.insert(0, "iou", key)
        curves.append(curve)
    with run.step("g1"):
        errors.to_csv(run.path / f"g1_{tag}_tile_quality.csv", index=False, float_format="%.4f")
        pd.concat(curves).to_csv(run.path / f"g1_{tag}_curve.csv", index=False,
                                 float_format="%.4f")
    info = {"mode": tag, "arm": args.arm, "splits": sorted(args.splits),
            "osm_greenhouses_dropped": True,
            "gate_status": str(scores["gate_status"].iloc[0]), "results": rows,
            "statistic": "Spearman rho(gate score, label F1 vs aligned OSM)",
            "note": "dev diagnostic on non-test tiles, not the pre-registered G1 test"
            if tag == "dev" else "G1"}
    run.write_summary(json.loads(json.dumps(info, default=str)))
    print(pd.concat(curves).to_string(index=False))
    print(json.dumps({"run": run.ref, **info}, indent=1, default=str))
    return 0


G1_CONF_NOTE = "plan 004 G1 confirmatory"      # journal note (a pending analysis, guard.py)
G1_CONF_FILES = ("g1_confirmatory_tiles.csv", "g1_confirmatory.json")


def _g1_confirmatory(project: Project, args: argparse.Namespace) -> int:
    """G1 confirmatory rho (prereg ``falsifiers.G1.confirmatory``; reported, not gating): one
    journalled test access that reads the recall_points / recall_nearmiss answers.

    Reuses the primary G1 run (``--primary``): its arm, its test tiles (the rows of
    ``g1_test_tile_quality.csv``) and the gate scores it read; the primary rho is not
    re-scored. Refuses unless prereg verify passes, the test hold is off, sittings 1 and 1b are
    complete with unchanged sitting-1b inputs, and no confirmatory run is journalled yet.
    """
    import shapely

    from nakkhu import labelqa
    from nakkhu.audit import core
    from nakkhu.audit.workflow import input_drift, recall_v2, require_complete
    from nakkhu.evaluate.guard import read_test_journal, sha256_file, verify
    from nakkhu.labels.build import labels_root
    from nakkhu.labels.obb import read_yolo_obb
    from nakkhu.labels.tiles import load_index, tile_dir, to_map
    from nakkhu.uncertainty.run import hold_state

    primary_dir = project.paths.runs / args.primary
    summary = json.loads((primary_dir / "summary.json").read_text(encoding="utf-8"))
    tile_csv = primary_dir / "g1_test_tile_quality.csv"
    gate_path = labels_root(project) / "_gate" / "gate_scores.parquet"
    arm = str(summary["arm"])
    arm_dir = labels_root(project) / arm
    prereg = verify(project)
    held, why = hold_state(project)
    cfg = core.load_audit_config(project)
    problems = [f"prereg: {p}" for p in prereg["problems"]]
    if held:
        problems.append(f"hold: {why}")
    if summary.get("mode") != "test" or summary.get("note") != "G1":
        problems.append(f"{args.primary} is not the primary G1 test run")
    done = [e for e in read_test_journal(project) if e.get("note") == G1_CONF_NOTE]
    if done:
        problems.append(f"already run once ({done[0].get('run')}, {done[0].get('utc')})")
    for task in (core.PILOT_TASK, core.NEARMISS_TASK):
        try:
            require_complete(project, cfg, task)
        except core.AuditError as error:
            problems.append(str(error))
    drift = input_drift(project, core.NEARMISS_TASK)
    if drift:
        problems.append(f"recall_nearmiss inputs changed since its sample: {', '.join(drift)}")
    if not gate_path.is_file() or not tile_csv.is_file():
        problems.append("gate scores or the primary tile table missing")
    settings = {"arm": arm, "primary_run": args.primary, "min_answered": 5,
                "tolerance_m": 1.0, "protocol": "v2", "outside_stratum": "excluded",
                "weights": "N_h / n_h over answered (Yes/No) units, Unsure excluded"}
    print(json.dumps({"ready": not problems, "problems": problems, **settings}, indent=1))
    if problems or args.dry_run:
        return 1 if problems else 0
    inputs = {"primary_summary_sha256": sha256_file(primary_dir / "summary.json"),
              "primary_tile_quality_sha256": sha256_file(tile_csv),
              "gate_scores_sha256": sha256_file(gate_path),
              "recall_points_items_sha256": sha256_file(core.items_path(project, core.PILOT_TASK)),
              "recall_points_responses_sha256": sha256_file(
                  core.responses_path(project, core.PILOT_TASK)),
              "recall_nearmiss_items_sha256": sha256_file(
                  core.items_path(project, core.NEARMISS_TASK)),
              "recall_nearmiss_responses_sha256": sha256_file(
                  core.responses_path(project, core.NEARMISS_TASK)),
              "labels_tiles_sha256": sha256_file(arm_dir / "tiles.txt")}
    # The run exists before any answer is read: the access is journalled under its ref.
    run = _run(project, "g1-confirmatory-test", {**settings, "inputs": inputs})
    try:
        access = labelqa._guard_split(project, {"test"}, (run.ref, run.run_id.rsplit("-", 1)[-1]),
                                      note=G1_CONF_NOTE)
    except labelqa.LabelQAError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    with run.step("g1-confirmatory"):
        items, v2, clicked, meta = recall_v2(project, cfg)
        wanted = set(pd.read_csv(tile_csv)["tile_id"])
        index = load_index(tile_dir(project))
        tiles = index.loc[index["tile_id"].isin(wanted)].sort_values("tile_id")
        px = _tile_px(project)
        obbs = {}
        for _, row in tiles.iterrows():
            _, corners = read_yolo_obb(arm_dir / "labels" / f"{row['tile_id']}.txt", px)
            obbs[row["tile_id"]] = to_map(np.array([shapely.Polygon(c) for c in corners],
                                                   dtype=object), row["affine"])
        quality, counts = labelqa.g1_audit_quality(items, v2, clicked, tiles, obbs,
                                                   settings["min_answered"],
                                                   settings["tolerance_m"])
        scores = pd.read_parquet(gate_path)
        stat = labelqa.g1_confirmatory(scores, quality)
        quality = quality.merge(scores[["tile_id", "gate_score"]], on="tile_id", how="left")
        quality.to_csv(run.path / G1_CONF_FILES[0], index=False, float_format="%.6f")
        result = {"statistic": "Spearman rho(gate score, audit quality of the L6 labels)",
                  "definition": "prereg falsifiers.G1.confirmatory", "gating": False,
                  **stat, **counts, "tiles_in_primary_run": len(tiles),
                  "tiles_with_answered_units": len(quality), **settings, **meta,
                  "inputs": inputs, "test_access": access}
        (run.path / G1_CONF_FILES[1]).write_text(
            json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    run.write_summary(json.loads(json.dumps(result, default=str)))
    if not args.no_promote:
        result["promoted"] = _promote(project, run, list(G1_CONF_FILES))
    print(json.dumps({"run": run.ref, **{k: result[k] for k in
                                         ("spearman_rho", "tiles", "eligible_tiles",
                                          "eligible_tiles_without_yes")}},
                     indent=1, default=str))
    return 0


# --- contact sheet --------------------------------------------------------------------------


def _contact_sheet(project: Project, args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw

    from nakkhu.labels.build import labels_root
    from nakkhu.labels.obb import read_yolo_obb
    from nakkhu.labels.tiles import load_index, read_tile, tile_dir

    arm_dir = labels_root(project) / args.arm
    if args.all_train:
        stats = pd.read_parquet(arm_dir / "tile_stats.parquet")
        tiles = stats.loc[(stats["split"] == "train") & ~stats["low_valid"], "tile_id"].tolist()
    else:
        tiles = [t for t in (arm_dir / "tiles.txt").read_text().split() if t]
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(tiles, size=min(args.n, len(tiles)), replace=False).tolist())
    index = load_index(tile_dir(project)).set_index("tile_id", drop=False)
    px, thumb, cols = _tile_px(project), args.thumb, 6
    rows_n = int(np.ceil(len(pick) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows_n * thumb), "white")
    for k, tile_id in enumerate(pick):
        rgb, _ = read_tile(tile_dir(project), index.loc[tile_id])
        image = Image.fromarray(rgb)
        ignore = arm_dir / "ignore" / f"{tile_id}.png"
        if ignore.is_file():
            mask = Image.open(ignore).convert("L").point(lambda v: 110 if v else 0)
            image.paste(Image.new("RGB", image.size, (255, 0, 255)), mask=mask)
        draw = ImageDraw.Draw(image)
        _, boxes = read_yolo_obb(arm_dir / "labels" / f"{tile_id}.txt", px)
        for corners in boxes:
            draw.polygon([(float(x), float(y)) for x, y in corners], outline=(255, 255, 0),
                         width=2)
        draw.text((6, 6), tile_id, fill=(255, 255, 255))
        sheet.paste(image.resize((thumb, thumb)), ((k % cols) * thumb, (k // cols) * thumb))
    out = arm_dir / f"contact_sheet_{args.n}_seed{args.seed}.jpg"
    sheet.save(out, quality=90)
    print(f"wrote {project.paths.relative(out)} ({len(pick)} tiles; local only: Maxar imagery "
          "is CC BY-NC and is not a paper figure)")
    return 0


def register(commands: Any) -> None:
    labels = commands.add_parser("labels", help="consensus weak labels and tile gate (plan 004)")
    sub = labels.add_subparsers(dest="labels_command", required=True, metavar="ACTION")

    def common(p: argparse.ArgumentParser, promote: bool = True) -> None:
        p.add_argument("--workers", type=int, default=8, help="processes (default 8)")
        if promote:
            p.add_argument("--no-promote", action="store_true",
                           help="do not promote outputs to artifacts/labels/")

    build = sub.add_parser("build", help="build label arms (L1-L8; pending arms are reported)")
    build.add_argument("--arm", nargs="+", required=True, help="arm ids or 'all'")
    build.add_argument("--no-cache", action="store_true", help="recompute match/registration")
    common(build)
    build.set_defaults(func=_build)

    prompts = sub.add_parser("sam2-prompts", help="write the SAM2 prompt table (L5 keeps)")
    common(prompts, promote=False)
    prompts.set_defaults(func=_sam2_prompts)
    srun = sub.add_parser("sam2-run", help="run SAM2 locally on a shard (resumable)")
    srun.add_argument("--part", default="0/1")
    srun.add_argument("--limit", type=int)
    srun.add_argument("--tiles-file", help="only these tile ids (one per line)")
    srun.add_argument("--device", default="cuda")
    srun.add_argument("--name", default="local")
    srun.set_defaults(func=_sam2_run)
    sdelta = sub.add_parser("sam2-delta",
                            help="delta bundle of tiles whose prompts changed since a bundle")
    sdelta.add_argument("--base", required=True, help="base bundle name under _sam2/bundle/")
    sdelta.add_argument("--shards", type=int, default=2)
    sdelta.set_defaults(func=_sam2_delta)
    smerge = sub.add_parser("sam2-merge",
                            help="merge SAM2 result shards (delta results override base tiles)")
    smerge.add_argument("--no-promote", action="store_true")
    smerge.set_defaults(func=_sam2_merge)

    gate = sub.add_parser("gate", help="tile quality gate scores for every tile")
    common(gate)
    gate.set_defaults(func=_gate)

    conv = sub.add_parser("convention", help="row-house convention vs OSM (Step 8)")
    conv.add_argument("--splits", nargs="+", default=["train"],
                      help="plan 005 splits used as the region (never test)")
    common(conv)
    conv.set_defaults(func=_convention)

    osm_reg = sub.add_parser("osm-register",
                             help="register OSM per tile to the imagery (before any scoring)")
    common(osm_reg)
    osm_reg.set_defaults(func=_osm_register)

    sp = sub.add_parser("setpoint", help="L7 admit fraction on val-split tiles")
    sp.add_argument("--write", action="store_true", help="write gate.admit_fraction")
    sp.add_argument("--no-promote", action="store_true")
    sp.set_defaults(func=_setpoint)

    g1 = sub.add_parser("g1", help="gate score vs label error against OSM")
    g1.add_argument("--arm", default="L5")
    g1.add_argument("--splits", nargs="+", default=["train"],
                    help="plan 005 splits; test needs the frozen prereg (guard + journal)")
    g1.set_defaults(func=_g1)
    g1c = sub.add_parser("g1-confirmatory",
                         help="G1 confirmatory rho vs the recall audit (one journalled test "
                              "access; reported, not gating)")
    g1c.add_argument("--primary", required=True,
                     help="the primary G1 test run, e.g. labels/<run_id>")
    g1c.add_argument("--dry-run", action="store_true",
                     help="check the prerequisites only (no journal entry, no answer read)")
    g1c.add_argument("--no-promote", action="store_true")
    g1c.set_defaults(func=_g1_confirmatory)

    sheet = sub.add_parser("contact-sheet", help="local visual check of an arm's labels")
    sheet.add_argument("--arm", default="L5")
    sheet.add_argument("--n", type=int, default=30)
    sheet.add_argument("--seed", type=int, default=0)
    sheet.add_argument("--thumb", type=int, default=320)
    sheet.add_argument("--all-train", action="store_true",
                       help="sample from all valid train tiles, not only admitted ones")
    sheet.set_defaults(func=_contact_sheet)
