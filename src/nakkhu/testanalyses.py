"""Plan 007 test analyses behind the one-shot guard (prereg v7 ``falsifiers.C1``, ``C5``,
``cross_sensor`` / ``C4``) and the WV03 prep step they need.

Prep (no detection read, no metric; orchestrator decision log 2026-10-13, same precedent as the
WV02 test-tile OSM registration of plan 004):

- :func:`cut_wv03_tiles`: WV03 2023-04-04 tiles on the plan 003 train grid (640 px, stride 640)
  with the plan 003 cutter, stretch and validity rule, into
  ``{data_root}/interim/tiles/maxar_wv03_20230404/train/``. The mosaic covers only the analysis
  extent + 500 m; these tiles are never written into a training set.
- :func:`register_wv03`: OSM registered per tile to the WV03 imagery with exactly the R-OSM
  method and settings (``labelqa.register_osm``: ``configs/labels/common.yaml#registration``,
  24 px cap, peak-ratio threshold from its own null). WV03 has no train-domain tile, so the
  null pool is the WV03 tiles themselves (every tile with an image and valid fraction >= 0.5).
  Output ``{data_root}/interim/labels/osm_registration_shifts_wv03.parquet``; its SHA-256 and the
  WV03 index SHA-256 are recorded in ``runs/detector/registry.json`` (``wv03_registration``) and
  checked before scoring.

Analyses (pure functions on per-block tables, so they are testable on synthetic data):

- C1: seed-mean (seeds 0-2) test F1@0.5 of the consensus arm minus the single-source arm, both
  chosen by mean val F1; paired block bootstrap (the same resamples for every run); pass if the
  CI lower bound > 0 and the difference > the larger test seed SD (ddof 1) of the two arms.
  IoU 0.3 reported, not gating.
- C5: per final seed, F1(seed) - F1(footprint) with a paired block-bootstrap CI; a footprint
  matches a seed if the CI includes 0 or lies below 0 (ci_low <= 0); it matches the model if it
  matches >= 3 of the 5 seeds.
- C4 / cross_sensor: per final seed at its own WV02 val threshold, F1@0.5 on WV03 against OSM
  registered to WV03 and on WV02 against R-OSM, both on exactly the test tiles whose WV03 valid
  fraction >= 0.5; drop = (WV02 - WV03) x 100 percentage points with a paired block-bootstrap CI;
  seed mean and SD; pass if the seed-mean drop <= 20. Sensitivity: WV03 scored against R-OSM
  (WV02-aligned) unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project, ResolvedConfig

WV03 = "maxar_wv03_20230404"
WV02 = "maxar_wv02_20240113"
WV03_SHIFTS = "osm_registration_shifts_wv03.parquet"
MIN_VALID = 0.5
DROP_PASS_PP = 20.0
C5_MIN_SEEDS = 3


class AnalysisError(RuntimeError):
    """A plan 007 test analysis cannot run as pre-registered."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- paired block bootstrap on per-block tables ----------------------------------------------


def _cols(per_block: pd.DataFrame, k: str) -> dict[str, np.ndarray]:
    return {c: per_block[c].to_numpy(dtype=float)
            for c in (f"tp_pred_{k}", "n_pred", f"tp_ref_{k}", "n_ref")}


def f1_point(per_block: pd.DataFrame, k: str = "50") -> float:
    c = {n: float(v.sum()) for n, v in _cols(per_block, k).items()}
    p = c[f"tp_pred_{k}"] / c["n_pred"] if c["n_pred"] else 0.0
    r = c[f"tp_ref_{k}"] / c["n_ref"] if c["n_ref"] else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def f1_reps(per_block: pd.DataFrame, weights: np.ndarray, k: str = "50") -> np.ndarray:
    from nakkhu.evaluate import bootstrap as B

    return B.additive(_cols(per_block, k), weights,
                      B.f1_stat(f"tp_pred_{k}", "n_pred", f"tp_ref_{k}", "n_ref"))


def _same_blocks(tables: list[pd.DataFrame]) -> list[str]:
    ref = list(tables[0]["block_id"].astype(str))
    for t in tables[1:]:
        if list(t["block_id"].astype(str)) != ref:
            raise AnalysisError("per-block tables differ in their blocks: not pairable")
    return ref


def weights_for(n_blocks: int, n_boot: int, seed: int) -> np.ndarray:
    from nakkhu.evaluate import bootstrap as B

    return B.multiplicities(n_blocks, n_boot, seed)


def _ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    from nakkhu.evaluate import bootstrap as B

    return B.percentile_ci(samples, level)


def c1(arm_a: list[pd.DataFrame], arm_b: list[pd.DataFrame], *, n_boot: int = 2000,
       seed: int = 0, level: float = 0.95) -> dict[str, Any]:
    """C1: ``arm_a`` (consensus) minus ``arm_b`` (single source), seed-mean test F1."""
    _same_blocks([*arm_a, *arm_b])
    W = weights_for(len(arm_a[0]), n_boot, seed)
    out: dict[str, Any] = {}
    for k in ("50", "30"):
        fa = np.array([f1_point(t, k) for t in arm_a])
        fb = np.array([f1_point(t, k) for t in arm_b])
        ra = np.mean([f1_reps(t, W, k) for t in arm_a], axis=0)
        rb = np.mean([f1_reps(t, W, k) for t in arm_b], axis=0)
        diff = float(fa.mean() - fb.mean())
        lo, hi = _ci(ra - rb, level)
        sd_a = float(fa.std(ddof=1)) if len(fa) > 1 else float("nan")
        sd_b = float(fb.std(ddof=1)) if len(fb) > 1 else float("nan")
        larger = float(np.nanmax([sd_a, sd_b]))
        out[f"iou_{k}"] = {"f1_a_seeds": fa.tolist(), "f1_b_seeds": fb.tolist(),
                           "f1_a_mean": float(fa.mean()), "f1_b_mean": float(fb.mean()),
                           "sd_a": sd_a, "sd_b": sd_b, "diff": diff, "ci_low": lo,
                           "ci_high": hi, "larger_seed_sd": larger,
                           "pass": bool(lo > 0 and diff > larger)}
    out["pass"] = out["iou_50"]["pass"]
    out["gating"] = "iou_50 (iou_30 reported)"
    return out


def c5(seeds: list[pd.DataFrame], footprint: pd.DataFrame, *, n_boot: int = 2000,
       seed: int = 0, level: float = 0.95, min_seeds: int = C5_MIN_SEEDS) -> dict[str, Any]:
    """C5 for one footprint inventory against the final seeds (IoU 0.5)."""
    _same_blocks([*seeds, footprint])
    W = weights_for(len(footprint), n_boot, seed)
    rf = f1_reps(footprint, W)
    ff = f1_point(footprint)
    rows = []
    for t in seeds:
        d = f1_point(t) - ff
        lo, hi = _ci(f1_reps(t, W) - rf, level)
        rows.append({"f1_seed": f1_point(t), "f1_footprint": ff, "diff": float(d),
                     "ci_low": lo, "ci_high": hi, "matches": bool(lo <= 0)})
    n_match = int(sum(r["matches"] for r in rows))
    return {"per_seed": rows, "seeds_matching": n_match, "min_seeds": min_seeds,
            "matches": bool(n_match >= min_seeds)}


def cross_sensor_stats(pairs: list[tuple[pd.DataFrame, pd.DataFrame]], *, n_boot: int = 2000,
                       seed: int = 0, level: float = 0.95,
                       pass_pp: float = DROP_PASS_PP) -> dict[str, Any]:
    """C4 from ``(wv02_per_block, wv03_per_block)`` per seed (same tiles and blocks)."""
    _same_blocks([t for p in pairs for t in p])
    W = weights_for(len(pairs[0][0]), n_boot, seed)
    rows = []
    for a, b in pairs:
        d = 100.0 * (f1_point(a) - f1_point(b))
        lo, hi = _ci(100.0 * (f1_reps(a, W) - f1_reps(b, W)), level)
        rows.append({"f1_wv02": f1_point(a), "f1_wv03": f1_point(b), "drop_pp": float(d),
                     "ci_low_pp": lo, "ci_high_pp": hi})
    drops = np.array([r["drop_pp"] for r in rows])
    mean = float(drops.mean())
    return {"per_seed": rows, "drop_pp_mean": mean,
            "drop_pp_sd": float(drops.std(ddof=1)) if len(drops) > 1 else float("nan"),
            "pass_pp": pass_pp, "pass": bool(mean <= pass_pp)}


# --- WV03 prep (no detections, no metric) ----------------------------------------------------


def wv03_tile_dir(project: Project) -> Path:
    from nakkhu.tiling import tiles_dir

    return tiles_dir(project, WV03, "train")


def wv03_shift_path(project: Project) -> Path:
    from nakkhu.labels.build import labels_root

    return labels_root(project) / WV03_SHIFTS


def cut_wv03_tiles(project: Project) -> dict[str, Any]:
    """Plan 003 cutter on the WV03 mosaic at the train stride (registration only)."""
    from nakkhu.runs import create_run
    from nakkhu.tiling import (
        _resolved,
        build_mosaic,
        cut_tiles,
        frozen_grid,
        load_tiling,
    )

    cfg = copy.deepcopy(load_tiling(project).values)
    purposes = cfg["scenes"][WV03]["purposes"]
    if "train" not in purposes:            # registration tiles only, never a training set
        purposes.append("train")
    run = create_run(project, "tiling", _resolved(cfg, scene=WV03, purpose="train",
                                                  note="C4 registration tiles only"))
    with run.step("grid"):
        grid = frozen_grid(project, cfg)
    with run.step("mosaic"):
        mosaic = build_mosaic(project, WV03, cfg, grid, rebuild=False)
    with run.step("tiles"):
        stats = cut_tiles(project, WV03, "train", cfg, grid, mosaic, run.ref)
    index = wv03_tile_dir(project) / "index.parquet"
    out = {"run": run.ref, "index": str(index), "index_sha256": _sha(index),
           "mosaic_reused": mosaic.get("reused"), "tiles": stats.get("tiles", stats)}
    run.write_summary(json.loads(json.dumps(out, default=str)))
    return out


def register_wv03(project: Project, workers: int = 1) -> dict[str, Any]:
    """R-OSM registration of OSM onto the WV03 tiles (own null from the WV03 tiles)."""
    from nakkhu import labelqa
    from nakkhu.labels.tiles import load_index
    from nakkhu.labels_cli import _arm_cfg
    from nakkhu.runs import create_run

    directory = wv03_tile_dir(project)
    if not (directory / "index.parquet").is_file():
        raise AnalysisError("WV03 registration tiles are missing (cut_wv03_tiles)")
    index = load_index(directory)
    index = index.loc[index["image"].notna()].reset_index(drop=True)
    cfg = copy.deepcopy(_arm_cfg(project, "L5"))
    null_domains = sorted(set(index.loc[~index["low_valid"].astype(bool), "domain"]))
    cfg["tiles"]["train_domains"] = null_domains        # own null: the WV03 tiles themselves
    run = create_run(project, "labels", ResolvedConfig(values={
        "command": "osm-register-wv03", "scene": WV03, "registration": cfg["registration"],
        "null_domains": null_domains, "index_sha256": _sha(directory / "index.parquet")},
        source="nakkhu.testanalyses"))
    with run.step("osm-register-wv03"):
        shifts, summary = labelqa.register_osm(project, index, directory, cfg, workers)
        path = wv03_shift_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        shifts.to_parquet(path)
        shifts.to_csv(run.path / "osm_registration_shifts_wv03.csv", index=False,
                      float_format="%.4f")
        (run.path / "osm_registration_summary_wv03.json").write_text(
            json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    record = {"run": run.ref, "shifts": project.paths.relative(path),
              "shifts_sha256": _sha(path),
              "index_sha256": _sha(directory / "index.parquet"),
              "null_domains": null_domains, "tiles_with_image": len(index),
              "registered_share": summary["by_domain"]["all"]["registered_share"],
              "min_peak_ratio": summary["min_peak_ratio"]}
    run.write_summary({**record, "summary": json.loads(json.dumps(summary, default=str))})
    return record


# --- C4 inputs and scoring (inside the guarded one-shot command only) ------------------------


def wv03_problems(project: Project, reg: dict[str, Any]) -> list[str]:
    """Why the cross-sensor analysis could not run now (checked by --dry-run too)."""
    rec = reg.get("wv03_registration")
    if not rec:
        return ["WV03 OSM registration missing (prep step not run)"]
    problems = []
    path = project.paths.resolve(rec["shifts"]) if not Path(rec["shifts"]).is_absolute() \
        else Path(rec["shifts"])
    if not path.is_file() or _sha(path) != rec["shifts_sha256"]:
        problems.append("WV03 registration shifts changed or missing since the prep run")
    index = wv03_tile_dir(project) / "index.parquet"
    if not index.is_file() or _sha(index) != rec["index_sha256"]:
        problems.append("WV03 registration tile index changed or missing")
    return problems


def wv03_units(project: Project, reg: dict[str, Any], seeds: list[int],
               arm: str = "L5", model: str = "yolo11n-obb") -> list[dict[str, Any]]:
    out = []
    for s in seeds:
        train_run = f"train/{arm}/{model}/seed{s}"
        units = list(reg.get("units", {}).values())
        wv03 = [u.get(f"infer_test_{WV03}") for u in units
                if u["train_run"] == train_run and u.get(f"infer_test_{WV03}")]
        ev = [u.get("eval_val") for u in units if u["train_run"] == train_run
              and u.get("eval_val")]
        out.append({"seed": s, "train_run": train_run, "wv03_infer": wv03[0] if wv03 else None,
                    "val_eval": ev[0] if ev else None})
    return out


def _aligned(project: Project, index: pd.DataFrame, shifts: pd.DataFrame, clip: Any
             ) -> np.ndarray:
    import shapely

    from nakkhu.labelqa import load_osm, shift_map

    s = shifts.set_index("tile_id")
    osm = load_osm(project, clip, drop_greenhouses=True)
    cent = shapely.centroid(osm.geometry.values)
    boxes = np.array([shapely.box(*b) for b in index["bounds"]], dtype=object)
    ci, ti = shapely.STRtree(boxes).query(cent, predicate="within")
    tile_of = np.full(len(osm), -1)
    tile_of[ci[::-1]] = ti[::-1]
    geoms = osm.geometry.values.copy()
    for t in np.unique(tile_of[tile_of >= 0]):
        row = index.iloc[int(t)]
        if row["tile_id"] not in s.index:
            continue
        sel = tile_of == t
        geoms[sel] = shift_map(geoms[sel], row["affine"], float(s.loc[row["tile_id"], "dx"]),
                               float(s.loc[row["tile_id"], "dy"]))
    return np.asarray(shapely.oriented_envelope(geoms), dtype=object)


def run_cross_sensor(project: Project, units: list[dict[str, Any]],
                     wv02_tests: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    """Score C4 (the caller has journalled the access). Returns the stats and per-seed rows."""
    import shapely

    from nakkhu.evaluate.guard import SPLIT_FILE, sha256_file
    from nakkhu.evaluate.score import (
        _threshold_from,
        load_detections,
        reference_geoms,
        region_and_tiles,
        score,
    )
    from nakkhu.georef import georef_run
    from nakkhu.labels.tiles import load_index

    region, tiles, index = region_and_tiles(project, "test")
    wv03_index = load_index(wv03_tile_dir(project))
    valid = wv03_index.set_index("tile_id")["valid_frac"]
    keep = tiles["tile_id"].map(valid).fillna(0.0).to_numpy() >= MIN_VALID
    sub = tiles.loc[keep].reset_index(drop=True)
    if sub.empty:
        raise AnalysisError("no test tile has WV03 valid fraction >= 0.5")
    boxes = np.array([shapely.box(*b) for b in sub["bounds"]], dtype=object)
    sub_region = shapely.intersection(region, shapely.union_all(boxes))
    margin = float(params["margin_m"])
    refs02, _ = reference_geoms(project, index, sub_region, margin)
    shifts03 = pd.read_parquet(wv03_shift_path(project))
    refs03 = _aligned(project, index, shifts03,
                      shapely.buffer(sub_region, margin + 50.0))
    split_sha = sha256_file(project.paths.resolve(SPLIT_FILE))
    pairs, pairs_sens, rows = [], [], []
    for u in units:
        thr = _threshold_from(project, u["val_eval"],
                              json.loads((project.paths.runs / u["wv03_infer"] / "inputs.json")
                                         .read_text(encoding="utf-8"))["model"], split_sha)
        t = float(thr["threshold"])
        dets = {}
        for key, ref in (("wv02", wv02_tests[u["train_run"]]), ("wv03", u["wv03_infer"])):
            if not (project.paths.runs / ref / "detections.gpkg").is_file():
                georef_run(project, ref)
            d = load_detections(project.paths.runs / ref)
            dets[key] = (np.asarray(d.geometry.values, dtype=object), d["conf"].to_numpy())
        s02 = score(*dets["wv02"], refs02, sub_region, sub, threshold=t, params=params,
                    ap_ci=False)
        s03 = score(*dets["wv03"], refs03, sub_region, sub, threshold=t, params=params,
                    ap_ci=False)
        s03s = score(*dets["wv03"], refs02, sub_region, sub, threshold=t, params=params,
                     ap_ci=False)
        pairs.append((s02.per_block, s03.per_block))
        pairs_sens.append((s02.per_block, s03s.per_block))
        rows.append({"seed": u["seed"], "threshold": t, "wv03_infer": u["wv03_infer"],
                     "wv02_infer": wv02_tests[u["train_run"]]})
    kw = {"n_boot": int(params["n_boot"]), "seed": int(params["seed"]),
          "level": float(params["level"])}
    out = cross_sensor_stats(pairs, **kw)
    sens = cross_sensor_stats(pairs_sens, **kw)
    for r, p in zip(rows, out["per_seed"], strict=True):
        p.update(r)
    return {"primary": out, "sensitivity_wv02_aligned_osm": sens, "tiles": len(sub),
            "tiles_test": len(tiles), "min_valid": MIN_VALID}
