"""Detector inference (plan 007): overlapping windows, pixel detections, infer runs.

``nakkhu infer --run train/<arm>/<model>/seed<k> --region val|train|test [--scene S]`` writes
``runs/infer/<id>/`` with ``detections_px.parquet`` (``tile_id, conf, cls, x1..y4`` in
tile-image pixels) and ``inputs.json`` (``tile_index``, ``model``, ``train_run``, ``region``,
``weights_sha256``); ``nakkhu georef`` then applies centre ownership + cross-tile NMS (plan 003,
frozen in the prereg) and writes ``detections.gpkg`` for ``nakkhu eval``.

Regions:

- ``val``: 640 px windows at stride 512 (the plan 003 infer lattice) over the val-split tiles,
  cut from the WV02 mosaic with the train tiles' global stretch (``nakkhu infer
  --cut-val-windows``; ``{data_root}/interim/tiles/<scene>/valinfer/``). Same procedure as the
  test region, so val and test metrics are comparable. No window touches a test tile.
- ``train``: the plan 003 train-grid tiles (stride 640) of an arm's ``tiles.txt``; used only for
  the L8 self-training round (detections in train-tile pixels).
- ``test``: the plan 003 infer tiles of the analysis extent (inference only; scoring is
  ``nakkhu eval --split test`` behind the prereg guard).

Low-confidence detections are kept (``conf_min``) so AP and the val F1-max threshold see the
whole curve; ``max_det`` is raised because val tiles hold up to ~800 label instances (the
pilot showed 1000 binding on ~190 of 265 val windows at a conf floor of 0.02-0.11; 3000 at
``conf_min`` 0.02 keeps the tail above the val F1 thresholds).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nakkhu.config import Project, ResolvedConfig

SCENE = "maxar_wv02_20240113"
CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
CONF_MIN = 0.02
MAX_DET = 3000
BATCH = 16


class InferError(RuntimeError):
    """Inference inputs are inconsistent."""


def tiles_dir(project: Project, scene: str, purpose: str) -> Path:
    return project.paths.data_root / "interim" / "tiles" / scene / purpose


# --- val windows ----------------------------------------------------------------------------


def cut_val_windows(project: Project, scene: str = SCENE) -> dict[str, Any]:
    """Cut stride-512 windows over the val-split tiles into ``tiles/<scene>/valinfer/``."""
    import shutil

    import rasterio
    import shapely
    from rasterio.transform import Affine
    from shapely.geometry import box

    from nakkhu import radiometry
    from nakkhu.splits import load_split
    from nakkhu.tiling import (
        GDAL_ENV,
        INDEX_FILE,
        _read_window,
        frozen_grid,
        load_tiling,
        read_index,
        select_tiles,
        write_index,
    )

    cfg = load_tiling(project).values
    grid = frozen_grid(project, cfg)
    stride = int(cfg["strides"]["infer"])
    train_index, train_meta = read_index(tiles_dir(project, scene, "train") / INDEX_FILE)
    frame, _ = load_split(project)
    bounds = train_index.set_index("tile_id")["bounds"]
    val_ids = frame.loc[frame["split"] == "val", "tile_id"]
    test_ids = frame.loc[frame["split"] == "test", "tile_id"]
    val_region = shapely.union_all([box(*bounds[t]) for t in val_ids])
    test_region = shapely.union_all([box(*bounds[t]) for t in test_ids])
    specs = select_tiles(grid, stride, val_region)
    rad = train_meta["radiometry"]
    quality = int(train_meta["jpeg_quality"])
    target = tiles_dir(project, scene, "valinfer")
    if target.exists():
        shutil.rmtree(target)
    (target / "images").mkdir(parents=True)
    (target / "masks").mkdir()
    rows = []
    mosaic = project.paths.resolve(train_meta["mosaic"])
    with rasterio.Env(**GDAL_ENV), rasterio.open(mosaic) as src:
        for spec in specs:
            wb = grid.window_bounds(spec.x0, spec.y0, grid.tile_px, grid.tile_px)
            if shapely.area(shapely.intersection(box(*wb), test_region)) > 0:
                raise InferError(f"val window {spec.tile_id} overlaps a test tile")
            data, valid = _read_window(src, grid, spec.x0, spec.y0)
            frac = float(valid.mean())
            image = mask = None
            if frac > 0:
                rgb = radiometry.stretch(data, valid, rad["mode"], rad["percentiles"])
                image = f"images/{spec.tile_id}.jpg"
                radiometry.write_jpg(target / image, rgb, quality)
                if frac < 1:
                    mask = f"masks/{spec.tile_id}.png"
                    radiometry.write_mask_png(target / mask, valid)
            affine = grid.transform @ Affine.translation(spec.x0, spec.y0)
            rows.append({"tile_id": spec.tile_id, "col": spec.col, "row": spec.row,
                         "x0_px": spec.x0, "y0_px": spec.y0,
                         "affine": [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f],
                         "crs": grid.crs, "bounds": list(wb), "valid_frac": frac,
                         "image": image, "mask": mask, "grid_sha256": train_meta["grid_sha256"]})
    out = pd.DataFrame(rows)
    meta = {**train_meta, "purpose": "valinfer", "stride_px": stride,
            "domain": "val-split tiles of configs/splits/split_v1.yaml",
            "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run": None}
    write_index(target / INDEX_FILE, out, meta)
    return {"windows": len(out), "with_image": int(out["image"].notna().sum()),
            "index": project.paths.relative(target / INDEX_FILE)}


# --- prediction -----------------------------------------------------------------------------


def predict(weights: Path, images: list[Path], ids: list[str], task: str, *,
            device: str | None = None, conf: float = CONF_MIN, max_det: int = MAX_DET,
            imgsz: int = 640, batch: int = BATCH) -> pd.DataFrame:
    """Ultralytics prediction -> ``tile_id, conf, cls, x1..y4`` (tile-image pixels)."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task=task)
    rows: list[np.ndarray] = []
    tile_col: list[str] = []
    for start in range(0, len(images), batch):
        chunk = [str(p) for p in images[start:start + batch]]
        results: Any = model.predict(chunk, imgsz=imgsz, conf=conf, max_det=max_det,
                                           device=device, verbose=False)
        for tid, res in zip(ids[start:start + batch], results, strict=True):
            if task == "obb":
                if res.obb is None or len(res.obb) == 0:
                    continue
                corners = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 8)
                scores = res.obb.conf.cpu().numpy()
                cls = res.obb.cls.cpu().numpy()
            else:
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                b = res.boxes.xyxy.cpu().numpy()
                corners = np.stack([b[:, 0], b[:, 1], b[:, 2], b[:, 1],
                                    b[:, 2], b[:, 3], b[:, 0], b[:, 3]], axis=1)
                scores = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
            rows.append(np.column_stack([scores, cls, corners]))
            tile_col += [tid] * len(scores)
    data = np.concatenate(rows) if rows else np.zeros((0, 10))
    frame = pd.DataFrame(data, columns=["conf", "cls", *CORNERS])
    frame.insert(0, "tile_id", tile_col)
    frame["cls"] = frame["cls"].astype("int64")
    return frame


def region_images(project: Project, region: str, scene: str, arm: str | None
                  ) -> tuple[Path, list[str], list[Path]]:
    """(tile index path, tile ids, image paths) of an inference region."""
    from nakkhu.tiling import INDEX_FILE, read_index

    purpose = {"val": "valinfer", "train": "train", "test": "infer"}[region]
    root = tiles_dir(project, scene, purpose)
    index, _ = read_index(root / INDEX_FILE)
    index = index.loc[index["image"].notna()].sort_values("tile_id")
    if region == "train":
        from nakkhu.dataset import labels_root, split_tiles

        if not arm:
            raise InferError("--region train needs the arm (its tiles.txt)")
        wanted = set((labels_root(project) / arm / "tiles.txt").read_text().split())
        split = split_tiles(project)
        if any(split[t]["split"] != "train" for t in wanted):
            raise InferError("an arm tile is not in the train split")
        index = index.loc[index["tile_id"].isin(wanted)]
    ids = list(index["tile_id"])
    images = [root / p for p in index["image"]]
    missing = [p for p in images if not p.is_file()]
    if missing:
        raise InferError(f"{len(missing)} images missing under {root}, e.g. {missing[:2]}")
    return root / INDEX_FILE, ids, images


def infer_run(project: Project, train_ref: str, region: str, *, scene: str = SCENE,
              device: str | None = None, conf: float = CONF_MIN) -> dict[str, Any]:
    """Predict with a trained unit's ``best.pt`` over a region into a new infer run."""
    from nakkhu.runs import atomic_write_json, create_run
    from nakkhu.train import sha256

    ref = train_ref.replace("\\", "/").strip("/").removeprefix("runs/")
    tdir = project.paths.runs / ref
    record_path = tdir / "train.json"
    if not record_path.is_file():
        raise InferError(f"{tdir} has no train.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    weights = tdir / "weights" / "best.pt"
    wsha = sha256(weights)
    if wsha != record["best_pt_sha256"]:
        raise InferError(f"{weights}: SHA-256 differs from train.json")
    index_path, ids, images = region_images(project, region, scene, record["arm"])
    model_key = f"{ref}@{wsha[:12]}"
    values = {"command": "infer", "train_run": ref, "region": region, "scene": scene,
              "weights_sha256": wsha, "conf_min": conf, "max_det": MAX_DET}
    run = create_run(project, "infer", ResolvedConfig(values=values, source="nakkhu.infer"))
    with run.step("predict"):
        t0 = time.time()
        frame = predict(weights, images, ids, str(record["task"]), device=device, conf=conf)
        wall = time.time() - t0
        frame.to_parquet(run.path / "detections_px.parquet", index=False)
        inputs = {"tile_index": project.paths.relative(index_path), "model": model_key,
                  "train_run": ref, "region": region, "scene": scene,
                  "weights_sha256": wsha, "conf_min": conf, "max_det": MAX_DET,
                  "tiles": len(ids), "detections": len(frame), "predict_wall_s": round(wall, 1),
                  "ms_per_tile": round(1000 * wall / max(len(ids), 1), 1)}
        atomic_write_json(run.path / "inputs.json", inputs)
    run.write_summary(inputs)
    return {"run": run.ref, **inputs}
