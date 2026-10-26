"""Step 4: SAM2 box-prompt refinement (L6-L8). Heavy part offloaded (``offload/sam2_refine``).

Prompts: every instance that L5 keeps and whose registered footprint passes the area filter, in
tile pixel coordinates; the prompt box is the footprint's axis-aligned box dilated by
``box_dilate_m`` and clipped to the tile. SAM2 (Hugging Face port of SAM 2.1, Apache 2.0)
returns one mask per box (``multimask_output=False``); the mask's largest connected component
is the refined polygon. Outputs per prompt: mask/footprint IoU, mask polygon (WKT, tile px) and
its minimum rotated rectangle. ``build_arm`` replaces a kept label by the mask polygon when
IoU >= ``min_mask_iou`` and keeps the registered footprint otherwise (Decisions 2026-10-06).

The runner is resumable (one JSON line per finished tile, keyed by tile id; finished keys are
skipped) and shards with ``--part i/P`` (every P-th tile). This module imports nothing from
``nakkhu`` at top level, so the offloaded job only needs it, the prompts and the tile images.

    python -m nakkhu.labels.sam2_refine run --prompts prompts.parquet --tiles <dir> \
        --out res_0.jsonl --part 0/2 [--device cuda] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely

PROMPT_COLUMNS = ("tile_id", "image", "source", "index", "footprint_wkt",
                  "x0", "y0", "x1", "y1")


# --- prompts (local; uses the label pipeline) -----------------------------------------------


def build_prompts(prep: Any, arm_cfg: dict[str, Any], tile_px: int) -> pd.DataFrame:
    """Prompt table from the L5 rule on prepared data (see module docstring)."""
    from nakkhu.labels.build import arm_inputs
    from nakkhu.labels.obb import KEEP

    pixel = float(prep.index.iloc[0]["affine"][0])
    min_area_px = float(arm_cfg["obb"]["min_area_m2"]) / pixel ** 2
    dilate = float(arm_cfg["sam2"]["box_dilate_m"]) / pixel
    inputs = arm_inputs(prep, {**arm_cfg, "rule": "consensus"})
    rows: list[dict[str, Any]] = []
    for i, row in prep.index.iterrows():
        for j in np.flatnonzero(inputs.actions[i] == KEEP):
            _append_prompt(rows, row, str(inputs.sources[i][j]), int(inputs.indices[i][j]),
                           inputs.polygons[i][j], min_area_px, dilate, tile_px)
    return pd.DataFrame(rows, columns=list(PROMPT_COLUMNS))


def prompt_box(poly: Any, min_area_px: float, dilate: float,
               tile_px: int) -> np.ndarray | None:
    """Prompt box of a kept footprint, or None when it gets no prompt (below the area filter,
    or its dilated box clipped to the tile is under 1 px wide, i.e. it lies outside the tile)."""
    if shapely.area(poly) < min_area_px:
        return None
    x0, y0, x1, y1 = shapely.bounds(poly)
    box = np.clip([x0 - dilate, y0 - dilate, x1 + dilate, y1 + dilate], 0, tile_px)
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return None
    return box


def _append_prompt(rows: list[dict[str, Any]], row: Any, source: str, index: int, poly: Any,
                   min_area_px: float, dilate: float, tile_px: int) -> None:
    box = prompt_box(poly, min_area_px, dilate, tile_px)
    if box is None:
        return
    rows.append({"tile_id": row["tile_id"], "image": row["image"], "source": source,
                 "index": index, "footprint_wkt": shapely.to_wkt(poly, rounding_precision=3),
                 "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})


# --- masks -> polygons ----------------------------------------------------------------------


def mask_polygon(mask: np.ndarray) -> Any:
    """Largest 4-connected component of a binary mask as a polygon (pixel coordinates)."""
    from rasterio.features import shapes

    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return shapely.Polygon()
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = mask[r0:r1, c0:c1]          # exact: every mask pixel lies in its bounding box
    parts = [shapely.geometry.shape(g) for g, v in
             shapes(crop.astype("uint8"), mask=crop.astype(bool), connectivity=4) if v == 1]
    if not parts:
        return shapely.Polygon()
    best = max(parts, key=lambda p: p.area)
    return shapely.transform(best, lambda xy: xy + np.array([c0, r0], dtype=float))


def iou(a: Any, b: Any) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = shapely.area(shapely.intersection(a, b))
    union = shapely.area(a) + shapely.area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def refine_records(masks: np.ndarray, prompts: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-prompt result records from binary masks (n, H, W) aligned with ``prompts``."""
    records = []
    columns = zip(prompts["source"], prompts["index"], prompts["footprint_wkt"], strict=True)
    for mask, (source, index, wkt) in zip(masks, columns, strict=True):
        footprint = shapely.from_wkt(wkt)
        poly = mask_polygon(mask)
        rect = shapely.minimum_rotated_rectangle(poly) if not poly.is_empty else poly
        mrr = (np.asarray(rect.exterior.coords)[:4].round(3).reshape(-1).tolist()
               if rect.geom_type == "Polygon" else None)
        records.append({"source": str(source), "index": int(index),
                        "iou": round(iou(poly, footprint), 6),
                        "mask_area_px": int(mask.sum()),
                        "mask_wkt": shapely.to_wkt(poly, rounding_precision=2)
                        if not poly.is_empty else "",
                        "mrr": mrr})
    return records


# --- SAM2 runner ----------------------------------------------------------------------------


class Sam2Runner:
    def __init__(self, model_id: str, device: str = "cuda", mask_threshold: float = 0.0,
                 batch_prompts: int = 64) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        model: Any = Sam2Model.from_pretrained(model_id)
        self.model = model.to(device).eval()
        self.mask_threshold = mask_threshold
        self.batch = batch_prompts
        self.model_id = model_id

    def masks(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = image.shape[:2]
        out = np.zeros((len(boxes), h, w), dtype=bool)
        with torch.inference_mode():
            base = self.processor(images=image, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_embeddings(base["pixel_values"])
            for start in range(0, len(boxes), self.batch):
                chunk = boxes[start:start + self.batch].tolist()
                inputs = self.processor(input_boxes=[chunk], original_sizes=[[h, w]],
                                        return_tensors="pt").to(self.device)
                result = self.model(input_boxes=inputs["input_boxes"],
                                    image_embeddings=embeddings, multimask_output=False)
                masks = self.processor.post_process_masks(
                    result.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                    mask_threshold=self.mask_threshold, binarize=True)[0]
                out[start:start + len(chunk)] = masks[:, 0].numpy().astype(bool)
        return out


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (ValueError, KeyError):
                continue   # truncated last line of a killed process
    return done


def run(prompts_path: Path, tiles_root: Path, out: Path, part: str = "0/1",
        model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda",
        mask_threshold: float = 0.0, batch_prompts: int = 64, limit: int | None = None,
        tile_ids: Iterable[str] | None = None) -> dict[str, Any]:
    from PIL import Image

    i, p = (int(x) for x in part.split("/"))
    prompts = pd.read_parquet(prompts_path)
    order = sorted(prompts["tile_id"].unique())
    if tile_ids is not None:
        wanted = set(tile_ids)
        order = [t for t in order if t in wanted]
    order = [t for k, t in enumerate(order) if k % p == i]
    if limit is not None:
        order = order[:limit]
    done = read_done(out)
    runner = Sam2Runner(model_id, device, mask_threshold, batch_prompts)
    stamp = {"host": platform.node(), "platform": platform.platform(),
             "python": sys.version.split()[0], "model": model_id, "device": device,
             "torch": runner.torch.__version__,
             "cuda_device": (runner.torch.cuda.get_device_name(0)
                             if device.startswith("cuda") else None),
             "commit": os.environ.get("CODE_COMMIT", "unknown")}
    groups = dict(tuple(prompts.groupby("tile_id")))
    out.parent.mkdir(parents=True, exist_ok=True)
    n_tiles = n_prompts = 0
    wall = 0.0
    with out.open("a", encoding="utf-8") as handle:
        for tile_id in order:
            if tile_id in done:
                continue
            group = groups[tile_id].reset_index(drop=True)
            t0 = time.perf_counter()
            try:
                image = np.asarray(Image.open(tiles_root / group["image"].iloc[0])
                                   .convert("RGB"))
                boxes = group[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
                records = refine_records(runner.masks(image, boxes), group)
                rec: dict[str, Any] = {"key": tile_id, "status": "ok", "results": records}
            except Exception as error:   # recorded, not fatal for the shard
                rec = {"key": tile_id, "status": "error", "error": repr(error)}
            elapsed = time.perf_counter() - t0
            rec.update(n_prompts=len(group), wall_s=round(elapsed, 4), **stamp)
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            n_tiles += 1
            n_prompts += len(group)
            wall += elapsed
    return {"tiles": n_tiles, "prompts": n_prompts, "wall_s": wall}


def changed_tiles(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Tiles whose prompt set differs between two prompt tables (added, removed or edited
    prompts; geometry and box compared exactly as stored)."""
    cols = ["source", "index", "footprint_wkt", "x0", "y0", "x1", "y1"]

    def sets(frame: pd.DataFrame) -> dict[str, frozenset[tuple[Any, ...]]]:
        return {str(t): frozenset(map(tuple, g[cols].itertuples(index=False, name=None)))
                for t, g in frame.groupby("tile_id")}

    a, b = sets(old), sets(new)
    return sorted(t for t in set(a) | set(b) if a.get(t) != b.get(t))


def _read_records(files: Iterable[Path]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue   # truncated last line of a killed process
            if rec.get("status") == "ok" or rec.get("key") not in best:
                best[rec["key"]] = rec
    return best


def merge(files: Iterable[Path],
          deltas: Iterable[tuple[set[str], Iterable[Path]]] = ()) -> tuple[pd.DataFrame,
                                                                          pd.DataFrame]:
    """Merge result JSONL files by tile key (last good record wins; truncated lines skipped).

    ``deltas`` are applied in order: each is (tiles whose prompts changed, that delta bundle's
    result files). For a delta tile only delta records count: base records of that tile are
    stale (computed from superseded prompts) and are discarded even if the delta has no record
    yet, so the tile is reported missing rather than silently stale.

    Returns (per-prompt results, per-tile status table with ``origin`` base / delta name)."""
    best = _read_records(files)
    origin = dict.fromkeys(best, "base")
    for k, (tiles, delta_files) in enumerate(deltas):
        for tile in tiles:
            best.pop(tile, None)
            origin.pop(tile, None)
        for key, rec in _read_records(delta_files).items():
            if key in tiles:
                best[key] = rec
                origin[key] = f"delta{k}"
    rows, tiles_out = [], []
    for key in sorted(best):
        rec = best[key]
        tiles_out.append({"tile_id": key, "origin": origin[key], "status": rec.get("status"),
                      "n_prompts": rec.get("n_prompts"), "wall_s": rec.get("wall_s"),
                      "device": rec.get("device"), "cuda_device": rec.get("cuda_device"),
                      "model": rec.get("model")})
        for r in rec.get("results", []) if rec.get("status") == "ok" else []:
            rows.append({"tile_id": key, **{k: r[k] for k in ("source", "index", "iou",
                                                              "mask_area_px", "mask_wkt")},
                         "mrr": json.dumps(r["mrr"])})
    columns = ["tile_id", "source", "index", "iou", "mask_area_px", "mask_wkt", "mrr"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(tiles_out)


def agreement_table(results: pd.DataFrame, min_iou: float) -> pd.DataFrame:
    """Per tile: prompts, mean IoU, share with IoU >= ``min_iou`` (``sam2_agreement.csv``)."""
    if results.empty:
        return pd.DataFrame(columns=["tile_id", "prompts", "sam2_mean_iou", "share_accepted"])
    grouped = results.groupby("tile_id")["iou"]
    return pd.DataFrame({
        "prompts": grouped.size(), "sam2_mean_iou": grouped.mean(),
        "share_accepted": grouped.apply(lambda s: float((s >= min_iou).mean())),
    }).reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--tiles", type=Path, required=True, help="directory holding images/")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--part", default="0/1")
    r.add_argument("--model", default="facebook/sam2.1-hiera-small")
    r.add_argument("--device", default="cuda")
    r.add_argument("--mask-threshold", type=float, default=0.0)
    r.add_argument("--batch", type=int, default=64)
    r.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    stats = run(args.prompts, args.tiles, args.out, args.part, args.model, args.device,
                args.mask_threshold, args.batch, args.limit)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
