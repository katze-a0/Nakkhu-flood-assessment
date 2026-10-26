"""Plan 007 D3: SAM3 "building" zero-shot detections on the plan 003 infer windows (Kaggle GPU).

    python sam3_infer.py --preset DIR --index .../valinfer/index.parquet --out OUT \
        --region val --variant full|quad [--prompt building] [--batch 4]

Writes ``OUT/detections_px.parquet`` (``tile_id, conf, cls, x1..y4`` in window pixels, the
plan 007 infer-run contract) and ``OUT/inputs.json``. Locally, ``nakkhu detector zeroshot``
turns it into ``runs/infer/<id>/`` (centre ownership + cross-tile NMS via ``nakkhu georef``)
and scores val only.

- Model: KerasHub ``SAM3PromptableConceptImageSegmenter`` (Kaggle model ``keras/sam3``, preset
  ``sam3_pcs``; SAM License; baseline only), JAX backend; TensorFlow (preprocessing only) is kept
  off the GPU.
- ``full``: each 640 px window is one input (resized to the model's 1008 px). ``quad``: four
  384 px crops at offsets 0 / 256 px (each resized to 1008 px), because SAM3 returns at most
  ``num_queries`` instances per input and dense row-house windows hold more buildings than
  that; a crop keeps only instances whose box centre lies in its half of the window (split at
  320 px), so overlapping crops do not duplicate.
- Geometry: the minimum-area rotated rectangle of the instance mask (> 0.5) at crop
  resolution (plan 004's footprint -> OBB conversion); the predicted axis box when the mask is
  empty. ``conf`` = SAM3 score (detection x presence); kept if >= ``--conf-min`` (0.02, the
  detectors' inference floor).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
WINDOW = 640
QUAD = 384
QUAD_OFFSETS = (0, 256)
SPLIT = 320


def crops(variant: str) -> list[tuple[int, int, int, tuple[float, float, float, float]]]:
    """(x0, y0, size, owned box in window px) per crop."""
    if variant == "full":
        return [(0, 0, WINDOW, (0.0, 0.0, float(WINDOW), float(WINDOW)))]
    out = []
    for oy in QUAD_OFFSETS:
        for ox in QUAD_OFFSETS:
            ax0, ax1 = (0.0, SPLIT) if ox == 0 else (SPLIT, WINDOW)
            ay0, ay1 = (0.0, SPLIT) if oy == 0 else (SPLIT, WINDOW)
            out.append((ox, oy, QUAD, (ax0, ay0, float(ax1), float(ay1))))
    return out


def mask_obb(mask: np.ndarray, size: int, box: np.ndarray) -> np.ndarray:
    """4 corners (crop px) of the min-area rectangle of ``mask`` resized to ``size``."""
    import cv2

    m = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
    ys, xs = np.nonzero(m)
    if len(xs) >= 3:
        pts = np.stack([xs, ys], axis=1).astype(np.float32) + 0.5
        rect = cv2.minAreaRect(pts)
        return cv2.boxPoints(rect).astype(np.float32)
    x0, y0, x1, y1 = (box * size).tolist()
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def locate_preset(root: str) -> str:
    p = Path(root)
    if (p / "metadata.json").is_file():
        return str(p)
    hits = sorted(p.rglob("metadata.json"))
    if not hits:
        raise SystemExit(f"no KerasHub preset (metadata.json) under {root}")
    return str(hits[0].parent)


def sha_dir(path: str) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", required=True, choices=("val", "test"))
    ap.add_argument("--variant", required=True, choices=("full", "quad"))
    ap.add_argument("--prompt", default="building")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf-min", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N windows")
    ap.add_argument("--tile-index-ref", required=True,
                    help="tile_index value written to inputs.json (local path template)")
    args = ap.parse_args()

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import jax
    import keras
    import keras_hub
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preset = locate_preset(args.preset)
    t0 = time.time()
    model = keras_hub.models.SAM3PromptableConceptImageSegmenter.from_preset(preset)
    load_s = time.time() - t0
    index = pd.read_parquet(args.index).sort_values("tile_id").reset_index(drop=True)
    if args.limit:
        index = index.head(args.limit)
    root = Path(args.index).parent
    plan = crops(args.variant)
    rows: list[tuple] = []
    n_inputs = 0
    t_pred = 0.0
    capped = 0
    batch_imgs: list[np.ndarray] = []
    batch_meta: list[tuple[str, int, int, int, tuple]] = []

    def flush() -> None:
        nonlocal t_pred, n_inputs, capped
        if not batch_imgs:
            return
        t = time.time()
        pred = model.predict({"images": np.stack(batch_imgs).astype("float32"),
                              "prompts": [args.prompt] * len(batch_imgs)}, verbose=0)
        t_pred += time.time() - t
        n_inputs += len(batch_imgs)
        scores = np.asarray(pred["scores"])
        boxes = np.asarray(pred["boxes"])
        masks = np.asarray(pred["masks"])
        for b, (tile_id, ox, oy, size, own) in enumerate(batch_meta):
            keep = np.flatnonzero(scores[b] >= args.conf_min)
            if len(keep) == scores.shape[1]:
                capped += 1
            for q in keep:
                bx = boxes[b, q]
                cx = ox + (bx[0] + bx[2]) / 2 * size
                cy = oy + (bx[1] + bx[3]) / 2 * size
                if not (own[0] <= cx < own[2] and own[1] <= cy < own[3]):
                    continue
                c = mask_obb(masks[b, q], size, bx) + np.array([ox, oy], dtype=np.float32)
                rows.append((tile_id, float(scores[b, q]), 0, *c.ravel().tolist()))
        batch_imgs.clear()
        batch_meta.clear()

    for k, row in enumerate(index.itertuples()):
        img = np.asarray(Image.open(root / row.image).convert("RGB"))
        if img.shape[:2] != (WINDOW, WINDOW):
            raise SystemExit(f"{row.tile_id}: window is {img.shape}, expected {WINDOW} px")
        for ox, oy, size, own in plan:
            batch_imgs.append(img[oy:oy + size, ox:ox + size])
            batch_meta.append((row.tile_id, ox, oy, size, own))
            if len(batch_imgs) == args.batch:
                flush()
        if k == 0:
            flush()
            print(json.dumps({"first_window": row.tile_id, "rows": len(rows),
                              "load_s": round(load_s, 1)}), flush=True)
    flush()
    frame = pd.DataFrame(rows, columns=["tile_id", "conf", "cls", *CORNERS])
    frame = frame.astype({"conf": "float32", "cls": "int64", **{c: "float32" for c in CORNERS}})
    frame.to_parquet(out / "detections_px.parquet", index=False)
    inputs = {
        "model": f"zeroshot:sam3:{args.prompt}:{args.variant}",
        "zeroshot": True, "region": args.region, "scene": "maxar_wv02_20240113",
        "tile_index": args.tile_index_ref, "tiles": len(index), "detections": len(frame),
        "conf_min": args.conf_min, "prompt": args.prompt, "variant": args.variant,
        "crops": [list(c[:3]) for c in plan], "inputs_per_window": len(plan),
        "inputs_capped_at_num_queries": capped,
        "ms_per_tile": round(1000 * t_pred / max(1, len(index)), 1),
        "ms_per_input": round(1000 * t_pred / max(1, n_inputs), 1),
        "predict_wall_s": round(t_pred, 1), "load_s": round(load_s, 1),
        "weights": "kaggle://keras/sam3/keras/sam3_pcs/1 (SAM License)",
        "preset_listing_sha256": sha_dir(preset),
        "versions": {"keras": keras.__version__, "keras_hub": keras_hub.__version__,
                     "jax": jax.__version__, "tensorflow": tf.__version__,
                     "python": platform.python_version()},
        "devices": [str(d) for d in jax.devices()], "host": platform.node(),
    }
    (out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    print(json.dumps({k: inputs[k] for k in ("model", "region", "detections", "ms_per_tile",
                                             "inputs_capped_at_num_queries")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
