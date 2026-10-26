"""Ultralytics training sets from a label arm and the frozen split (plan 007 Step 1).

``nakkhu dataset build --arm L5 [--task obb|detect]`` writes
``{data_root}/interim/detector/datasets/<arm>-<task>/``:

- ``images/{train,val}/<tile>.{jpg,png}``: the plan 003 WV02 train-grid tile; where the arm has
  an ignore region it is filled with the tile's mean valid colour (``labels.obb.fill_ignore``;
  Ultralytics has no ignore loss) and written losslessly as PNG, otherwise the JPEG is copied
  byte for byte;
- ``labels/{train,val}/<tile>.txt``: the arm's YOLO-OBB labels (``obb``) or their axis-aligned
  envelopes in YOLO detect format (``detect``, baseline D2);
- ``data.yaml`` (Ultralytics) and ``dataset.json`` (tile lists, counts, input hashes).

Train tiles = the arm's admitted tiles (``tiles.txt``: usable ``train``-split tiles); val tiles
= usable ``val``-split tiles with the same arm's labels (Ultralytics checkpoint selection).
Test and buffer tiles are never written: the builder refuses if any appears (``DatasetError``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from nakkhu.config import Project

TASKS = ("obb", "detect")
SCENE = "maxar_wv02_20240113"
TILE_PX = 640


class DatasetError(RuntimeError):
    """A training set would violate the split rules."""


def labels_root(project: Project) -> Path:
    return project.paths.data_root / "interim" / "labels"


def tiles_root(project: Project) -> Path:
    return project.paths.data_root / "interim" / "tiles" / SCENE / "train"


def datasets_root(project: Project) -> Path:
    return project.paths.data_root / "interim" / "detector" / "datasets"


def dataset_dir(project: Project, arm: str, task: str = "obb") -> Path:
    return datasets_root(project) / f"{arm}-{task}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_tiles(project: Project) -> dict[str, dict[str, Any]]:
    """``tile_id -> {split, usable}`` from the frozen split file."""
    from nakkhu.splits import load_split

    frame, _ = load_split(project)
    return {str(t): {"split": str(s), "usable": bool(u)}
            for t, s, u in zip(frame["tile_id"], frame["split"], frame["usable"], strict=True)}


def select(project: Project, arm: str) -> tuple[list[str], list[str]]:
    """(train tiles, val tiles) of an arm, checked against the split."""
    split = split_tiles(project)
    arm_dir = labels_root(project) / arm
    listed = arm_dir / "tiles.txt"
    if not listed.is_file():
        raise DatasetError(f"{listed} is missing (build the arm first: nakkhu labels build)")
    train = sorted(set(listed.read_text(encoding="utf-8").split()))
    val = sorted(t for t, v in split.items() if v["split"] == "val" and v["usable"])
    check_ids(split, train, val)
    return train, val


def check_ids(split: dict[str, dict[str, Any]], train: list[str], val: list[str]) -> None:
    """Refuse any test, buffer or unknown tile, and any train/val overlap."""
    bad = [t for t in [*train, *val] if t not in split]
    if bad:
        raise DatasetError(f"{len(bad)} tiles are not in the split file, e.g. {bad[:3]}")
    for name, ids, want in (("train", train, "train"), ("val", val, "val")):
        wrong = [t for t in ids if split[t]["split"] != want]
        if wrong:
            kinds = sorted({split[t]["split"] for t in wrong})
            raise DatasetError(f"{len(wrong)} {name} tiles have split {kinds}, e.g. {wrong[:3]}")
        unusable = [t for t in ids if not split[t]["usable"]]
        if unusable:
            raise DatasetError(f"{len(unusable)} {name} tiles are not usable, e.g. "
                               f"{unusable[:3]}")
    if set(train) & set(val):
        raise DatasetError("a tile is in both train and val")


def obb_to_detect(text: str) -> str:
    """YOLO-OBB label lines -> YOLO detect lines (axis-aligned envelope, clipped to [0, 1])."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        xy = np.clip(np.array(parts[1:9], dtype=float).reshape(4, 2), 0.0, 1.0)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(f"{parts[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} "
                   f"{x1 - x0:.6f} {y1 - y0:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def _write_tile(tile: str, split_name: str, out: Path, arm_dir: Path, tiles: Path,
                task: str) -> dict[str, Any]:
    from nakkhu.labels.obb import fill_ignore

    src_img = tiles / "images" / f"{tile}.jpg"
    if not src_img.is_file():
        raise DatasetError(f"missing tile image {src_img}")
    label_src = arm_dir / "labels" / f"{tile}.txt"
    text = label_src.read_text(encoding="utf-8") if label_src.is_file() else ""
    ignore_png = arm_dir / "ignore" / f"{tile}.png"
    mask = np.array(Image.open(ignore_png)).astype(bool) if ignore_png.is_file() else None
    filled = bool(mask is not None and mask.any())
    if filled:
        rgb = np.array(Image.open(src_img).convert("RGB"))
        valid_png = tiles / "masks" / f"{tile}.png"
        valid = np.array(Image.open(valid_png)).astype(bool) if valid_png.is_file() else None
        dst = out / "images" / split_name / f"{tile}.png"
        Image.fromarray(fill_ignore(rgb, mask, valid)).save(dst, "PNG")
    else:
        dst = out / "images" / split_name / f"{tile}.jpg"
        shutil.copyfile(src_img, dst)
    if task == "detect":
        text = obb_to_detect(text)
    (out / "labels" / split_name / f"{tile}.txt").write_text(text, encoding="utf-8")
    return {"tile_id": tile, "split": split_name, "instances": len(text.split("\n")) - 1
            if text else 0, "ignore_px": int(mask.sum()) if mask is not None else 0,
            "filled": filled}


def build(project: Project, arm: str, task: str = "obb", *, force: bool = False,
          out: Path | None = None) -> dict[str, Any]:
    """Materialise one arm as an Ultralytics dataset; reuses a complete identical build."""
    if task not in TASKS:
        raise DatasetError(f"task must be one of {TASKS}")
    arm_dir = labels_root(project) / arm
    tiles = tiles_root(project)
    train, val = select(project, arm)
    out = out or dataset_dir(project, arm, task)
    inputs = {"arm": arm, "task": task, "train": train, "val": val,
              "split_sha256": _sha256(project.paths.resolve("configs/splits/split_v1.yaml")),
              "arm_json_sha256": _sha256(arm_dir / "arm.json")
              if (arm_dir / "arm.json").is_file() else None,
              "tiles_txt_sha256": _sha256(arm_dir / "tiles.txt")}
    record = out / "dataset.json"
    if record.is_file() and not force:
        old = json.loads(record.read_text(encoding="utf-8"))
        if old.get("inputs") == inputs:
            return {**old, "reused": True}
    if out.exists():
        shutil.rmtree(out)
    for split_name in ("train", "val"):
        (out / "images" / split_name).mkdir(parents=True)
        (out / "labels" / split_name).mkdir(parents=True)
    rows = [_write_tile(t, "train", out, arm_dir, tiles, task) for t in train]
    rows += [_write_tile(t, "val", out, arm_dir, tiles, task) for t in val]
    data = {"path": out.resolve().as_posix(), "train": "images/train", "val": "images/val",
            "names": {0: "building"}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    counts = {s: {"tiles": sum(r["split"] == s for r in rows),
                  "instances": sum(r["instances"] for r in rows if r["split"] == s),
                  "filled_tiles": sum(r["filled"] for r in rows if r["split"] == s)}
              for s in ("train", "val")}
    result = {"inputs": inputs, "counts": counts, "tile_px": TILE_PX,
              "ignore_fill": "tile mean valid colour (nakkhu.labels.obb.fill_ignore)"}
    record.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return {**result, "reused": False}
