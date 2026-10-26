"""Build the SAM2 refinement offload bundle (plan 004 Step 4). Run from nakkhu/:

    uv run python offload/sam2_refine/make_bundle.py [--shards 2]

Writes ``{data_root}/interim/labels/_sam2/bundle/<NAME>/`` with:

- ``code.zip``: ``sam2_refine.py`` (standalone runner), ``run.py`` (kernel entry), ``MANIFEST.json``
  (sha256 per file, git HEAD/dirty, pinned versions from uv.lock) and the marker ``_job_<NAME>.json``;
- ``data/prompts.parquet`` and ``data/images/*.jpg`` (only prompted tiles; Maxar Open Data,
  CC BY-NC 4.0: upload as a **private** Kaggle dataset, never public);
- ``kaggle/dataset-metadata.json`` and ``kaggle/kernel-metadata.json`` templates (owner = KAGGLE_USER).

``--kind delta --base <bundle> --tiles-file F`` (normally via ``nakkhu labels sam2-delta``) writes
``_sam2/delta/<NAME>/`` with only the changed tiles and ``delta.json``; their results go to
``_sam2/delta/<NAME>/results/`` and ``nakkhu labels sam2-merge`` uses only those for the tiles
listed in ``delta.json`` (base records of those tiles are discarded as stale).

Unit = one tile (key = tile id); ``--part i/P`` takes every P-th tile; results are one JSON line
per tile, resumable, merged locally with ``nakkhu labels sam2-merge`` after copying the result
files to ``{data_root}/interim/labels/_sam2/results/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PINNED = ("torch", "torchvision", "transformers", "numpy", "pandas", "pyarrow", "shapely",
          "rasterio", "pillow")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    out = {}
    for package in lock.get("package", []):
        name = package["name"]
        if name in PINNED and name not in out:
            out[name] = package["version"]
    return out


def git_info() -> dict[str, object]:
    def run(*cmd: str) -> str | None:
        try:
            res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
        except OSError:
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    status = run("git", "status", "--porcelain")
    return {"head": run("git", "rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=2, help="processes = GPUs (Kaggle 2x T4)")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--kind", choices=("bundle", "delta"), default="bundle",
                        help="delta: only --tiles-file tiles, written to _sam2/delta/<name>/")
    parser.add_argument("--base", help="delta: the base bundle name it amends")
    parser.add_argument("--tiles-file", help="delta: changed tile ids, one per line")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    sam2_dir = data_root / "interim" / "labels" / "_sam2"
    tiles = data_root / "interim" / "tiles" / "maxar_wv02_20240113" / "train"
    prompts = pd.read_parquet(sam2_dir / "prompts.parquet")
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M")
    changed: list[str] = []
    if args.kind == "delta":
        if not args.base or not args.tiles_file:
            raise SystemExit("--kind delta needs --base and --tiles-file")
        changed = [t for t in Path(args.tiles_file).read_text().split() if t]
        prompts = prompts.loc[prompts["tile_id"].isin(set(changed))].reset_index(drop=True)
        name = f"nakkhu-sam2-delta-{stamp}"
        out = sam2_dir / "delta" / name
    else:
        name = f"nakkhu-sam2-{stamp}"
        out = sam2_dir / "bundle" / name
    (out / "data" / "images").mkdir(parents=True)
    (out / "kaggle").mkdir()
    prompts_path = out / "data" / "prompts.parquet"
    prompts.to_parquet(prompts_path)
    for image in sorted(prompts["image"].unique()):
        shutil.copy2(tiles / image, out / "data" / "images" / Path(image).name)

    common = json.loads((sam2_dir / "prompts.json").read_text(encoding="utf-8"))
    sources = {"sam2_refine.py": ROOT / "src" / "nakkhu" / "labels" / "sam2_refine.py",
               "run.py": HERE / "run.py"}
    manifest = {
        "job": name, "created_utc": stamp, "git": git_info(), "versions": lock_versions(),
        "model": "facebook/sam2.1-hiera-small", "shards": args.shards,
        "kind": args.kind, "base": args.base,
        "units": int(prompts["tile_id"].nunique()), "prompts": len(prompts),
        "prompts_sha256": sha256(prompts_path), "prompts_info": common,
        "files": {k: sha256(v) for k, v in sources.items()},
    }
    marker = {"job": name, "shards": args.shards}
    with zipfile.ZipFile(out / "code.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for arc, path in sources.items():
            bundle.write(path, arc)
        bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
        bundle.writestr(f"_job_{name}.json", json.dumps(marker))
    shutil.copy2(out / "code.zip", out / "data" / "code.zip")
    owner = "KAGGLE_USER"
    (out / "kaggle" / "dataset-metadata.json").write_text(json.dumps({
        "title": f"{name}-data", "id": f"{owner}/{name}-data",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, indent=1), encoding="utf-8")
    (out / "kaggle" / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{name}", "title": name, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [f"{owner}/{name}-data"],
        "kernel_sources": []}, indent=1), encoding="utf-8")
    shutil.copy2(HERE / "run.py", out / "kaggle" / "run.py")
    if args.kind == "delta":
        # Read by `nakkhu labels sam2-merge`: these tiles take results only from
        # <out>/results/*.jsonl (copy the delta kernel's res_*.jsonl there).
        (out / "results").mkdir()
        (out / "delta.json").write_text(json.dumps({
            "name": name, "base": args.base, "changed_tiles": changed,
            "tiles_with_prompts": int(prompts["tile_id"].nunique()),
            "prompts": len(prompts)}, indent=1), encoding="utf-8")
    print(json.dumps({"bundle": str(out), **{k: manifest[k] for k in
                                             ("units", "prompts", "shards")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
