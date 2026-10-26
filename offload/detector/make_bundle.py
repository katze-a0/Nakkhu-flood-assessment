"""Build the plan 007 detector offload bundle. Run from nakkhu/:

    uv run python offload/detector/make_bundle.py [--arms L1 ... L7] [--name NAME]
    uv run python offload/detector/make_bundle.py --kind extra --arms L8 --base <NAME>
    uv run python offload/detector/make_bundle.py --kind extra --test-tiles --base <NAME>

Writes ``{data_root}/interim/detector/offload/<NAME>/dataset/`` for a **private** Kaggle dataset
(Maxar Open Data is CC BY-NC 4.0: never ``--public``) and the same files for Colab upload:

- (separate small dataset ``<bundle>/code/nkdet-code-STAMP/``) ``code.zip``: ``src/nakkhu``, ``configs/{project,tiling}.yaml``, ``configs/train/``,
  ``configs/splits/split_v1.yaml``, ``launcher.py``, ``requirements.txt`` (hashed, exported from
  uv.lock: project dependencies + group ``train``), ``MANIFEST.json`` (git HEAD/dirty, SHA-256
  per file and per data archive, lock hash) and the marker ``_job_<NAME>.json``;
- ``tiles.tar.pack``: WV02 train-grid tiles of the arms' train tiles and the usable val tiles (plus
  masks and the index), and the stride-512 val windows (``valinfer``); never a test or buffer
  tile image;
- ``labels.tgz.pack``: the arms' labels, ignore masks, ``tiles.txt``, ``arm.json`` (train + val
  tiles only);
- ``weights.tar.pack``: the pretrained checkpoints named in ``configs/train/*.yaml`` (SHA-256 checked).

``--kind extra`` writes a small add-on bundle (``extra.tar[.gz]`` + its own code.zip) for L8
labels or the test-region infer tiles; attach it to a kernel next to the base dataset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNER = "bishalpanta"
SCENE = "maxar_wv02_20240113"
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def split_ids() -> dict[str, dict]:
    split = yaml.safe_load((ROOT / "configs" / "splits" / "split_v1.yaml").read_text())
    return split["tiles"]


def add_files(tf: tarfile.TarFile, data_root: Path, files: list[Path]) -> int:
    n = 0
    for f in files:
        tf.add(f, arcname=f.relative_to(data_root).as_posix())
        n += 1
    return n


def code_bundle(b: Path, data_files: dict[str, str], extra: dict) -> Path:
    """``<bundle>/code/<nkdet-code-STAMP>/``: code.zip + metadata for a small private dataset.

    The data archives' SHA-256 (``data_files``) come from the base bundle's
    ``data_manifest.json``; the launcher refuses to run if an attached archive differs."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    name = f"nkdet-code-{stamp}"
    out = b / "code" / name
    out.mkdir(parents=True)
    sources: dict[str, Path] = {}
    for p in sorted((ROOT / "src" / "nakkhu").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            sources[f"src/{p.relative_to(ROOT / 'src').as_posix()}"] = p
    for rel in ("configs/project.yaml", "configs/tiling.yaml", "configs/splits/split_v1.yaml"):
        sources[rel] = ROOT / rel
    for p in sorted((ROOT / "configs" / "train").glob("*.yaml")):
        sources[f"configs/train/{p.name}"] = p
    sources["launcher.py"] = HERE / "launcher.py"
    req = out / "requirements.txt"
    res = subprocess.run(["uv", "export", "--frozen", "--no-default-groups", "--group", "train",
                          "--no-emit-project", "--format", "requirements-txt", "-o", str(req)],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode:
        raise SystemExit(res.stderr)
    sources["requirements.txt"] = req
    base_cfg = yaml.safe_load((ROOT / "configs" / "train" / "base.yaml").read_text())
    manifest = {"job": name, "data_bundle": b.name, "created_utc": stamp, "git": git_info(),
                "lock_sha256": sha256(ROOT / "uv.lock"), "python": "3.12.13",
                "epochs_frozen": base_cfg.get("epochs"), "data_files": data_files, **extra,
                "files": {k: sha256(v) for k, v in sources.items()}}
    with zipfile.ZipFile(out / "code.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, p in sources.items():
            zf.write(p, f"code/{arc}")
        zf.writestr("code/MANIFEST.json", json.dumps(manifest, indent=1))
        zf.writestr(f"code/_job_{name}.json", json.dumps({"job": name}))
    req.unlink()
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": name, "id": f"{OWNER}/{name}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    print(json.dumps({"code_bundle": str(out), "code_zip_sha256": sha256(out / "code.zip")}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["L1", "L2", "L3", "L4", "L5", "L6", "L7"])
    ap.add_argument("--kind", choices=("base", "extra", "code"), default="base")
    ap.add_argument("--base", help="extra: the base bundle it amends")
    ap.add_argument("--test-tiles", action="store_true",
                    help="extra: WV02 + WV03 infer tiles of the analysis extent (inference only)")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="extra: trained units (train/<arm>/<model>/seed<k>) for "
                         "inference-only units (runs_<name>.tar.pack)")
    ap.add_argument("--name")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    args = ap.parse_args()
    data_root = Path(args.data_root).resolve()
    if args.kind == "code":
        b = data_root / "interim" / "detector" / "offload" / args.base
        dm = json.loads((b / "data_manifest.json").read_text())
        code_bundle(b, dm["data_files"], {"data_manifest": dm})
        return 0
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M")
    name = args.name or (f"nkdet-{stamp}" if args.kind == "base" else f"nkdet-x-{stamp}")
    out = data_root / "interim" / "detector" / "offload" / name
    (out / "dataset").mkdir(parents=True)
    split = split_ids()
    tiles_root = data_root / "interim" / "tiles" / SCENE
    labels_root = data_root / "interim" / "labels"
    data_files: dict[str, str] = {}
    # ".pack" extensions: Kaggle auto-extracts .zip/.tar uploads, which would hide the archives
    # whose SHA-256 the launcher checks (tarfile detects the format itself).
    labels_tar = "labels.tgz.pack" if args.kind == "base" else f"labels_{name}.tgz.pack"
    test_tar = f"test_tiles_{name}.tar.pack"
    counts: dict[str, int] = {}

    arm_tiles = {a: set((labels_root / a / "tiles.txt").read_text().split()) for a in args.arms}
    val = {t for t, v in split.items() if v["split"] == "val" and v["usable"]}
    if args.kind == "base":
        train = set().union(*arm_tiles.values())
        bad = [t for t in train if split[t]["split"] != "train"]
        if bad:
            raise SystemExit(f"{len(bad)} arm tiles are not train-split tiles: {bad[:3]}")
        need = sorted(train | val)
        assert all(split[t]["split"] in ("train", "val") for t in need)
        files = [tiles_root / "train" / "index.parquet"]
        for t in need:
            files.append(tiles_root / "train" / "images" / f"{t}.jpg")
            m = tiles_root / "train" / "masks" / f"{t}.png"
            if m.is_file():
                files.append(m)
        vi = tiles_root / "valinfer"
        files += [vi / "index.parquet", *sorted((vi / "images").glob("*.jpg")),
                  *sorted((vi / "masks").glob("*.png"))]
        with tarfile.open(out / "dataset" / "tiles.tar.pack", "w") as tf:
            counts["tiles_files"] = add_files(tf, data_root, files)
        weights = set()
        for cfg in (ROOT / "configs" / "train").glob("*.yaml"):
            v = yaml.safe_load(cfg.read_text())
            if v.get("weights"):
                w = data_root / "interim" / "detector" / "weights" / v["weights"]
                if sha256(w) != v["weights_sha256"]:
                    raise SystemExit(f"{w.name}: SHA-256 differs from {cfg.name}")
                weights.add(w)
        with tarfile.open(out / "dataset" / "weights.tar.pack", "w") as tf:
            counts["weights"] = add_files(tf, data_root, sorted(weights))
        archives = ["tiles.tar.pack", labels_tar, "weights.tar.pack"]
    else:
        need = sorted(set().union(*arm_tiles.values()) | val) if args.arms else []
        archives = [labels_tar] if not args.test_tiles else []
    if args.kind == "base" or not (args.test_tiles or args.runs):
        with tarfile.open(out / "dataset" / labels_tar, "w:gz", compresslevel=6) as tf:
            n = 0
            for a in args.arms:
                d = labels_root / a
                tf.add(d / "tiles.txt", arcname=(d / "tiles.txt").relative_to(data_root)
                       .as_posix())
                if (d / "arm.json").is_file():
                    tf.add(d / "arm.json", arcname=(d / "arm.json").relative_to(data_root)
                           .as_posix())
                for t in need:
                    for f in (d / "labels" / f"{t}.txt", d / "ignore" / f"{t}.png"):
                        if f.is_file():
                            tf.add(f, arcname=f.relative_to(data_root).as_posix())
                            n += 1
            counts["label_files"] = n
    if args.kind == "extra" and args.test_tiles:
        files = []
        for scene in (SCENE, "maxar_wv03_20230404"):
            d = data_root / "interim" / "tiles" / scene / "infer"
            files += [d / "index.parquet", *sorted((d / "images").glob("*.jpg")),
                      *sorted((d / "masks").glob("*.png"))]
        with tarfile.open(out / "dataset" / test_tar, "w") as tf:
            counts["test_tile_files"] = add_files(tf, data_root, files)
        archives = [test_tar]
    if args.kind == "extra" and args.runs:
        runs_tar = f"runs_{name}.tar.pack"
        keep = ("train.json", "config.resolved.yaml", "env.json", "journal.jsonl",
                "summary.json", "results.csv", "args.yaml", "remote.json", "weights/best.pt")
        with tarfile.open(out / "dataset" / runs_tar, "w") as tf:
            for ref in args.runs:
                d = ROOT / "runs" / ref
                tj = json.loads((d / "train.json").read_text())
                if sha256(d / "weights" / "best.pt") != tj["best_pt_sha256"]:
                    raise SystemExit(f"{ref}: best.pt differs from train.json")
                for k in keep:
                    if (d / k).is_file():
                        tf.add(d / k, arcname=f"runs/{ref}/{k}")
        counts["runs"] = len(args.runs)
        archives = [runs_tar]
    for a in archives:
        data_files[a] = sha256(out / "dataset" / a)
    base_epochs = yaml.safe_load((ROOT / "configs" / "train" / "base.yaml").read_text())
    manifest = {"job": name, "kind": args.kind, "base": args.base, "created_utc": stamp,
                "git": git_info(), "lock_sha256": sha256(ROOT / "uv.lock"), "python": "3.12.13",
                "arms": args.arms, "epochs_frozen": base_epochs.get("epochs"),
                "data_files": data_files, "counts": counts,
                "licence": "Maxar Open Data CC BY-NC 4.0 (imagery-derived tiles): PRIVATE only"}
    if args.kind == "extra":
        # An extra bundle carries no code; the launcher verifies its archives through this
        # manifest (written next to the archives).
        (out / "dataset" / f"extra_{name}.json").write_text(json.dumps(manifest, indent=1))
    else:
        (out / "data_manifest.json").write_text(json.dumps(manifest, indent=1))
    title = f"{name}-data"
    (out / "dataset" / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{OWNER}/{title}",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, indent=1))
    sizes = {p.name: p.stat().st_size for p in (out / "dataset").iterdir()}
    print(json.dumps({"bundle": str(out), "counts": counts, "bytes": sizes,
                      "total_mb": round(sum(sizes.values()) / 1e6, 1)}, indent=1))
    if args.kind == "base":
        code_bundle(out, data_files, {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
