"""Bundles for the main-river recompute on a remote host. Run from nakkhu/ (laptop):

    uv run python offload/mainriver/make_bundle.py code      [--name NAME]
    uv run python offload/mainriver/make_bundle.py surfaces  [--name NAME]
    uv run python offload/mainriver/make_bundle.py data      [--name NAME]
    uv run python offload/mainriver/make_bundle.py hand      [--name NAME] --run hand-mr/<id>
    uv run python offload/mainriver/make_bundle.py receipts  [--name NAME]

Writes into ``data/interim/mainriver/offload/<NAME>/`` (gitignored):

- ``code.zip``: ``git archive HEAD`` of the code paths + ``BUNDLE_MANIFEST.json``; refuses a
  dirty tree (``configs/paper/`` excepted: the manuscript numbers file is read by no analysis).
- ``surfaces.zip`` (label-free): the two source sweep runs' metadata, their routing rasters for
  the gate-passing DEM x conditioning pairs (fdir, acc, dem_ref; open DEMs only), the grid
  rasters, the inventories (Microsoft footprints, detector_final, greenhouse caches) and the
  plan 014 prep tables of the consistency check.
- ``data.zip``: what the two analyses read: audit items / responses / members of every task,
  every promoted artifact (tables only), the detector run metadata and the detection vectors of
  the sitting-2 pool and the L2 test runs, the eval thresholds, the footprint parquets, the
  GLO-30 and ASF HAND rasters (competitors) and the UNOSAT zip.
- ``hand.zip``: a pulled hand-mr run is already on the host; not needed unless re-shipped.
- ``receipts.zip``: every reserved ``runs/mainriver/*`` (config, env, receipt) + a read-only
  copy of the live journal.

No Maxar pixel is ever bundled: a path naming maxar, an image chip or an imagery mosaic is
refused; .tif files are allowed only from the open DEM / HAND rasters listed here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "3.12.13"
CODE = ["src", "configs", "pyproject.toml", "uv.lock", ".python-version", "README.md",
        "LICENSE", "offload/mainriver", "offload/followup"]
# The bundle is `git archive HEAD`, so uncommitted work elsewhere in the tree never reaches the
# host; these paths (this recompute and what it executes) must equal HEAD all the same.
GUARDED = ["src/nakkhu/mainriver", "configs/hand/main_river_v1.yaml", "configs/mainriver",
           "src/nakkhu/uncertainty", "src/nakkhu/flood", "src/nakkhu/hazard.py",
           "src/nakkhu/evaluate", "pyproject.toml", "uv.lock", "offload/mainriver"]
IMAGE = (".jpg", ".jpeg", ".png", ".jp2", ".vrt", ".ntf", ".img")
TIF_OK = ("data/interim/hand/", "data/raw/dem_glo30/", "data/raw/hand_asf_glo30/")
# Vector / table files whose path names the Maxar scene but which hold no pixel: the WV02 train
# tile index (tile ids, bounds, affine, file names) that the recall frame registers footprints by.
NO_PIXELS = ("data/interim/tiles/maxar_wv02_20240113/train/index.parquet",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def out_dir(name: str) -> Path:
    d = ROOT / "data" / "interim" / "mainriver" / "offload" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def allowed(rel: str) -> None:
    if rel in NO_PIXELS:
        return
    low = rel.lower()
    data = not low.startswith(("src/", "tests/", "configs/", "offload/"))
    if low.endswith(IMAGE) or (data and ("maxar" in low or "/chips" in low or "mosaic" in low)):
        sys.exit(f"refused: {rel} may hold imagery")
    if low.endswith((".tif", ".tiff")) and not rel.startswith(TIF_OK):
        sys.exit(f"refused: {rel} is a raster outside the open DEM / HAND rasters")


def code(name: str, commit: str | None = None) -> Path:
    """``git archive`` of HEAD, or of ``commit`` (the commit a receipt names: another agent may
    commit to the repository between the reservation and the bundle)."""
    if commit is None and git("status", "--porcelain", "--", *GUARDED):
        sys.exit("refused: code paths differ from HEAD (commit first)")
    head = git("rev-parse", commit or "HEAD")
    raw = subprocess.run(["git", "archive", "--format=zip", head, "--", *CODE], cwd=ROOT,
                         check=True, capture_output=True).stdout
    src = zipfile.ZipFile(io.BytesIO(raw))
    files = {}
    path = out_dir(name) / "code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in src.infolist():
            if info.is_dir():
                continue
            data = src.read(info)
            allowed(info.filename)
            files[info.filename] = hashlib.sha256(data).hexdigest()
            zf.writestr(info.filename, data)
        other = git("status", "--porcelain", "--untracked-files=no", "--", "src", "configs")
        man = {"git_head": head, "dirty": False, "python": PY, "files": files,
               "laptop_other_uncommitted": [ln[3:] for ln in other.splitlines() if ln],
               "uv_lock_sha256": files["uv.lock"], "created_utc": stamp(), "kind": "code"}
        zf.writestr("BUNDLE_MANIFEST.json", json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({"bundle": str(path), "git_head": head, "files": len(files),
                      "bytes": path.stat().st_size}))
    return path


def pack(name: str, kind: str, rels: list[str]) -> Path:
    path = out_dir(name) / f"{kind}.zip"
    files = {}
    rels = sorted(dict.fromkeys(rels))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in rels:
            f = ROOT / rel
            if not f.is_file():
                sys.exit(f"missing {rel}")
            allowed(rel)
            files[rel] = sha256(f)
            zf.write(f, rel)
        zf.writestr(f"{kind.upper()}_MANIFEST.json", json.dumps(
            {"kind": kind, "files": files, "created_utc": stamp()}, indent=1, sort_keys=True))
    print(json.dumps({"bundle": str(path), "files": len(files), "bytes": path.stat().st_size}))
    return path


def project():
    sys.path.insert(0, str(ROOT / "src"))
    from nakkhu.config import load_project

    return load_project(ROOT)


def run_meta(ref: str) -> list[str]:
    d = ROOT / "runs" / ref
    return [(d / f).relative_to(ROOT).as_posix() for f in sorted(p.name for p in d.iterdir())
            if (d / f).is_file() and (d / f).stat().st_size < 5_000_000
            and not f.endswith((".tif", ".gpkg"))]


def surfaces(name: str) -> Path:
    p = project()
    from nakkhu.hydro.sweep import load_settings
    from nakkhu.mainriver import spec as S

    cfg = S.load_ensemble(p)
    rels: list[str] = []
    for ref in cfg["sources"].values():
        rels += run_meta(ref)
        interim = S.sweep_interim(p, ref)
        rels += [(interim / "grid" / f).relative_to(ROOT).as_posix()
                 for f in ("cell_index.tif", "domain.tif")]
    for s in S.surfaces(p, cfg):
        if s.computed:
            d = S.sweep_interim(p, s.sweep_run) / f"{s.dem}__{s.conditioning}"
            rels += [(d / f"{k}.tif").relative_to(ROOT).as_posix()
                     for k in ("fdir", "acc", "dem_ref")]
    settings, _ = load_settings(p, cfg["base_config"])
    for inv in cfg["inventories"]:
        rels.append(settings.resolve(settings.v["inventories"][inv]["path"]).relative_to(
            ROOT).as_posix())
        for c in cfg["prep_check"].values():
            rels.append(str(c["file"]).format(inventory=inv))
    rels += [c.relative_to(ROOT).as_posix()
             for c in sorted((ROOT / "data/interim/reference").glob("osm_greenhouses_*.parquet"))]
    return pack(name, "surfaces", rels)


def data(name: str) -> Path:
    p = project()
    from nakkhu.audit import core as audit_core
    from nakkhu.ensemble import final_members
    from nakkhu.followup import run as FR
    from nakkhu.followup import spec as FS
    from nakkhu.hydro.sweep import load_settings

    rels: list[str] = []
    for task in ("precision_pool", "precision_microsoft", "recall_points", "recall_nearmiss",
                 "flood_trace"):
        d = audit_core.task_dir(p, task)
        rels += [f.relative_to(ROOT).as_posix() for f in sorted(d.iterdir())
                 if f.is_file() and f.suffix in (".parquet", ".jsonl", ".json", ".yaml")]
    alloc = ROOT / "audit" / "allocation.yaml"
    if alloc.is_file():
        rels.append("audit/allocation.yaml")
    for f in sorted((ROOT / "artifacts").rglob("*")):
        rel = f.relative_to(ROOT).as_posix()
        if f.is_file() and not rel.lower().endswith((*IMAGE, ".tif", ".pdf")) \
                and "maxar" not in rel.lower():
            rels.append(rel)
    rels.append("runs/detector/registry.json")
    for f in sorted((ROOT / "runs" / "infer").glob("*/inputs.json")):
        rels.append(f.relative_to(ROOT).as_posix())
    cfg = audit_core.load_audit_config(p)
    for s in cfg["tasks"]["precision_pool"]["runs"]:
        rels += run_meta(str(s["run"])) + [f"runs/{s['run']}/detections.gpkg"]
        rels += run_meta(str(s["threshold_from"]))
    for arm, seeds in (("L2", [0, 1, 2]), ("L5", [0, 1, 2, 3, 4])):
        for m in final_members(p, arm, "yolo11n-obb", seeds):
            rels += run_meta(str(m["eval_val"])) + run_meta(str(m["infer_test"]))
            if arm == "L2":
                rels.append(f"runs/{m['infer_test']}/detections.gpkg")
    rels += run_meta("hand/20261005T191226Z-dc9658f5")
    settings, _ = load_settings(p)
    for inv in ("google_v3", "microsoft", "overture", "osm"):
        rels.append(settings.resolve(settings.v["inventories"][inv]["path"]).relative_to(
            ROOT).as_posix())
    rels.append(settings.resolve(settings.v["inventories"]["google_v3"]["confidence_cut"][
        "thresholds"]).relative_to(ROOT).as_posix())
    rels += ["data/raw/dem_glo30/dem_glo30_routing2km.tif",
             "data/raw/hand_asf_glo30/hand_asf_glo30_routing2km.tif"]
    rels.append(FR.unosat_zip(p, FS.load_prep_config(p)).relative_to(ROOT).as_posix())
    rels += [c.relative_to(ROOT).as_posix()
             for c in sorted((ROOT / "data/interim/reference").glob("osm_greenhouses_*.parquet"))]
    rels += list(NO_PIXELS)
    return pack(name, "data", [r for r in rels if (ROOT / r).is_file()])


def extra(name: str, files: list[str]) -> Path:
    return pack(name, "extra", files)


def receipts(name: str) -> Path:
    rels = []
    for rec in sorted((ROOT / "runs" / "mainriver").glob("*/receipt.json")):
        rels += [(rec.parent / f).relative_to(ROOT).as_posix()
                 for f in ("config.resolved.yaml", "env.json", "receipt.json")]
    path = out_dir(name) / "receipts.zip"
    journal = ROOT / "runs/eval/test_journal.jsonl"
    files = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in rels:
            files[rel] = sha256(ROOT / rel)
            zf.write(ROOT / rel, rel)
        zf.writestr("receipts/journal_copy.jsonl", journal.read_bytes())
        files["receipts/journal_copy.jsonl"] = sha256(journal)
        zf.writestr("RECEIPTS_MANIFEST.json", json.dumps(
            {"kind": "receipts", "files": files, "created_utc": stamp()}, indent=1,
            sort_keys=True))
    print(json.dumps({"bundle": str(path), "files": len(files), "bytes": path.stat().st_size}))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["code", "surfaces", "data", "extra", "receipts"])
    ap.add_argument("--name", default="mr1")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--commit", default=None, help="code: archive this commit, not HEAD")
    a = ap.parse_args()
    if a.kind == "extra":
        extra(a.name, a.files)
    elif a.kind == "code":
        code(a.name, a.commit)
    else:
        {"code": code, "surfaces": surfaces, "data": data, "receipts": receipts}[a.kind](a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
