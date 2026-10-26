"""Bundles for the plan 014 follow-up tests on remote hosts. Run from nakkhu/ (laptop):

    uv run python offload/followup/make_bundle.py code     [--name NAME]
    uv run python offload/followup/make_bundle.py data     [--name NAME]
    uv run python offload/followup/make_bundle.py p18      [--name NAME]
    uv run python offload/followup/make_bundle.py receipts [--name NAME] --tests P1-3 P1-4 ...
    uv run python offload/followup/make_bundle.py p2       [--name NAME]

Writes into ``{data_root}/interim/followup/offload/<NAME>/`` (gitignored):

- ``code.zip``: ``git archive HEAD`` of src/, configs/, pyproject.toml, uv.lock, .python-version,
  README.md, LICENSE and offload/followup/, plus ``BUNDLE_MANIFEST.json`` (git HEAD, dirty flag
  of the code paths, sha256 per file, the pinned interpreter). Refuses a dirty tree.
- ``data.zip``: the minimal derived, label-bearing tables the tests read (no imagery):
  the promoted prep (artifacts/followup/prep_*), the two change maps and their sidecars, the
  sitting-3 items and responses, and the flood_trace estimates; ``DATA_MANIFEST.json``.
- ``p18.zip``: P1-8 only (label-free inputs): the Microsoft footprints and the cached greenhouse
  extracts. The OSM PBF (395 MB) is copied separately and the UNOSAT zip is downloaded on the
  host from its public URL; both are sha256-checked by the receipt.
- ``receipts.zip``: after ``nakkhu followup reserve``: each reserved run directory
  (config.resolved.yaml, env.json, receipt.json) and ``receipts/journal_copy.jsonl`` (a
  read-only copy of the live journal).

- ``p2.zip``: every input of ``P2`` pinned in ``configs/followup/phase2/p2_inputs.yaml`` (site
  predictor tables, site change maps, the P2-audit estimates, the M-C model; label-bearing).

No Maxar pixel is ever bundled: every file is a derived table, a vector or a config.
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
        "LICENSE", "offload/followup"]
GUARDED = ["src", "configs", "pyproject.toml", "uv.lock", ".python-version"]
DATA = ["artifacts/flood/change_map_buildings_microsoft.parquet",
        "artifacts/flood/change_map_buildings_microsoft.json",
        "artifacts/flood/change_map_buildings_detector_final.parquet",
        "artifacts/flood/change_map_buildings_detector_final.json",
        "artifacts/audit/flood_trace_estimates.json",
        "audit/flood_trace/items.parquet", "audit/flood_trace/responses.jsonl"]
P18 = ["data/raw/fp_microsoft/fp_microsoft.parquet"]
FORBIDDEN = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2", ".vrt", ".ntf", ".img")


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
    d = ROOT / "data" / "interim" / "followup" / "offload" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def code(name: str) -> Path:
    if git("status", "--porcelain", "--", *GUARDED):
        sys.exit("refused: code paths differ from HEAD (commit first)")
    head = git("rev-parse", "HEAD")
    raw = subprocess.run(["git", "archive", "--format=zip", "HEAD", "--", *CODE], cwd=ROOT,
                         check=True, capture_output=True).stdout
    src = zipfile.ZipFile(io.BytesIO(raw))
    files = {}
    path = out_dir(name) / "code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in src.infolist():
            if info.is_dir():
                continue
            data = src.read(info)
            if info.filename.lower().endswith(FORBIDDEN):
                sys.exit(f"refused: {info.filename} looks like imagery")
            files[info.filename] = hashlib.sha256(data).hexdigest()
            zf.writestr(info.filename, data)
        man = {"git_head": head, "dirty": False, "python": PY, "files": files,
               "uv_lock_sha256": files["uv.lock"], "created_utc": stamp(), "kind": "code"}
        zf.writestr("BUNDLE_MANIFEST.json", json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({"bundle": str(path), "git_head": head, "files": len(files),
                      "bytes": path.stat().st_size}))
    return path


def pack(name: str, kind: str, rels: list[str]) -> Path:
    path = out_dir(name) / f"{kind}.zip"
    files = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in rels:
            f = ROOT / rel
            if not f.is_file():
                sys.exit(f"missing {rel}")
            if f.suffix.lower() in FORBIDDEN:
                sys.exit(f"refused: {rel} looks like imagery")
            files[rel] = sha256(f)
            zf.write(f, rel)
        zf.writestr(f"{kind.upper()}_MANIFEST.json", json.dumps(
            {"kind": kind, "files": files, "created_utc": stamp()}, indent=1, sort_keys=True))
    print(json.dumps({"bundle": str(path), "files": len(files), "bytes": path.stat().st_size}))
    return path


def data(name: str) -> Path:
    prep = json.loads((ROOT / "artifacts/followup/prep_manifest.json").read_text("utf-8"))
    rels = ["artifacts/followup/prep_manifest.json",
            *[f"artifacts/followup/{f}" for f in sorted(prep["files"])], *DATA]
    return pack(name, "data", rels)


def p18(name: str) -> Path:
    cache = sorted((ROOT / "data/interim/reference").glob("osm_greenhouses_*.parquet"))
    return pack(name, "p18", [*P18, *[c.relative_to(ROOT).as_posix() for c in cache]])


def p2(name: str) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from nakkhu.config import load_project
    from nakkhu.followup import confirm

    project = load_project(ROOT)
    doc = confirm.load_inputs(project)
    problems = confirm.check_inputs(project, doc)
    if problems:
        sys.exit("refused: " + "; ".join(problems))
    rels = [project.paths.relative(project.paths.resolve(r))
            for r in confirm.input_files(project, doc)]
    return pack(name, "p2", rels)


def receipts(name: str, tests: list[str]) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from nakkhu.config import load_project
    from nakkhu.followup.remote import RECEIPT_FILE, find_receipt

    project = load_project(ROOT)
    rels = []
    for t in tests:
        rec = find_receipt(project, t)
        run_dir = rec.parent
        rels += [(run_dir / f).relative_to(ROOT).as_posix()
                 for f in ("config.resolved.yaml", "env.json", RECEIPT_FILE)]
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
            {"kind": "receipts", "tests": tests, "files": files, "created_utc": stamp()},
            indent=1, sort_keys=True))
    print(json.dumps({"bundle": str(path), "tests": tests, "bytes": path.stat().st_size}))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["code", "data", "p18", "receipts", "p2"])
    ap.add_argument("--name", default="f014")
    ap.add_argument("--tests", nargs="*", default=[])
    a = ap.parse_args()
    {"code": lambda: code(a.name), "data": lambda: data(a.name), "p18": lambda: p18(a.name),
     "receipts": lambda: receipts(a.name, a.tests), "p2": lambda: p2(a.name)}[a.kind]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
