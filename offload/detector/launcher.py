#!/usr/bin/env python3
"""Multi-platform launcher for plan 007 detector units (Kaggle / Colab / any Linux GPU box).

Unit = one (arm, model, seed): ``nakkhu train`` then ``nakkhu infer --region val`` (and
``--region train`` when the unit is marked ``+train``, for the L8 self-training round). Units
are given as ``ARM/MODEL/SEED[+train]``, comma-separated; one worker thread per GPU pulls the
next unit from a shared queue. Finished units write ``<work>/nk/done/<key>.json`` (status,
UTC start/end, wall seconds, GPU, SHA-256 of every output file) and are skipped on a relaunch
with the same work directory (resumable).

Setup uses only the standard library: uv (official installer), Python 3.12.13, a venv, then
the hashed requirements exported from nakkhu's uv.lock (``--require-hashes``). If the lock's
Linux torch build cannot see the GPU, torch/torchvision of the same version are reinstalled
from the PyTorch cu126 index and that is recorded. Layout on the remote:

- code: ``/tmp/nk/src`` (PYTHONPATH), project root ``<work>/nk`` (configs + runs; pulled back);
- data: ``/tmp/nk/data`` (``--set paths.data_root=...``; datasets are built there, not pulled).

    python launcher.py --input DIR --work DIR --units L1/yolo11n-obb/0,L2/yolo11n-obb/0 \
        [--epochs N] [--cfg key=value ...]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path

PY = "3.12.13"
CU126 = "https://download.pytorch.org/whl/cu126"
BASE = Path(os.environ.get("NAKKHU_BASE", "/tmp/nk"))
TORCH = ("2.14.1", "0.29.1")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sh(cmd, log, check=True, env=None) -> int:
    log.write(f"$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\n")
    log.flush()
    res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, shell=isinstance(cmd, str),
                         env=env)
    if check and res.returncode:
        raise SystemExit(f"command failed ({res.returncode}): {cmd}")
    return res.returncode


def cpus() -> int:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


PROBE = r"""
import json, sys, torch
print(json.dumps({"torch": torch.__version__, "cuda_build": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "cuda": torch.cuda.is_available(),
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  "python": sys.version.split()[0]}))
"""

PINCHECK = r"""
import json, sys, importlib.metadata as m
from packaging.requirements import Requirement
bad, ok = {}, 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith(("#", "--")) or "==" not in line:
        continue
    req = Requirement(line.split(" \\")[0].rstrip("\\").strip())
    if req.marker is not None and not req.marker.evaluate():
        continue
    want = str(req.specifier).lstrip("=")
    try:
        have = m.version(req.name)
    except m.PackageNotFoundError:
        have = None
    if have == want:
        ok += 1
    else:
        bad[req.name] = {"lock": want, "installed": have}
print(json.dumps({"matched": ok, "mismatched": bad}))
"""


def setup(code: Path, log) -> dict:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        sh("curl -LsSf https://astral.sh/uv/install.sh | sh", log)
        uv = str(Path.home() / ".local" / "bin" / "uv")
    venv = BASE / "venv"
    py = str(venv / "bin" / "python")
    reqs = str(code / "requirements.txt")
    if not Path(py).exists():
        sh([uv, "python", "install", PY], log)
        sh([uv, "venv", "--python", PY, str(venv)], log)
        sh([uv, "pip", "install", "--python", py, "--require-hashes", "-r", reqs], log)
    probe = json.loads(subprocess.run([py, "-c", PROBE], capture_output=True,
                                      text=True).stdout or "{}")
    fallback = None
    if not probe.get("cuda"):
        fallback = "cu126 (lock build could not use the GPU: " + json.dumps(probe) + ")"
        sh([uv, "pip", "install", "--python", py, "--reinstall-package", "torch",
            "--reinstall-package", "torchvision", "--index-url", CU126,
            f"torch=={TORCH[0]}+cu126", f"torchvision=={TORCH[1]}+cu126"], log)
        probe = json.loads(subprocess.run([py, "-c", PROBE], capture_output=True,
                                          text=True).stdout or "{}")
    if not probe.get("cuda"):
        raise SystemExit(f"no CUDA after fallback: {probe}")
    pins = json.loads(subprocess.run([py, "-c", PINCHECK, reqs], capture_output=True,
                                     text=True).stdout)
    allowed = set()
    if fallback:
        allowed = {"torch", "torchvision", "triton"} | {
            k for k in pins["mismatched"] if k.startswith(("nvidia", "cuda"))}
    hard = {k: v for k, v in pins["mismatched"].items() if k not in allowed}
    if hard:
        raise SystemExit(f"pin check failed: {hard}")
    freeze = subprocess.run([uv, "pip", "freeze", "--python", py], capture_output=True,
                            text=True).stdout.splitlines()
    smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total "
                         "--format=csv,noheader", shell=True, capture_output=True,
                         text=True).stdout.strip().splitlines()
    return {"python_exe": py, "probe": probe, "torch_fallback": fallback, "pin_check": pins,
            "nvidia_smi": smi, "freeze": freeze}


def stage(inp: Path, work: Path, log) -> tuple[Path, Path, dict]:
    """Code -> /tmp/nk/src + <work>/nk/configs; data archives -> /tmp/nk/data."""
    markers = sorted(glob.glob(str(inp / "**" / "_job_nkdet-*.json"), recursive=True))
    if markers:
        code = Path(markers[0]).parent
    else:
        zips = sorted(glob.glob(str(inp / "**" / "code.zip"), recursive=True))
        if not zips:
            raise SystemExit(f"code bundle not found under {inp}")
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(BASE / "code")
        code = Path(sorted(glob.glob(str(BASE / "code" / "**" / "_job_nkdet-*.json"),
                                     recursive=True))[0]).parent
    manifest = json.loads((code / "MANIFEST.json").read_text())
    src = BASE / "src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(code / "src", src)
    root = work / "nk"
    (root / "runs").mkdir(parents=True, exist_ok=True)
    if (root / "configs").exists():
        shutil.rmtree(root / "configs")
    shutil.copytree(code / "configs", root / "configs")
    data = BASE / "data"
    data.mkdir(parents=True, exist_ok=True)
    seen = {}
    wanted = dict(manifest["data_files"])
    for extra in sorted(glob.glob(str(inp / "**" / "extra_nkdet-x-*.json"), recursive=True)):
        wanted.update(json.loads(Path(extra).read_text())["data_files"])
    for name, want in wanted.items():
        hits = sorted(glob.glob(str(inp / "**" / name), recursive=True))
        if not hits:
            raise SystemExit(f"data file {name} not found under {inp}")
        got = sha256(Path(hits[0]))
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != manifest {want}")
        # runs_* archives hold trained units (runs/train/...) for inference-only units and
        # are extracted into the project root; everything else goes to the data root.
        dest = root if name.startswith("runs_") else data
        done = dest / f".extracted_{name}"
        if not done.exists():
            with tarfile.open(hits[0], "r:*") as tf:
                tf.extractall(dest, filter="data")
            done.write_text(got)
        seen[name] = got
    log.write(f"staged code {code} data {sorted(seen)}\n")
    return code, root, manifest


def parse_spec(spec: str) -> tuple[str, str, str, str, list[str]]:
    """``ARM/MODEL/SEED[+train][+test][+wv03][@k=v&k=v]`` -> (arm, model, seed, flags,
    overrides); flags joined by ``+``."""
    body, _, ovr = spec.partition("@")
    body, _, flag = body.partition("+")
    arm, model, seed = body.split("/")
    return arm, model, seed, flag, [o for o in ovr.split("&") if o]


def spec_key(spec: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in spec)


class Units:
    def __init__(self, specs: list[str], done_dir: Path):
        self.q: queue.Queue = queue.Queue()
        for s in specs:
            key = spec_key(s)
            if (done_dir / f"{key}.json").is_file():
                rec = json.loads((done_dir / f"{key}.json").read_text())
                if rec.get("status") == "ok":
                    continue
            self.q.put(s)


def unit_files(root: Path, rels: list[str]) -> dict[str, str]:
    out = {}
    for rel in rels:
        for p in sorted((root / rel).rglob("*")):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = sha256(p)
    return out


def run_unit(spec: str, gpu: int, args, env_info: dict, root: Path, log_dir: Path,
             done_dir: Path) -> dict:
    arm, model, seed, flag, overrides = parse_spec(spec)
    key = spec_key(spec)
    run_model = next((o.split("=", 1)[1] for o in overrides if o.startswith("model=")), model)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(BASE / "src"),
               NAKKHU_ROOT=str(root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               YOLO_OFFLINE="0")
    py = env_info["python_exe"]
    nk = [py, "-c", "import sys; from nakkhu.detector_cli import remote_main; "
          "sys.exit(remote_main(sys.argv[1:]))",
          "--set", f"paths.data_root={BASE / 'data'}"]
    cfg = []
    if args.epochs:
        cfg += ["--cfg", f"epochs={args.epochs}"]
    for c in [*(args.cfg or []), *overrides]:
        cfg += ["--cfg", c]
    rec = {"unit": spec, "key": key, "gpu_index": gpu, "start_utc": utc(), "steps": {},
           "host": platform.node(), "job": args.job}
    t_unit = time.time()
    with open(log_dir / f"{key}.txt", "a") as log:
        tref = f"train/{arm}/{run_model}/seed{seed}"
        flags = flag.split("+")
        steps = [("train", [*nk, "train", "--config", f"configs/train/{model}.yaml",
                            "--arm", arm, "--seed", seed, "--device", "0", "--force", *cfg]),
                 ("infer_val", [*nk, "infer", "--run", tref, "--region", "val",
                                "--device", "0"])]
        if "inferonly" in flags:         # trained unit shipped in a runs_* archive
            if not (root / "runs" / tref / "weights" / "best.pt").is_file():
                raise SystemExit(f"{tref}: no shipped weights for an inference-only unit")
            steps = []
        if "train" in flags:
            steps.append(("infer_train", [*nk, "infer", "--run", tref, "--region", "train",
                                          "--device", "0"]))
        if "test" in flags:              # inference only; scoring is `nakkhu eval` (guard)
            steps.append(("infer_test", [*nk, "infer", "--run", tref, "--region", "test",
                                         "--device", "0"]))
        if "wv03" in flags:              # D5 cross-sensor inference (final model only)
            steps.append(("infer_wv03", [*nk, "infer", "--run", tref, "--region", "test",
                                         "--scene", "maxar_wv03_20230404", "--device", "0"]))
        status = "ok"
        for name, cmd in steps:
            t0 = time.time()
            rc = sh(cmd, log, check=False, env=env)
            rec["steps"][name] = {"rc": rc, "wall_s": round(time.time() - t0, 1)}
            if rc:
                status = f"failed:{name}"
                break
    rec["status"] = status
    rec["end_utc"] = utc()
    rec["wall_s"] = round(time.time() - t_unit, 1)
    rels = [f"runs/train/{arm}/{run_model}/seed{seed}"]
    rec["train_run"] = f"train/{arm}/{run_model}/seed{seed}"
    for inputs in glob.glob(str(root / "runs" / "infer" / "*" / "inputs.json")):
        d = json.loads(Path(inputs).read_text())
        if d.get("train_run") == rec["train_run"]:
            rels.append(Path(inputs).parent.relative_to(root).as_posix())
    rec["files"] = unit_files(root, rels)
    (done_dir / f"{key}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--cfg", action="append")
    ap.add_argument("--job", default="")
    args = ap.parse_args()
    t_start = time.time()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    log = open(work / "setup.txt", "a")
    session = {"start_utc": utc(), "units": args.units, "epochs": args.epochs, "cfg": args.cfg}
    code, root, manifest = stage(Path(args.input), work, log)
    t_stage = time.time() - t_start
    env_info = setup(code, log)
    t_setup = time.time() - t_start
    gpus = env_info["probe"]["gpus"]
    session.update({"job": manifest["job"], "host": platform.node(),
                    "platform": platform.platform(), "cpus": cpus(), "stage_s": round(t_stage, 1),
                    "setup_s": round(t_setup, 1), **env_info})
    args.job = manifest["job"]
    (work / "startup.json").write_text(json.dumps(session, indent=1))
    done_dir = root / "done"
    log_dir = work / "logs"
    done_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    units = Units([u for u in args.units.split(",") if u], done_dir)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                spec = units.q.get_nowait()
            except queue.Empty:
                return
            rec = run_unit(spec, gpu, args, env_info, root, log_dir, done_dir)
            with lock:
                results.append(rec)
                (work / "progress.json").write_text(json.dumps(
                    [{k: r[k] for k in ("unit", "status", "wall_s", "gpu_index")}
                     for r in results], indent=1))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(max(1, len(gpus)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finished = {"start_utc": session["start_utc"], "end_utc": utc(),
                "total_wall_s": round(time.time() - t_start, 1), "setup_s": round(t_setup, 1),
                "gpus": gpus, "results": [{k: r[k] for k in ("unit", "status", "wall_s")}
                                          for r in results]}
    (work / "finished.json").write_text(json.dumps(finished, indent=1))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
