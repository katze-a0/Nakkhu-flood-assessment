"""Package and push the plan 007 D3 SAM3 zero-shot kernel to Kaggle (free GPU, PRIVATE).

    uv run python offload/sam3/push.py [--name nksam3-MMDDHHMM] [--limit N] [--dry-run]

Builds ``{data_root}/interim/detector/offload/sam3/<name>/`` with
- ``code/``: ``sam3_infer.py``, ``pyproject.toml``, ``uv.lock``, ``requirements.txt`` (hashed,
  exported from this directory's uv.lock), ``MANIFEST.json`` (SHA-256 per file, git HEAD) and
  the marker ``_job_<name>.json``; uploaded as the private dataset ``<name>-code``;
- ``kernel/``: ``run.py`` + ``kernel-metadata.json`` (private, GPU T4 x2, internet on for the
  uv installer) with the base detector dataset (val windows), the test-tile extra (WV02 test
  windows, inference only) and the Kaggle model ``keras/sam3/keras/sam3_pcs/1`` attached.

The kernel runs ``full`` on GPU 0 and ``quad`` on GPU 1, val first, then test (inference only;
test scoring stays behind the prereg guard). Pull with ``kaggle kernels output``; import with
``nakkhu detector zeroshot <pulled dir>``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OWNER = "bishalpanta"
BASE_DATA = f"{OWNER}/nkdet-202610051952-data"
TEST_DATA = f"{OWNER}/nkdet-x-202610052213-data"
MODEL = "keras/sam3/keras/sam3_pcs/1"
PY = "3.12.13"

RUN_PY = r'''
import glob, json, os, platform, shutil, subprocess, sys, tarfile, time
from pathlib import Path

ARGS = {args}
PY = "{py}"
INP, WORK = Path("/kaggle/input"), Path("/kaggle/working")
BASE = Path("/tmp/nks3")
BASE.mkdir(parents=True, exist_ok=True)
log = open(WORK / "setup_log.txt", "a")


def sh(cmd, **kw):
    log.write(f"$ {{cmd}}\n"); log.flush()
    r = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=log, stderr=subprocess.STDOUT, **kw)
    if r.returncode:
        raise SystemExit(f"failed ({{r.returncode}}): {{cmd}}")


t0 = time.time()
code = Path(sorted(glob.glob(str(INP / "**" / "_job_nksam3-*.json"), recursive=True))[0]).parent
uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
if not Path(uv).exists():
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
    uv = str(Path.home() / ".local/bin/uv")
venv = BASE / "venv"
sh([uv, "python", "install", PY])
sh([uv, "venv", "--python", PY, str(venv)])
py = str(venv / "bin/python")
sh([uv, "pip", "install", "--python", py, "--require-hashes", "-r", str(code / "requirements.txt")])
freeze = subprocess.run([uv, "pip", "freeze", "--python", py], capture_output=True, text=True).stdout.splitlines()
smi = subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout.strip().splitlines()
data = BASE / "data"
want = {{"val": "interim/tiles/maxar_wv02_20240113/valinfer/", "test": "interim/tiles/maxar_wv02_20240113/infer/"}}
for pack in glob.glob(str(INP / "**" / "*.tar.pack"), recursive=True):
    name = Path(pack).name
    if not (name == "tiles.tar.pack" or name.startswith("test_tiles_")):
        continue
    with tarfile.open(pack) as tf:
        members = [m for m in tf.getmembers() if any(m.name.startswith(w) for w in want.values())]
        tf.extractall(data, members=members, filter="data")
preset = sorted(glob.glob(str(INP / "**" / "metadata.json"), recursive=True))
preset = [p for p in preset if "sam3" in p.lower()]
if not preset:
    raise SystemExit("SAM3 preset not attached")
startup = {{"job": code.name, "python": PY, "freeze": freeze, "nvidia_smi": smi, "setup_s": round(time.time() - t0, 1),
           "preset": str(Path(preset[0]).parent), "host": platform.node(),
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
(WORK / "startup.json").write_text(json.dumps(startup, indent=1))
env = dict(os.environ, KERAS_BACKEND="jax", XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2")
procs = []
for gpu, variant in ((0, "full"), (1, "quad")):
    cmds = []
    for region in ("val", "test"):
        idx = data / want[region] / "index.parquet"
        ref = "{{data_root}}/" + want[region] + "index.parquet"
        cmds.append([py, str(code / "sam3_infer.py"), "--preset", str(Path(preset[0]).parent),
                     "--index", str(idx), "--out", str(WORK / "out" / f"{{region}}_{{variant}}"),
                     "--region", region, "--variant", variant, "--tile-index-ref", ref, *ARGS])
    script = " && ".join(" ".join(f"'{{c}}'" for c in cmd) for cmd in cmds)
    lf = open(WORK / f"log_{{variant}}.txt", "w")
    procs.append(subprocess.Popen(script, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                  env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))))
rcs = [p.wait() for p in procs]
(WORK / "finished.json").write_text(json.dumps({{"rcs": rcs, "total_wall_s": round(time.time() - t0, 1),
    "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}, indent=1))
print("rcs", rcs, flush=True)
sys.exit(max(rcs))
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stamp = dt.datetime.now(dt.UTC).strftime("%m%d%H%M")
    name = args.name or f"nksam3-{stamp}"
    out = ROOT / "data" / "interim" / "detector" / "offload" / "sam3" / name
    code = out / "code"
    kern = out / "kernel"
    code.mkdir(parents=True)
    kern.mkdir(parents=True)
    req = code / "requirements.txt"
    res = run(["uv", "export", "--frozen", "--no-emit-project", "--format", "requirements-txt",
               "-o", str(req)], cwd=HERE)
    if res.returncode:
        raise SystemExit(res.stderr)
    for f in ("sam3_infer.py", "pyproject.toml", "uv.lock"):
        (code / f).write_bytes((HERE / f).read_bytes())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "offload/sam3"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    manifest = {"job": name, "git_head": git, "offload_sam3_dirty": dirty, "python": PY,
                "model": MODEL, "datasets": [BASE_DATA, TEST_DATA],
                "files": {p.name: sha256(p) for p in sorted(code.iterdir())}}
    (code / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (code / f"_job_{name}.json").write_text(json.dumps({"job": name}))
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": f"{name}-code", "id": f"{OWNER}/{name}-code",
        "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    extra = ["--batch", str(args.batch)] + (["--limit", str(args.limit)] if args.limit else [])
    (kern / "run.py").write_text(RUN_PY.format(args=extra, py=PY))
    (kern / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{name}", "title": name, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [BASE_DATA, TEST_DATA, f"{OWNER}/{name}-code"],
        "model_sources": [MODEL], "kernel_sources": [], "competition_sources": []}, indent=1))
    print(json.dumps({"bundle": str(out), "manifest": manifest}, indent=1))
    if args.dry_run:
        return 0
    res = run(["kaggle", "datasets", "create", "-p", str(code)])
    print(res.stdout.strip(), res.stderr.strip()[-500:])
    if res.returncode or "error" in res.stdout.lower():
        return 1
    for _ in range(30):                       # wait until the new dataset is ready
        st = run(["kaggle", "datasets", "status", f"{OWNER}/{name}-code"])
        if "ready" in st.stdout.lower():
            break
        time.sleep(20)
    res = run(["kaggle", "kernels", "push", "-p", str(kern)])
    print(res.stdout.strip(), res.stderr.strip()[-500:])
    (out / "launch.json").write_text(json.dumps({"name": name, "pushed_utc": dt.datetime.now(
        dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "push_rc": res.returncode,
        "push_out": res.stdout.strip()[-500:]}, indent=1))
    return 0 if res.returncode == 0 and "error" not in res.stdout.lower() else 1


if __name__ == "__main__":
    sys.exit(main())
