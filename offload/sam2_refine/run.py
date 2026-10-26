"""Kaggle kernel entry for the plan 004 SAM2 refinement job (one process per GPU).

Finds the bundle by its marker under /kaggle/input, pins the packages listed in MANIFEST.json
with ``uv pip install --system`` (transformers and friends; torch keeps the image's CUDA build
and its version is recorded),
then runs ``sam2_refine.py run --part i/P`` on GPU i. Results and logs go to /kaggle/working/:
``res_<i>of<P>.jsonl`` (one line per tile), ``startup.json``, ``log_<i>.txt``.
Also runs unchanged on Colab or a Linux GPU box: set NAKKHU_INPUT and NAKKHU_WORK.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path(os.environ.get("NAKKHU_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("NAKKHU_WORK", "/kaggle/working"))
ROOT = Path("/tmp/nakkhu_sam2")


def find(pattern: str) -> Path:
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    if not hits:
        raise SystemExit(f"not found under {INPUT}: {pattern}")
    return Path(hits[0])


def versions(names: list[str]) -> dict[str, str | None]:
    code = ("import importlib.metadata as m, json, sys;"
            "print(json.dumps({n: (lambda n: m.version(n) if n else None)(n) "
            "for n in sys.argv[1:]}))")
    out = subprocess.run([sys.executable, "-c", code, *names], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else {}


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    # Kaggle may auto-extract zips in a dataset: use the extracted marker if present.
    markers = sorted(glob.glob(str(INPUT / "**" / "_job_*.json"), recursive=True))
    if markers:
        for name in ("MANIFEST.json", "sam2_refine.py", "run.py"):
            (ROOT / name).write_bytes((Path(markers[0]).parent / name).read_bytes())
    else:
        with zipfile.ZipFile(find("code.zip")) as bundle:
            bundle.extractall(ROOT)
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    prompts = find("prompts.parquet")
    images = prompts.parent           # data/ holds prompts.parquet and images/
    pins = {k: v for k, v in manifest["versions"].items() if k not in ("torch", "torchvision")}
    before = versions(list(manifest["versions"]))
    if os.environ.get("NAKKHU_NO_PIN") != "1":   # local smoke tests use the uv environment
        # Project rule: uv, never pip. Install uv with its official installer, then pin.
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True,
                       check=False)
        uv = str(Path.home() / ".local" / "bin" / "uv")
        subprocess.run([uv, "pip", "install", "--system", "--python", sys.executable,
                        *[f"{k}=={v}" for k, v in pins.items()]], check=False)
    after = versions(list(manifest["versions"]))
    import torch  # noqa: E402  (after pinning)

    gpus = torch.cuda.device_count()
    shards = int(manifest["shards"])
    startup = {"manifest": manifest, "before": before, "after": after, "gpus": gpus,
               "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpus)],
               "torch_build": torch.__version__, "python": sys.version}
    (WORK / "startup.json").write_text(json.dumps(startup, indent=1))
    procs = []
    for i in range(shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(gpus, 1)),
                   OMP_NUM_THREADS="1", CODE_COMMIT=str(manifest["git"].get("head")),
                   HF_HUB_DISABLE_TELEMETRY="1")
        log = open(WORK / f"log_{i}.txt", "w")
        cmd = [sys.executable, str(ROOT / "sam2_refine.py"), "run", "--prompts", str(prompts),
               "--tiles", str(images), "--out", str(WORK / f"res_{i}of{shards}.jsonl"),
               "--part", f"{i}/{shards}", "--model", manifest["model"],
               "--device", "cuda" if gpus else "cpu"]
        if os.environ.get("NAKKHU_LIMIT"):
            cmd += ["--limit", os.environ["NAKKHU_LIMIT"]]
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    (WORK / "done.json").write_text(json.dumps({"returncodes": codes}))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
