"""Kaggle kernel entry for plan 014 (private kernel; private datasets only).

Finds the bundle zips under /kaggle/input by marker (mount layout varies), copies them to
/tmp/base, installs uv, runs setup.sh (pinned env), then the job file ``kaggle_job.json`` that
sits next to the zips: ``{"mode": "check"}`` (label-free env check only) or
``{"mode": "run", "host_type": "kaggle", "tasks": [{"test": "P1-6", "shard": "0-3/26",
"jobs": 4}, ...]}``. Tasks run in parallel processes; outputs (run dirs, logs) are copied to
/kaggle/working/.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("/tmp/base")
OUT = Path("/kaggle/working")


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=False, **kw)


def main() -> int:
    hits = glob.glob("/kaggle/input/**/kaggle_job.json", recursive=True)
    if not hits:
        print("no kaggle_job.json under /kaggle/input")
        return 2
    BASE.mkdir(parents=True, exist_ok=True)
    job = json.loads(Path(hits[0]).read_text())
    for name in ("code.zip", "data.zip", "p18.zip", "receipts.zip"):
        # uploaded as <name>.pack so that Kaggle does not auto-extract the zip
        found = glob.glob(f"/kaggle/input/**/{name}.pack", recursive=True)
        if found:
            shutil.copy(found[0], BASE / name)
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
    os.environ["PATH"] = f"{Path.home()}/.local/bin:{os.environ['PATH']}"
    here = BASE / "setup.sh"
    sh(["python3", "-m", "zipfile", "-e", str(BASE / "code.zip"), str(BASE / "code_tmp")])
    shutil.copy(BASE / "code_tmp/offload/followup/setup.sh", here)
    shutil.copy(BASE / "code_tmp/offload/followup/run.sh", BASE / "run.sh")
    shutil.rmtree(BASE / "code_tmp")
    rc = sh(["bash", str(here), str(BASE)]).returncode
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if rc != 0:
        shutil.copytree(BASE / "logs", OUT / "logs", dirs_exist_ok=True)
        return rc
    nk = str(BASE / "root/.venv/bin/nakkhu")
    env = {**os.environ, "NAKKHU_HOST_TYPE": job.get("host_type", "kaggle")}
    chk = sh([nk, "followup", "check-env"], cwd=BASE / "root", env=env, capture_output=True,
             text=True)
    (OUT / "logs/check_env.json").write_text(chk.stdout + chk.stderr)
    if job.get("mode") == "check":
        if job.get("calibrate"):
            cal = sh([nk, "followup", "calibrate", "--fits", str(job["calibrate"])],
                     cwd=BASE / "root", env=env, capture_output=True, text=True)
            (OUT / "logs/calibrate.json").write_text(cal.stdout + cal.stderr)
        shutil.copytree(BASE / "logs", OUT / "logs", dirs_exist_ok=True)
        return chk.returncode
    procs = []
    for t in job["tasks"]:
        args = ["bash", str(BASE / "run.sh"), str(BASE), env["NAKKHU_HOST_TYPE"], t["test"]]
        if t.get("shard"):
            args += [t["shard"], str(t.get("jobs", 1))]
        procs.append(subprocess.Popen(args, env=env))
    codes = [p.wait() for p in procs]
    shutil.copytree(BASE / "logs", OUT / "logs", dirs_exist_ok=True)
    for d in glob.glob(str(BASE / "root/runs/followup/*")):
        shutil.copytree(d, OUT / "runs/followup" / Path(d).name, dirs_exist_ok=True)
    print("exit codes", codes)
    return max(codes) if codes else 0


if __name__ == "__main__":
    sys.exit(main())
