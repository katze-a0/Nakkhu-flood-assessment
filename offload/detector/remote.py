"""Local orchestration of plan 007 detector units on Kaggle and Colab (free tiers only).

    uv run python offload/detector/remote.py kaggle-dataset BUNDLE
    uv run python offload/detector/remote.py kaggle-push BUNDLE SUFFIX --units U,U [--epochs N]
        [--extra OWNER/SLUG ...]
    uv run python offload/detector/remote.py colab-start BUNDLE SESSION --units U,U [--epochs N]
        [--extra-dir DIR ...]
    uv run python offload/detector/remote.py tick BUNDLE [--no-ingest]
    uv run python offload/detector/remote.py status BUNDLE

BUNDLE = bundle name under ``{data_root}/interim/detector/offload/``. Launches are recorded in
``<bundle>/launches.json``. ``tick`` is short and idempotent: it polls each open launch once,
pulls finished ones into ``<bundle>/pulled/<name>/`` (Kaggle: kernel output at the end; Colab:
incrementally whenever new units finished), ingests them (``nakkhu detector ingest``), stops a
finished Colab session, and prints one line per launch. Every Kaggle dataset and kernel is
PRIVATE (Maxar Open Data is CC BY-NC 4.0).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNER = "bishalpanta"
STUB = Path(os.environ.get("TEMP", "C:/Temp")) / "colab_stub"
TERMINAL = ("pulled", "lost")


def bundle_dir(name: str) -> Path:
    return ROOT / "data" / "interim" / "detector" / "offload" / name


def utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def launches(b: Path) -> list[dict]:
    p = b / "launches.json"
    return json.loads(p.read_text()) if p.is_file() else []


def save(b: Path, items: list[dict]) -> None:
    tmp = b / "launches.json.tmp"
    tmp.write_text(json.dumps(items, indent=1))
    tmp.replace(b / "launches.json")


def run(cmd: list[str], cwd: Path | None = None, timeout: float = 600, env=None,
        stdin: str | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                              env=env, encoding="utf-8", errors="replace", input=stdin)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, stdout=str(e.stdout or ""),
                                           stderr=f"timeout after {timeout}s")


def colab(*args: str, timeout: float = 900, stdin: str | None = None
          ) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(STUB), MSYS_NO_PATHCONV="1",
               PYTHONIOENCODING="utf-8")
    return run(["colab", "--auth", "adc", *args], timeout=timeout, env=env, stdin=stdin)


def colab_exec(session: str, code: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return colab("exec", "-s", session, "--timeout", str(timeout), timeout=timeout + 120,
                 stdin=code)


# --- Kaggle ---------------------------------------------------------------------------------


def kaggle_dataset(b: Path, code: str | None) -> int:
    # The CLI needs a one-level relative -p (nested or absolute Windows paths break its
    # upload cache file names).
    parent, rel = (b / "code", code) if code else (b, "dataset")
    meta = json.loads((parent / rel / "dataset-metadata.json").read_text())
    res = run(["kaggle", "datasets", "create", "-p", rel, "--dir-mode", "skip"], cwd=parent,
              timeout=3600)
    print(res.stdout[-2000:], res.stderr[-2000:])
    for _ in range(90):
        st = run(["kaggle", "datasets", "status", meta["id"]])
        print(st.stdout.strip(), st.stderr.strip()[-120:])
        if "ready" in st.stdout.lower():
            return 0
        time.sleep(20)
    return 1


RUN_PY = '''# Kaggle kernel entry (private): run the nakkhu plan 007 launcher on the attached datasets.
import glob, subprocess, sys, zipfile
ARGS = {args!r}
hits = sorted(glob.glob("/kaggle/input/**/launcher.py", recursive=True))
if not hits:
    z = sorted(glob.glob("/kaggle/input/**/code.zip", recursive=True))[0]
    zipfile.ZipFile(z).extractall("/tmp/nkcode")
    hits = sorted(glob.glob("/tmp/nkcode/**/launcher.py", recursive=True))
print("launcher:", hits[0], "args:", ARGS, flush=True)
rc = subprocess.run([sys.executable, hits[0], "--input", "/kaggle/input", "--work",
                     "/kaggle/working", *ARGS]).returncode
print("launcher rc", rc, flush=True)
sys.exit(rc)
'''


def launcher_args(units: str, epochs: int | None, cfg: list[str] | None) -> list[str]:
    a = ["--units", units]
    if epochs:
        a += ["--epochs", str(epochs)]
    for c in cfg or []:
        a += ["--cfg", c]
    return a


def kaggle_push(b: Path, suffix: str, units: str, epochs: int | None, cfg: list[str] | None,
                extra: list[str], code: str) -> int:
    slug = f"nkdet-{suffix}"
    d = b / "kaggle" / f"k_{suffix}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "run.py").write_text(RUN_PY.format(args=launcher_args(units, epochs, cfg)))
    data_id = json.loads((b / "dataset" / "dataset-metadata.json").read_text())["id"]
    code_id = json.loads((b / "code" / code / "dataset-metadata.json").read_text())["id"]
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [data_id, code_id, *extra], "kernel_sources": [],
        "competition_sources": []}, indent=1))
    res = run(["kaggle", "kernels", "push", "-p", f"k_{suffix}"], cwd=b / "kaggle")
    print(res.stdout.strip(), res.stderr.strip()[-1000:])
    ok = res.returncode == 0 and "error" not in res.stdout.lower()
    items = launches(b)
    items.append({"platform": "kaggle", "name": slug, "units": units, "epochs": epochs,
                  "cfg": cfg, "extra": extra, "code": code, "pushed_utc": utc(),
                  "state": "pushed" if ok else "push_failed"})
    save(b, items)
    return 0 if ok else 1


def kaggle_status(slug: str) -> str:
    res = run(["kaggle", "kernels", "status", f"{OWNER}/{slug}"], timeout=120)
    m = re.search(r"(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL\w*)", res.stdout.upper())
    return m.group(1).lower() if m else f"unknown({res.stdout.strip()[:80]}"


# --- Colab ----------------------------------------------------------------------------------


def colab_start(b: Path, session: str, units: str, epochs: int | None, cfg: list[str] | None,
                extra_dirs: list[str], code: str) -> int:
    res = colab("new", "-s", session, "--gpu", "T4", timeout=1200)
    print(res.stdout[-1500:], res.stderr[-1500:])
    if res.returncode:
        return 1
    files = [p for p in sorted((b / "dataset").iterdir()) if p.name != "dataset-metadata.json"]
    files.append(b / "code" / code / "code.zip")
    for d in extra_dirs:
        files += [p for p in sorted(Path(d).iterdir()) if p.name != "dataset-metadata.json"]
    colab_exec(session, "import os; os.makedirs('/content/nkin', exist_ok=True)")
    for p in files:
        t0 = time.time()
        up = colab("upload", "-s", session, str(p), f"/content/nkin/{p.name}", timeout=3600)
        print(p.name, up.returncode, round(time.time() - t0), "s", up.stderr.strip()[-300:])
        if up.returncode:
            return 1
    args = " ".join(launcher_args(units, epochs, cfg))
    ex = colab("exec", "-s", session, "--env", f"NK_ARGS={args}", "--timeout", "120", "-f",
               str(HERE / "colab_launch.py"), timeout=300)
    print(ex.stdout[-1500:], ex.stderr[-800:])
    items = launches(b)
    items.append({"platform": "colab", "name": session, "units": units, "epochs": epochs,
                  "cfg": cfg, "code": code, "pushed_utc": utc(), "state": "running", "pulled_done": 0})
    save(b, items)
    return 0


def colab_pull(b: Path, session: str) -> Path | None:
    pack = ("import subprocess; r=subprocess.run('cd /content/nkwork && tar czf /content/out.tgz "
            "nk/runs nk/done logs startup.json setup.txt $(ls finished.json progress.json "
            "2>/dev/null)', shell=True, capture_output=True, text=True); print(r.returncode, "
            "r.stderr[-300:])")
    colab_exec(session, pack, timeout=600)
    dst = b / "pulled" / session
    dst.mkdir(parents=True, exist_ok=True)
    local = dst / "out.tgz"
    res = colab("download", "-s", session, "/content/out.tgz", str(local), timeout=3600)
    if res.returncode or not local.is_file():
        print("download failed", res.stderr[-500:])
        return None
    with tarfile.open(local) as tf:
        tf.extractall(dst, filter="data")
    local.unlink()
    return dst


# --- tick -----------------------------------------------------------------------------------


def ingest(path: Path) -> str:
    res = run(["uv", "run", "nakkhu", "detector", "ingest", str(path)], cwd=ROOT, timeout=7200)
    tail = (res.stdout.strip().splitlines() or [""])[-1]
    if res.returncode:
        tail += " | " + res.stderr.strip()[-400:]
    return tail


def tick(b: Path, do_ingest: bool) -> int:
    items = launches(b)
    open_n = 0
    for it in items:
        if it["state"] in TERMINAL or it["state"] == "push_failed":
            print(f"{it['platform']:6} {it['name']:28} {it['state']}")
            continue
        if it["platform"] == "kaggle":
            st = kaggle_status(it["name"])
            it["last_status"] = st
            if st in ("complete", "error") or st.startswith("cancel"):
                dst = b / "pulled" / it["name"]
                dst.mkdir(parents=True, exist_ok=True)
                res = run(["kaggle", "kernels", "output", f"{OWNER}/{it['name']}", "-p",
                           str(dst)], timeout=3600)
                it["state"] = "pulled"
                it["final_status"] = st
                it["pulled_utc"] = utc()
                it["pull_rc"] = res.returncode
                if do_ingest:
                    it["ingest"] = ingest(dst)
            else:
                open_n += 1
        else:
            dst = None
            res = colab("exec", "-s", it["name"], "-f", str(HERE / "colab_tick.py"),
                        "--timeout", "60", timeout=180)
            line = next((ln for ln in res.stdout.splitlines() if ln.startswith("{")), None)
            if line is None:
                it["misses"] = it.get("misses", 0) + 1
                it["last_status"] = f"no tick ({res.stderr.strip()[-120:]})"
                if it["misses"] >= 3:
                    it["state"] = "lost"
                open_n += it["state"] not in TERMINAL
            else:
                it["misses"] = 0
                st = json.loads(line)
                it["last_status"] = f"alive={st['alive']} done={st['done']}"
                finished = not st["alive"]
                if st["done"] > it.get("pulled_done", 0) or finished:
                    dst = colab_pull(b, it["name"])
                    if dst is not None:
                        it["pulled_done"] = st["done"]
                        if do_ingest:
                            it["ingest"] = ingest(dst)
                if finished and dst is not None:
                    stop = colab("stop", "-s", it["name"], timeout=300)
                    it["state"] = "pulled"
                    it["stopped"] = stop.returncode == 0
                    it["pulled_utc"] = utc()
                else:
                    open_n += 1
        print(f"{it['platform']:6} {it['name']:28} {it['state']:8} {it.get('last_status', '')}"
              f" {it.get('ingest', '')}")
        save(b, items)
    # Kaggle allows 2 concurrent GPU batch sessions per account: fill free slots from
    # <bundle>/queue.json ([{"suffix", "units", "code", "epochs"?, "extra"?}], FIFO).
    qpath = b / "queue.json"
    queue = json.loads(qpath.read_text()) if qpath.is_file() else []
    running = sum(1 for it in items if it["platform"] == "kaggle"
                  and it["state"] not in TERMINAL and it["state"] != "push_failed")
    while queue and running < 2:
        job = queue[0]
        if job.get("hold"):
            print(f"queue head {job['suffix']} on hold ({job['hold']})")
            break
        rc = kaggle_push(b, job["suffix"], job["units"], job.get("epochs"), job.get("cfg"),
                         job.get("extra", []), job["code"])
        if rc:
            print(f"push of {job['suffix']} failed; stays queued")
            launches_now = launches(b)
            launches_now[-1]["state"] = "push_failed"
            save(b, launches_now)
            shutil.rmtree(b / "kaggle" / f"k_{job['suffix']}", ignore_errors=True)
            break
        queue.pop(0)
        qpath.write_text(json.dumps(queue, indent=1))
        running += 1
        open_n += 1
        print(f"pushed {job['suffix']}")
    print(f"open launches: {open_n}; queued kernels: {len(queue)}")
    return 0 if open_n == 0 and not queue else 3


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ty: ignore[unresolved-attribute]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("kaggle-dataset")
    s.add_argument("bundle")
    s.add_argument("--code", help="upload this code bundle instead of the data")
    s = sub.add_parser("kaggle-push")
    s.add_argument("bundle")
    s.add_argument("suffix")
    s.add_argument("--units", required=True)
    s.add_argument("--epochs", type=int)
    s.add_argument("--cfg", action="append")
    s.add_argument("--extra", nargs="*", default=[])
    s.add_argument("--code", required=True)
    s = sub.add_parser("colab-start")
    s.add_argument("bundle")
    s.add_argument("session")
    s.add_argument("--units", required=True)
    s.add_argument("--epochs", type=int)
    s.add_argument("--cfg", action="append")
    s.add_argument("--extra-dir", nargs="*", default=[])
    s.add_argument("--code", required=True)
    s = sub.add_parser("tick")
    s.add_argument("bundle")
    s.add_argument("--no-ingest", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("bundle")
    a = ap.parse_args()
    b = bundle_dir(a.bundle)
    if a.cmd == "kaggle-dataset":
        return kaggle_dataset(b, a.code)
    if a.cmd == "kaggle-push":
        return kaggle_push(b, a.suffix, a.units, a.epochs, a.cfg, a.extra, a.code)
    if a.cmd == "colab-start":
        return colab_start(b, a.session, a.units, a.epochs, a.cfg, a.extra_dir, a.code)
    if a.cmd == "tick":
        return tick(b, not a.no_ingest)
    print(json.dumps(launches(b), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
