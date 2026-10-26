"""Reserve on the laptop, execute on a remote host (plan 014 Step 4; offload-experiments 2.8).

The test journal (``runs/eval/test_journal.jsonl``) is single-writer and lives on the laptop. A
file lock cannot span machines, so a declared test runs in three parts:

1. :func:`reserve` (laptop): every refusal check of the runner (:func:`run.prerequisites`:
   prereg verify, hold off and naming the addendum, run-once, prep sha256), a clean code
   commit, then the run directory, :func:`~nakkhu.evaluate.addendum.record_addendum_access`
   (journal append under the lock) and a receipt ``runs/followup/<run>/receipt.json``: test,
   run, journal entry index + sha256 of that line, addendum sha256, code commit, prep sha256s
   and the sha256 of every label-bearing input the test reads.
2. :func:`execute` (remote host, from a bundle): :func:`verify_receipt` against the shipped,
   read-only journal copy (``receipts/journal_copy.jsonl``), the bundle manifest (code commit,
   file sha256s), the addendum, the prep and the inputs. It refuses on a host that holds the
   live journal (the laptop) and never writes any journal. Labels are read only after the
   verification. Results go into the reserved run directory; nothing is promoted there.
   P1-6 runs as shards (``--shard i/n``): one resumable JSONL per shard, one line per
   (family, held-out block) unit.
3. :func:`promote_test` / :func:`merge_p1_6` (laptop): the pulled run directory is checked
   against the receipt and the live journal (receipt sha256, journal line, code commit, output
   sha256s); only then are the outputs promoted. ``merge`` also requires every expected P1-6
   unit exactly once before it computes the declared summary.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nakkhu.config import Project
from nakkhu.followup import spec as S
from nakkhu.followup.explore import EXPLORE_P2

SCHEMA = 1
RECEIPT_FILE = "receipt.json"
EXECUTION_FILE = "execution.json"
BUNDLE_MANIFEST = "BUNDLE_MANIFEST.json"
JOURNAL_COPY = "receipts/journal_copy.jsonl"
SHARD_DIR = "shards"
SHARDED = ("P1-6",)
# Code that must equal the reserved commit on the executing host.
CODE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock", ".python-version")
HOST_ENV = "NAKKHU_HOST_TYPE"


class ReceiptError(S.FollowupError):
    """A receipt, its run or the executing host does not verify."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_type() -> str:
    return os.environ.get(HOST_ENV, "unknown")


# --- the code commit ------------------------------------------------------------------------


def code_state(project: Project) -> dict[str, Any]:
    """HEAD of the nakkhu repository and whether any code path differs from it."""
    root = project.paths.root

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                 check=False, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", *CODE_PATHS)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def bundle_problems(project: Project, commit: str) -> list[str]:
    """The executing tree is an unpacked bundle of ``commit``: ``BUNDLE_MANIFEST.json`` names
    that commit (clean), every listed file has its sha256, and no code file is unlisted."""
    root = project.paths.root
    path = root / BUNDLE_MANIFEST
    if not path.is_file():
        return [f"{BUNDLE_MANIFEST} missing: tests execute only from a bundle on a remote host"]
    man = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if man.get("git_head") != commit:
        problems.append(f"bundle commit {str(man.get('git_head'))[:12]} != receipt "
                        f"{commit[:12]}")
    if man.get("dirty") is not False:
        problems.append("bundle was built from a dirty tree")
    files = man.get("files") or {}
    for rel, digest in files.items():
        f = root / rel
        if not f.is_file() or sha256_file(f) != digest:
            problems.append(f"bundle file {rel} missing or altered")
    for top in CODE_PATHS:
        base = root / top
        found = [base] if base.is_file() else (
            [p for p in base.rglob("*") if p.is_file()] if base.is_dir() else [])
        for p in found:
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            if rel not in files:
                problems.append(f"code file {rel} is not in the bundle manifest")
    return problems


# --- label-bearing inputs ---------------------------------------------------------------------


def label_inputs(project: Project, cfg: dict[str, Any], test: str) -> dict[str, str]:
    """Every file the scoring of ``test`` reads besides the code, configs and the promoted prep
    (which the prep manifest pins): relative path -> sha256."""
    from nakkhu.audit import core as audit_core
    from nakkhu.flood import validate as V

    if test == "P2-audit":              # the Phase-2 audit items and primary answers
        from nakkhu.followup import p2audit

        out = {}
        for f in p2audit.input_files(project):
            if not Path(f).is_file():
                raise S.FollowupError(f"input {f} missing")
            out[project.paths.relative(f)] = sha256_file(Path(f))
        return out
    if test == "P2":                    # the pinned Phase-2 inputs (confirm.INPUTS)
        from nakkhu.followup import confirm

        doc = confirm.load_inputs(project)
        out = {}
        for rel in confirm.input_files(project, doc):
            f = project.paths.resolve(rel)
            if not f.is_file():
                raise S.FollowupError(f"input {rel} missing")
            out[project.paths.relative(f)] = sha256_file(f)
        return out
    files: list[Path] = []
    for inv in cfg["inventories"]:
        cm = V.change_map_path(project, inv)
        if cm is None:
            raise S.FollowupError(f"change map of {inv} missing")
        files.append(cm)
    files += [audit_core.items_path(project, "flood_trace"),
              audit_core.responses_path(project, "flood_trace"),
              project.paths.artifacts / V.ESTIMATES]
    files += extra_inputs(project, cfg, test)
    out = {}
    for f in files:
        if not Path(f).is_file():
            raise S.FollowupError(f"input {f} missing")
        out[project.paths.relative(f)] = sha256_file(Path(f))
    return out


def extra_inputs(project: Project, cfg: dict[str, Any], test: str) -> list[Path]:
    """P1-8: the UNOSAT zip, the Microsoft footprints and the greenhouse PBF it reads."""
    if test != "P1-8":
        return []
    from nakkhu.followup.run import unosat_zip
    from nakkhu.hydro.sweep import load_settings

    files = [unosat_zip(project, cfg)]
    settings, _ = load_settings(project, cfg["sweep_config"])
    files.append(settings.resolve(settings.v["inventories"]["microsoft"]["path"]))
    gh = settings.v.get("greenhouse_filter") or {}
    if gh.get("enabled"):
        files.append(settings.resolve(gh["pbf"]))
    return files


# --- 1. reserve (laptop) ----------------------------------------------------------------------


def _line_index(project: Project, entry: dict[str, Any]) -> tuple[int, str]:
    """1-based index and sha256 of the journal line holding ``entry``."""
    from nakkhu.evaluate.guard import journal_path

    lines = journal_path(project).read_text(encoding="utf-8").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] and json.loads(lines[i]) == entry:
            return i + 1, sha256_bytes(lines[i].encode("utf-8"))
    raise ReceiptError("the journal entry just written was not found in the journal")


def reserve(project: Project, test: str, *, retry: bool = False, dry_run: bool = False,
            config: str = S.PREP_CONFIG, n_boot: int | None = None) -> dict[str, Any]:
    """Step 1 (laptop only). ``n_boot`` overrides the resample count for synthetic tests only
    (the CLI never passes it)."""
    from nakkhu.config import ResolvedConfig
    from nakkhu.evaluate.addendum import addendum_path, record_addendum_access
    from nakkhu.evaluate.guard import journal_path
    from nakkhu.followup import ml
    from nakkhu.followup import run as R
    from nakkhu.runs import CONFIG_FILE, atomic_write_json, create_run, utc_now

    cfg = S.load_prep_config(project, config)
    spec = S.load_spec(project, cfg["addendum"])
    problems, pre = R.prerequisites(project, test, cfg, spec, retry=retry)
    code = code_state(project)
    if not code["commit"]:
        problems.append("code: no git commit of the nakkhu repository")
    elif code["dirty"] is not False:
        problems.append("code: src/, configs/ or the lock differ from HEAD (commit first; the "
                        "remote bundle must be exactly the reserved commit)")
    inputs: dict[str, str] = {}
    if not problems:
        try:
            inputs = label_inputs(project, cfg, test)
        except S.FollowupError as error:
            problems.append(f"inputs: {error}")
    if problems:
        return {"ready": False, "test": test, "problems": problems}
    if dry_run:
        return {"ready": True, "test": test, "dry_run": True,
                "prep_run": pre["manifest"]["run"], "code_commit": code["commit"]}
    boot, seed = S.bootstrap_settings(spec)
    prep_sha = sha256_file(project.paths.artifacts / S.PREP_MANIFEST)
    values = {"command": "followup-run", "addendum": cfg["addendum"], "test": test,
              "retry": retry, "prep_run": pre["manifest"]["run"],
              "prep_manifest_sha256": prep_sha,
              "n_boot": boot if n_boot is None else int(n_boot), "seed": seed,
              "min_cell": S.min_cell(spec), "execution": "remote (receipt)",
              "code_commit": code["commit"]}
    if test in SHARDED:
        values["p1_6_repeats"] = int(ml.REPEATS)
    resolved = ResolvedConfig(values=values, source=config)
    run = create_run(project, S.RUN_STAGE, resolved)
    if test == EXPLORE_P2:
        access = record_explore_access(project, cfg["addendum"], test, run.ref,
                                       resolved.cfghash)
    else:
        access = record_addendum_access(project, cfg["addendum"], test, run.ref,
                                        resolved.cfghash, retry=retry)
    index, line_sha = _line_index(project, access["entry"])
    receipt = {
        "schema": SCHEMA, "test": test, "addendum": cfg["addendum"],
        "addendum_sha256": access["entry"]["addendum_sha256"],
        "addendum_file_sha256": sha256_file(addendum_path(project, cfg["addendum"])),
        "run": run.ref, "run_cfghash": resolved.cfghash,
        "config_sha256": sha256_file(run.path / CONFIG_FILE), "retry": retry,
        "journal": {"index": index, "line_sha256": line_sha, "entry": access["entry"],
                    "path": project.paths.relative(journal_path(project))},
        "code_commit": code["commit"], "prep_manifest_sha256": prep_sha,
        "prep_files": pre["manifest"]["files"], "inputs": inputs, "values": values,
        "prep_config": config, "reserved_utc": utc_now(),
        "outputs": list(R.outputs_of(test)),
    }
    atomic_write_json(run.path / RECEIPT_FILE, receipt)
    return {"ready": True, "test": test, "run": run.ref,
            "receipt": project.paths.relative(run.path / RECEIPT_FILE),
            "journal_index": index, "journal_note": access["entry"]["note"]}


def record_explore_access(project: Project, addendum_id: str, test: str, run_ref: str,
                          eval_cfghash: str) -> dict[str, Any]:
    """Journal the Phase-2 preparation explore run (note ``<addendum>:<test>``) under the
    journal lock, once: the addendum must verify and declare ``P1-explore``."""
    from nakkhu.evaluate import guard
    from nakkhu.evaluate.addendum import AddendumError, prior_accesses, require_addendum

    if not test.startswith("explore "):
        raise ReceiptError(f"{test!r} is not an explore run")
    with guard.journal_lock(project):
        state = require_addendum(project, addendum_id, "P1-explore")
        if prior_accesses(project, addendum_id, test):
            raise AddendumError(f"{addendum_id}:{test} already ran; it runs once")
        return guard.record_test_access(project, run_ref, eval_cfghash,
                                        state["prereg_sha256"], state["split_sha256"],
                                        note=f"{addendum_id}:{test}",
                                        extra={"addendum_sha256": state["addendum_sha256"]})


# --- 2. verify + execute (remote host) ----------------------------------------------------------


def load_receipt(path: str | Path) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    receipt = json.loads(data.decode("utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ReceiptError(f"{path}: receipt schema {receipt.get('schema')} != {SCHEMA}")
    return receipt, sha256_bytes(data)


def _expected_note(receipt: dict[str, Any]) -> str:
    return f"{receipt['addendum']}:{receipt['test']}" + (" retry" if receipt["retry"] else "")


def _journal_line_problems(lines: list[str], receipt: dict[str, Any], where: str) -> list[str]:
    j = receipt["journal"]
    idx = int(j["index"])
    if not 1 <= idx <= len(lines):
        return [f"{where}: has {len(lines)} lines, receipt names line {idx}"]
    line = lines[idx - 1]
    problems = []
    if sha256_bytes(line.encode("utf-8")) != j["line_sha256"]:
        problems.append(f"{where}: line {idx} sha256 differs from the receipt")
    entry = json.loads(line)
    if entry.get("run") != receipt["run"]:
        problems.append(f"{where}: line {idx} is run {entry.get('run')}, not {receipt['run']}")
    if entry.get("note") != _expected_note(receipt):
        problems.append(f"{where}: line {idx} note {entry.get('note')!r} != "
                        f"{_expected_note(receipt)!r}")
    if entry.get("addendum_sha256") != receipt["addendum_sha256"]:
        problems.append(f"{where}: line {idx} addendum sha256 differs")
    if entry.get("eval_cfghash") != receipt["run_cfghash"]:
        problems.append(f"{where}: line {idx} cfghash differs from the run")
    base = f"{receipt['addendum']}:{receipt['test']}"
    same = [i + 1 for i, ln in enumerate(lines) if ln and
            str(json.loads(ln).get("note") or "") in (base, f"{base} retry")]
    want = 2 if receipt["retry"] else 1
    if len(same) != want or same[-1] != idx:
        problems.append(f"{where}: {base} has journal lines {same}; the receipt's line {idx} "
                        f"must be its {'retry' if receipt['retry'] else 'only access'}")
    return problems


def _run_dir(project: Project, receipt: dict[str, Any]) -> Path:
    stage, run_id = receipt["run"].split("/", 1)
    return project.paths.runs / stage / run_id


def verify_receipt(project: Project, receipt_path: str | Path, test: str
                   ) -> tuple[dict[str, Any], str]:
    """Remote-side check before any label is read; raises :class:`ReceiptError`."""
    from nakkhu.evaluate.addendum import addendum_path
    from nakkhu.evaluate.guard import journal_path
    from nakkhu.runs import CONFIG_FILE

    receipt, rsha = load_receipt(receipt_path)
    problems: list[str] = []
    if receipt["test"] != test:
        raise ReceiptError(f"receipt is for {receipt['test']}, not {test}")
    if journal_path(project).exists():
        problems.append("this host holds the live test journal (the laptop): declared tests "
                        "execute only on a remote host, from a bundle, with a receipt")
    problems += bundle_problems(project, receipt["code_commit"])
    copy = project.paths.root / JOURNAL_COPY
    if not copy.is_file():
        problems.append(f"{JOURNAL_COPY} (the shipped journal copy) missing")
    else:
        lines = copy.read_text(encoding="utf-8").splitlines()
        problems += _journal_line_problems(lines, receipt, JOURNAL_COPY)
    add = addendum_path(project, receipt["addendum"])
    if not add.is_file() or sha256_file(add) != receipt["addendum_file_sha256"]:
        problems.append(f"addendum {receipt['addendum']} file differs from the receipt")
    pm = project.paths.artifacts / S.PREP_MANIFEST
    if not pm.is_file() or sha256_file(pm) != receipt["prep_manifest_sha256"]:
        problems.append("prep manifest differs from the receipt")
    for name, digest in receipt["prep_files"].items():
        f = project.paths.artifacts / S.ARTIFACT_DIR / name
        if not f.is_file() or sha256_file(f) != digest:
            problems.append(f"prep file {name} missing or altered")
    for rel, digest in receipt["inputs"].items():
        f = project.paths.resolve(rel)
        if not f.is_file() or sha256_file(f) != digest:
            problems.append(f"input {rel} missing or altered")
    run_dir = _run_dir(project, receipt)
    cf = run_dir / CONFIG_FILE
    if not cf.is_file() or sha256_file(cf) != receipt["config_sha256"]:
        problems.append(f"run {receipt['run']}: {CONFIG_FILE} missing or not the reserved one")
    if problems:
        raise ReceiptError("receipt refused: " + "; ".join(problems))
    return receipt, rsha


def _ctx(project: Project, receipt: dict[str, Any], spec: dict[str, Any]) -> Any:
    from nakkhu.followup import analyses as A

    v = receipt["values"]
    return A.Ctx(n_boot=int(v["n_boot"]), seed=int(v["seed"]), min_cell=int(v["min_cell"]),
                 edges=S.tier_edges(spec))


def _env() -> dict[str, Any]:
    from nakkhu.runs import collect_env

    env = collect_env(Path.cwd())
    env.pop("git", None)
    return {**env, "host_type": host_type()}


def execute(project: Project, receipt_path: str | Path, test: str, *, jobs: int = 1,
            shard: str | None = None) -> dict[str, Any]:
    """Step 2 (remote host). Verify, then read labels, score and write into the run dir."""
    from nakkhu.flood.validate import jsonable
    from nakkhu.followup import prep as P
    from nakkhu.followup import run as R
    from nakkhu.runs import SUMMARY_FILE, Run, atomic_write_json, utc_now

    receipt, rsha = verify_receipt(project, receipt_path, test)
    if test in SHARDED:
        if shard is None:
            raise ReceiptError(f"{test} runs as shards: pass --shard i/n")
        return execute_shards(project, receipt, rsha, shard, jobs=jobs)
    if shard is not None:
        raise ReceiptError(f"{test} is not sharded")
    run_dir = _run_dir(project, receipt)
    if (run_dir / EXECUTION_FILE).exists() or (run_dir / SUMMARY_FILE).exists():
        raise ReceiptError(f"run {receipt['run']} was already executed")
    cfg = S.load_prep_config(project, receipt["prep_config"])
    spec = S.load_spec(project, receipt["addendum"])
    prep, manifest = P.load_prep(project)
    ctx = _ctx(project, receipt, spec)
    stage, run_id = receipt["run"].split("/", 1)
    run = Run(stage=stage, run_id=run_id, path=run_dir)
    started = utc_now()
    step = "explore-p2" if test == EXPLORE_P2 else test.lower()
    with run.step(step):
        written = {}
        if test == "P2-audit":
            from nakkhu.followup import p2audit as PA

            tables, info = PA.score(project, spec, n_boot=ctx.n_boot, seed=ctx.seed)
            for name, value in tables.items():
                PA.write_output(run.path / name, value)
                written[name] = sha256_file(run.path / name)
        elif test == "P2":
            from nakkhu.followup import confirm as C

            tables, info = C.score(project, cfg, spec, n_boot=ctx.n_boot, seed=ctx.seed)
            for name, value in tables.items():
                C.write_output(run.path / name, value)
                written[name] = sha256_file(run.path / name)
        elif test == EXPLORE_P2:
            from nakkhu.followup import explore as E

            tables, info = E.score(project, cfg, spec, prep)
            for name, value in tables.items():
                E.write_output(run.path / name, value)
                written[name] = sha256_file(run.path / name)
        else:
            tables, info = R.score_test(project, test, cfg, spec, prep, manifest, ctx, jobs)
            for name, frame in tables.items():
                frame.assign(test=test, reporting=f"post hoc (prereg_v9_a3 {test})").to_csv(
                    run.path / name, index=False, lineterminator="\n")
                written[name] = sha256_file(run.path / name)
        result = jsonable({**receipt["values"], "run": run.ref,
                           "test_access": receipt["journal"]["entry"], **info,
                           "outputs": list(tables)})
        tag = "explore_p2" if test == EXPLORE_P2 else test.lower().replace("-", "_")
        atomic_write_json(run.path / f"{tag}_summary.json", result)
        run.write_summary({"test": test, "outputs": list(tables),
                           "journal_note": receipt["journal"]["entry"]["note"],
                           "executed_on": host_type()})
        atomic_write_json(run.path / EXECUTION_FILE, jsonable({
            "test": test, "run": run.ref, "receipt_sha256": rsha,
            "code_commit": receipt["code_commit"], "host_type": host_type(),
            "started_utc": started, "finished_utc": utc_now(), "outputs": written,
            "env": _env()}))
    return {"ready": True, "test": test, "run": run.ref, "outputs": written,
            "host_type": host_type()}


# --- P1-6 shards ------------------------------------------------------------------------------


def parse_shard(spec: str) -> tuple[list[int], int]:
    """``i/n``, ``a-b/n`` (inclusive) or ``i,j,k/n``; 0 <= i < n."""
    m = re.fullmatch(r"\s*([\d,\-\s]+)/(\d+)\s*", spec)
    if not m:
        raise ReceiptError(f"--shard {spec!r}: expected i/n, a-b/n or i,j/n")
    n = int(m.group(2))
    idx: list[int] = []
    for part in m.group(1).split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(v) for v in part.split("-", 1))
            idx += list(range(a, b + 1))
        elif part:
            idx.append(int(part))
    if n < 1 or not idx or any(not 0 <= i < n for i in idx) or len(set(idx)) != len(idx):
        raise ReceiptError(f"--shard {spec!r}: indices must be distinct and in [0, {n})")
    return sorted(idx), n


def shard_path(run_dir: Path, i: int, n: int) -> Path:
    return run_dir / SHARD_DIR / f"p1_6_{i:03d}of{n:03d}.jsonl"


def read_units(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Complete lines of a shard file and whether a truncated last line was dropped."""
    if not path.is_file():
        return [], False
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    out, dropped = [], False
    for k, line in enumerate(lines):
        if not line:
            continue
        last = k == len(lines) - 1          # no trailing newline: written partially
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped = True
            continue
        if last:
            dropped = True
            continue
        out.append(rec)
    return out, dropped


def _b64(a: Any) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode("ascii")


def _unb64(text: str, shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(tuple(shape)).copy()


def encode_unit(u: dict[str, Any]) -> dict[str, Any]:
    from nakkhu.flood.validate import jsonable

    perm = np.asarray(u["perm"], dtype=float)
    return jsonable({k: v for k, v in u.items() if k not in ("p", "p_det", "perm")} | {
        "p": _b64(u["p"]), "n": len(u["p"]), "p_det": _b64(u["p_det"]),
        "n_det": len(u["p_det"]), "perm": _b64(perm), "perm_shape": list(perm.shape)})


def decode_unit(rec: dict[str, Any]) -> dict[str, Any]:
    return {**rec, "p": _unb64(rec["p"], [rec["n"]]),
            "p_det": _unb64(rec["p_det"], [rec["n_det"]]),
            "perm": _unb64(rec["perm"], rec["perm_shape"])}


def _p1_6_data(project: Project, receipt: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Labels (after verification) -> the P1-6 rows."""
    from nakkhu.followup import ml
    from nakkhu.followup import prep as P
    from nakkhu.followup import run as R

    spec = S.load_spec(project, receipt["addendum"])
    prep, _ = P.load_prep(project)
    labels = R.load_labels(project, list(prep))
    refs = R.references(prep, labels)
    ms = R.pick(refs, ("microsoft", "R1"))[0]
    det = (R.pick(refs, ("detector_final", "R1")) or [None])[0]
    ml.check_spec(spec)
    return ml.prepare(ms, det), spec


def assigned_keys(keys: Sequence[str], i: int, n: int) -> list[str]:
    return [k for j, k in enumerate(keys) if j % n == i]


def execute_shards(project: Project, receipt: dict[str, Any], rsha: str, shard: str, *,
                   jobs: int = 1) -> dict[str, Any]:
    """Run the units of the selected shards; resumable (finished keys are skipped)."""
    from nakkhu.followup import ml
    from nakkhu.runs import atomic_write_bytes, atomic_write_json, utc_now

    idx, n = parse_shard(shard)
    run_dir = _run_dir(project, receipt)
    data, spec = _p1_6_data(project, receipt)
    keys = ml.unit_keys(data)
    if n > len(keys):
        raise ReceiptError(f"{n} shards for {len(keys)} units")
    owner: dict[str, int] = {}
    pending: list[str] = []
    for i in idx:
        path = shard_path(run_dir, i, n)
        done, dropped = read_units(path)
        if dropped:                         # rewrite without the partial line
            atomic_write_bytes(path, "".join(json.dumps(r, sort_keys=True) + "\n"
                                             for r in done).encode("utf-8"))
        have = {r["key"] for r in done}
        for k in assigned_keys(keys, i, n):
            owner[k] = i
            if k not in have:
                pending.append(k)
    status = run_dir / SHARD_DIR / f"status_{'_'.join(map(str, idx))}of{n:03d}.json"
    meta = {"run": receipt["run"], "receipt_sha256": rsha, "code_commit": receipt["code_commit"],
            "host_type": host_type(), "shards": idx, "n_shards": n, "pending": len(pending),
            "started_utc": utc_now(), "env": _env()}
    atomic_write_json(status, meta)
    repeats = int(receipt["values"].get("p1_6_repeats", ml.REPEATS))
    seed = int(receipt["values"]["seed"])
    finished = 0
    for u in ml.run_units(pending, spec, data, repeats=repeats, seed=seed, jobs=jobs):
        rec = encode_unit(u) | {"receipt_sha256": rsha, "code_commit": receipt["code_commit"],
                                "host_type": host_type(), "shard": owner[u["key"]],
                                "n_shards": n, "utc": utc_now()}
        path = shard_path(run_dir, owner[u["key"]], n)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finished += 1
    atomic_write_json(status, {**meta, "finished_now": finished, "finished_utc": utc_now()})
    return {"ready": True, "test": receipt["test"], "run": receipt["run"], "shards": idx,
            "n_shards": n, "units_run": finished, "host_type": host_type()}


# --- 3. laptop: verify, merge, promote --------------------------------------------------------


def find_receipt(project: Project, test: str, run: str | None = None) -> Path:
    """The receipt of ``test`` (the latest by journal line if a retry exists)."""
    base = project.paths.runs / S.RUN_STAGE
    found = []
    for path in sorted(base.glob(f"*/{RECEIPT_FILE}")) if base.is_dir() else []:
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("test") == test and (run is None or r.get("run") == run):
            found.append((int(r["journal"]["index"]), path))
    if not found:
        raise ReceiptError(f"no receipt for {test} under runs/{S.RUN_STAGE}/")
    return max(found)[1]


def verify_local(project: Project, receipt_path: Path) -> tuple[dict[str, Any], str]:
    """Laptop-side: the receipt matches the live journal line and the reserved run config."""
    from nakkhu.evaluate.guard import journal_path
    from nakkhu.runs import CONFIG_FILE

    receipt, rsha = load_receipt(receipt_path)
    jp = journal_path(project)
    if not jp.is_file():
        raise ReceiptError("live test journal missing: promotion happens on the laptop")
    problems = _journal_line_problems(jp.read_text(encoding="utf-8").splitlines(), receipt,
                                      "live journal")
    cf = _run_dir(project, receipt) / CONFIG_FILE
    if not cf.is_file() or sha256_file(cf) != receipt["config_sha256"]:
        problems.append(f"run {receipt['run']}: {CONFIG_FILE} is not the reserved one")
    if problems:
        raise ReceiptError("verification refused: " + "; ".join(problems))
    return receipt, rsha


def promote_test(project: Project, test: str, *, run: str | None = None) -> dict[str, Any]:
    """Check the pulled run against receipt + live journal, then promote its outputs."""
    from nakkhu.artifacts import promote
    from nakkhu.runs import journal_status

    if test in SHARDED:
        raise ReceiptError(f"{test} is promoted by `nakkhu followup merge --test {test}`")
    receipt, rsha = verify_local(project, find_receipt(project, test, run))
    run_dir = _run_dir(project, receipt)
    ex_path = run_dir / EXECUTION_FILE
    if not ex_path.is_file():
        raise ReceiptError(f"run {receipt['run']}: {EXECUTION_FILE} missing (not pulled yet?)")
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    problems = []
    if ex.get("receipt_sha256") != rsha:
        problems.append("execution was not made from this receipt")
    if ex.get("code_commit") != receipt["code_commit"] or ex.get("run") != receipt["run"]:
        problems.append("execution commit or run differs from the receipt")
    if sorted(ex.get("outputs") or {}) != sorted(receipt["outputs"]):
        problems.append(f"outputs {sorted(ex.get('outputs') or {})} != declared "
                        f"{sorted(receipt['outputs'])}")
    for name, digest in (ex.get("outputs") or {}).items():
        f = run_dir / name
        if not f.is_file() or sha256_file(f) != digest:
            problems.append(f"output {name} missing or altered since execution")
    if journal_status(run_dir) != "complete":
        problems.append(f"run journal is {journal_status(run_dir)}")
    if problems:
        raise ReceiptError("promotion refused: " + "; ".join(problems))
    promoted = [promote(run_dir / name, f"{S.ARTIFACT_DIR}/{name}", project.paths,
                        replace=False)["path"] for name in receipt["outputs"]]
    return {"ready": True, "test": test, "run": receipt["run"], "host_type": ex.get("host_type"),
            "journal_index": receipt["journal"]["index"], "promoted": promoted}


def collect_units(run_dir: Path, rsha: str, commit: str
                  ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Every unit line of every shard file; refuses foreign receipts or duplicate keys."""
    units: dict[str, dict[str, Any]] = {}
    seen: dict[str, list[str]] = {}
    problems = []
    truncated = []
    for path in sorted((run_dir / SHARD_DIR).glob("p1_6_*of*.jsonl")):
        recs, dropped = read_units(path)
        if dropped:
            truncated.append(path.name)
        for rec in recs:
            seen.setdefault(rec["key"], []).append(path.name)
            if rec.get("receipt_sha256") != rsha or rec.get("code_commit") != commit:
                problems.append(f"{path.name}: unit {rec['key']} is from another receipt or "
                                "commit")
            units[rec["key"]] = rec
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        problems.append(f"{len(dups)} duplicate unit key(s), e.g. {next(iter(dups.items()))}")
    if problems:
        raise ReceiptError("merge refused: " + "; ".join(problems[:10]))
    hosts: dict[str, int] = {}
    for rec in units.values():
        hosts[rec.get("host_type", "unknown")] = hosts.get(rec.get("host_type", "unknown"),
                                                           0) + 1
    return units, {"n_units": len(units), "duplicates": 0, "truncated_lines_ignored": truncated,
                   "units_by_host": hosts}


def merge_p1_6(project: Project, *, run: str | None = None) -> dict[str, Any]:
    """Laptop: all expected units exactly once, then the declared P1-6 summary, promoted."""
    from nakkhu.artifacts import promote
    from nakkhu.flood.validate import jsonable
    from nakkhu.followup import ml
    from nakkhu.runs import SUMMARY_FILE, Run, atomic_write_json, journal_status, utc_now

    test = "P1-6"
    receipt, rsha = verify_local(project, find_receipt(project, test, run))
    run_dir = _run_dir(project, receipt)
    if journal_status(run_dir) != "empty" or (run_dir / SUMMARY_FILE).exists():
        raise ReceiptError(f"run {receipt['run']} was already merged")
    units_raw, info = collect_units(run_dir, rsha, receipt["code_commit"])
    data, spec = _p1_6_data(project, receipt)
    expected = ml.unit_keys(data)
    missing = sorted(set(expected) - set(units_raw))
    extra = sorted(set(units_raw) - set(expected))
    if missing or extra:
        raise ReceiptError(f"merge refused: {len(missing)} unit(s) missing (e.g. "
                           f"{missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")
    units = {k: decode_unit(v) for k, v in units_raw.items()}
    repeats = int(receipt["values"].get("p1_6_repeats", ml.REPEATS))
    ctx = _ctx(project, receipt, spec)
    stage, run_id = receipt["run"].split("/", 1)
    rn = Run(stage=stage, run_id=run_id, path=run_dir)
    with rn.step("p1-6"):
        models, importance, details = ml.summarize(units, data, spec, ctx, repeats=repeats)
        tables = {"p1_6_models.csv": models, "p1_6_importance.csv": importance}
        written = {}
        for name, frame in tables.items():
            frame.assign(test=test, reporting=f"post hoc (prereg_v9_a3 {test})").to_csv(
                run_dir / name, index=False, lineterminator="\n")
            written[name] = sha256_file(run_dir / name)
        cpu = {f: float(sum(u.get("cpu_s", 0.0) for u in units.values() if u["family"] == f))
               for f in ml.FAMILIES}
        atomic_write_json(run_dir / "p1_6_summary.json", jsonable({
            **receipt["values"], "run": receipt["run"],
            "test_access": receipt["journal"]["entry"], "folds": details, "merge": info,
            "unit_cpu_s_by_family": cpu, "outputs": list(tables)}))
        rn.write_summary({"test": test, "outputs": list(tables),
                          "journal_note": receipt["journal"]["entry"]["note"],
                          "executed_on": info["units_by_host"], "merged_on": "laptop"})
        atomic_write_json(run_dir / EXECUTION_FILE, jsonable({
            "test": test, "run": receipt["run"], "receipt_sha256": rsha,
            "code_commit": receipt["code_commit"], "host_type": info["units_by_host"],
            "merged_utc": utc_now(), "outputs": written, "merge": info}))
    promoted = [promote(run_dir / name, f"{S.ARTIFACT_DIR}/{name}", project.paths,
                        replace=False)["path"] for name in tables]
    return {"ready": True, "test": test, "run": receipt["run"], "merge": info,
            "promoted": promoted}


# --- label-free environment check and calibration -------------------------------------------------


def check_env(project: Project, commit: str | None = None) -> dict[str, Any]:
    """Label-free: bundle integrity, prep integrity, addendum parse, P1-6 budget, versions."""
    from importlib import metadata

    from nakkhu.followup import ml
    from nakkhu.followup import prep as P

    out: dict[str, Any] = {"host_type": host_type()}
    man_path = project.paths.root / BUNDLE_MANIFEST
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    out["bundle_commit"] = man.get("git_head")
    out["bundle_problems"] = bundle_problems(project, commit or str(man.get("git_head")))
    _, manifest = P.load_prep(project)
    out["prep_run"] = manifest["run"]
    cfg = S.load_prep_config(project)
    spec = S.load_spec(project, cfg["addendum"])
    ml.check_spec(spec)
    out["p1_6_hgb_configs"] = len(ml.hgb_configs(spec))
    out["p1_6_gam_lambdas"] = len(ml.gam_lambdas(spec))
    import platform

    out["python"] = platform.python_version()
    out["packages"] = {p: metadata.version(p) for p in ("numpy", "scipy", "pandas",
                                                        "scikit-learn", "geopandas", "shapely")}
    out["threads"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "MKL_NUM_THREADS")}
    with contextlib.suppress(OSError):
        out["cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    out["ok"] = not out["bundle_problems"]
    return out


def calibrate(project: Project, *, fits: int = 6, seed: int = 0) -> dict[str, Any]:
    """Label-free timing of P1-6 HGB fits: the real prep features of the Microsoft rows, a
    RANDOM target (Bernoulli, 5 %, ``seed``), every ``60 // fits``-th configuration, on 80 % of
    the rows (an inner-training fold's size). Reads no change map or audit file."""
    import time

    from nakkhu.followup import ml
    from nakkhu.followup import prep as P

    prep, _ = P.load_prep(project)
    cfg = S.load_prep_config(project)
    spec = S.load_spec(project, cfg["addendum"])
    X = ml.matrix(prep["microsoft"])
    X = X[np.all(np.isfinite(X), axis=1)]
    rng = np.random.default_rng(seed)
    n = min(len(X), 6769)
    X = X[rng.choice(len(X), n, replace=False)][: int(0.8 * n)]
    y = (rng.random(len(X)) < 0.05).astype(int)
    cands = ml.hgb_configs(spec)
    step = max(1, len(cands) // fits)
    picked = list(range(0, len(cands), step))[:fits]
    times = []
    for k in picked:
        t0 = time.perf_counter()
        ml.make("M-D", cands[k]).fit(X, y)
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    unit = mean * (len(cands) * ml.INNER_FOLDS + 1)
    return {"host_type": host_type(), "rows": len(X), "configs": picked,
            "fit_s": [round(t, 3) for t in times], "mean_fit_s": round(mean, 3),
            "est_hgb_unit_s": round(unit, 1),
            "est_p1_6_core_h": round(unit * 2 * 26 / 3600, 2)}
