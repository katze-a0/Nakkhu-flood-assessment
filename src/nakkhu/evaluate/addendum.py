"""Pre-registration addenda: follow-up analyses declared after the primary analyses (plan 014).

The guard (:mod:`nakkhu.evaluate.guard`) freezes ``configs/prereg.yaml`` and allows one audited
amendment only *before* the pending analyses run. An addendum is the other case: new tests
declared *after* some test access, without changing the frozen prereg.

- An addendum is one YAML file ``configs/prereg_addenda/<id>.yaml`` whose ``addendum`` field is
  ``<id>`` (e.g. ``prereg_v9_a1``). It names the frozen prereg it extends (``base_version`` and
  ``base_sha256``), other addenda it builds on (``requires``) and the declared ``tests`` (each
  with ``access: test`` when it reads test labels or test-region audit answers, else
  ``access: none``). It may hold ``mirror_of`` / ``sha256_of`` / ``constant_of`` checks.
- :func:`freeze_addendum` hashes it into a run and promotes ``prereg/addenda/<id>.sha256``, a
  frozen copy ``<id>.yaml``, optionally the plan text ``<id>.plan.md`` and ``index.json``; the
  changelog, if given, is appended to ``prereg/changelog.md``. It needs a clean
  :func:`guard.verify` (the base prereg is untouched) and is allowed whatever the journal holds.
  A frozen addendum is immutable: a changed file needs a new id. A new addendum may name an
  older one in ``supersedes`` (with ``supersede_reason``) only while no journal entry of the
  older one exists; the older one then stays frozen and integrity-checked, its pins no longer
  bind and its tests are refused.
- :func:`addendum_problems` (part of ``nakkhu prereg verify``) checks every frozen addendum
  against its live file, its frozen copies, its base hash, its ``requires`` and its checks.
- :func:`record_addendum_access` journals one access of a declared ``access: test`` test with
  the note ``<id>:<test>``. Each test runs once; one retry (note ``<id>:<test> retry``) is
  allowed only when explicitly requested, for a run that failed before writing any metric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from nakkhu.config import Project, ResolvedConfig
from nakkhu.evaluate import guard
from nakkhu.runs import atomic_write_bytes, utc_now

ADDENDA_DIR = "configs/prereg_addenda"
ARTIFACT_DIR = f"{guard.PREREG_DIR}/addenda"          # under artifacts/
INDEX = f"{ARTIFACT_DIR}/index.json"
ID_PATTERN = re.compile(r"^prereg_v\d+_a\d+$")
ACCESS_KINDS = ("test", "none")


class AddendumError(guard.GuardError):
    """An addendum cannot be frozen, or an addendum test may not run."""


def addendum_path(project: Project, addendum_id: str) -> Path:
    return project.paths.resolve(f"{ADDENDA_DIR}/{addendum_id}.yaml")


def _artifact(project: Project, name: str) -> Path:
    return project.paths.artifacts / ARTIFACT_DIR / name


def read_index(project: Project) -> list[dict[str, Any]]:
    path = project.paths.artifacts / INDEX
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def frozen_addenda(project: Project) -> dict[str, str]:
    """Every frozen addendum id -> its hash (from ``prereg/addenda/*.sha256``)."""
    folder = project.paths.artifacts / ARTIFACT_DIR
    if not folder.is_dir():
        return {}
    return {p.stem: p.read_text(encoding="utf-8").split()[0]
            for p in sorted(folder.glob("*.sha256"))}


def load_addendum(path: Path) -> dict[str, Any]:
    """Parse an addendum file; refuses duplicate keys and a missing or malformed header."""
    import yaml

    if not path.is_file():
        raise AddendumError(f"{path} does not exist")
    text = path.read_text(encoding="utf-8")
    dups = guard.duplicate_keys(text)
    if dups:
        raise AddendumError(f"{path.name} has duplicate keys (YAML keeps only the last): "
                            + "; ".join(dups))
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise AddendumError(f"{path.name}: top level must be a mapping")
    addendum_id = str(doc.get("addendum") or "")
    if not ID_PATTERN.match(addendum_id):
        raise AddendumError(f"{path.name}: `addendum` must look like prereg_v<N>_a<M>, "
                            f"got {addendum_id!r}")
    if addendum_id != path.stem:
        raise AddendumError(f"{path.name}: `addendum` is {addendum_id!r}, not the file stem")
    for key in ("base_version", "base_sha256"):
        if not doc.get(key):
            raise AddendumError(f"{path.name}: `{key}` is required")
    if not addendum_id.startswith(str(doc["base_version"]) + "_a"):
        raise AddendumError(f"{path.name}: id {addendum_id} does not extend "
                            f"{doc['base_version']}")
    tests = doc.get("tests")
    if not isinstance(tests, dict) or not tests:
        raise AddendumError(f"{path.name}: `tests` must be a non-empty mapping")
    for name, spec in tests.items():
        if not isinstance(spec, dict) or spec.get("access") not in ACCESS_KINDS:
            raise AddendumError(f"{path.name}: test {name} needs `access: test` or "
                                "`access: none`")
    requires = doc.get("requires") or []
    if not isinstance(requires, list):
        raise AddendumError(f"{path.name}: `requires` must be a list of addendum ids")
    if doc.get("supersedes") is not None:
        if not ID_PATTERN.match(str(doc["supersedes"])) or doc["supersedes"] == addendum_id:
            raise AddendumError(f"{path.name}: `supersedes` must name another addendum id")
        if not str(doc.get("supersede_reason") or "").strip():
            raise AddendumError(f"{path.name}: `supersedes` needs a `supersede_reason`")
    return doc


def superseded(project: Project) -> dict[str, str]:
    """Superseded addendum id -> the frozen addendum that supersedes it (from the index)."""
    return {str(r["supersedes"]): str(r["addendum"]) for r in read_index(project)
            if r.get("supersedes")}


def _accesses_of(project: Project, addendum_id: str) -> list[dict[str, Any]]:
    """Journal entries of any test of ``addendum_id`` (explore runs included)."""
    return [e for e in guard.read_test_journal(project)
            if str(e.get("note") or "").startswith(f"{addendum_id}:")]


def _content_problems(project: Project, doc: dict[str, Any], frozen: dict[str, str]) -> list[str]:
    """Base hash, ``requires`` and checked sections of one parsed addendum."""
    problems = []
    addendum_id = doc["addendum"]
    base = guard.frozen_versions(project).get(str(doc["base_version"]))
    if base is None:
        problems.append(f"{addendum_id}: base {doc['base_version']} has no frozen hash")
    elif base != doc["base_sha256"]:
        problems.append(f"{addendum_id}: base_sha256 {str(doc['base_sha256'])[:12]} != frozen "
                        f"{doc['base_version']} {base[:12]}")
    for other in doc.get("requires") or []:
        if other not in frozen:
            problems.append(f"{addendum_id}: requires {other}, which has no frozen hash")
    problems.extend(f"{addendum_id}: {p}" for p in guard.mirror_problems(project, doc))
    return problems


def addendum_problems(project: Project, superseding: tuple[str, ...] = ()) -> list[str]:
    """Problems of every frozen addendum (empty when all hold). A superseded addendum (or one
    in ``superseding``, about to be superseded) is still checked for integrity (live file,
    frozen copies, record) but no longer for its base, ``requires`` and pins: it can never be
    run, so its pins no longer bind."""
    problems: list[str] = []
    frozen = frozen_addenda(project)
    records = {r.get("addendum"): r for r in read_index(project)}
    retired = set(superseded(project)) | set(superseding)
    for addendum_id, sha in frozen.items():
        live = addendum_path(project, addendum_id)
        if not live.is_file():
            problems.append(f"{addendum_id}: {ADDENDA_DIR}/{addendum_id}.yaml missing")
            continue
        live_sha = guard.sha256_file(live)
        if live_sha != sha:
            problems.append(f"{addendum_id} edited after its freeze (sha256 {live_sha[:12]} != "
                            f"frozen {sha[:12]})")
        copy = _artifact(project, f"{addendum_id}.yaml")
        if not copy.is_file() or guard.sha256_file(copy) != sha:
            problems.append(f"{addendum_id}: frozen copy artifacts/{ARTIFACT_DIR}/"
                            f"{addendum_id}.yaml missing or altered")
        record = records.get(addendum_id)
        if record is None or record.get("sha256") != sha:
            problems.append(f"{addendum_id}: no matching record in artifacts/{INDEX}")
        elif record.get("plan_sha256"):
            plan = _artifact(project, f"{addendum_id}.plan.md")
            if not plan.is_file() or guard.sha256_file(plan) != record["plan_sha256"]:
                problems.append(f"{addendum_id}: frozen plan copy artifacts/{ARTIFACT_DIR}/"
                                f"{addendum_id}.plan.md missing or altered")
        try:
            doc = load_addendum(live)
        except AddendumError as error:
            problems.append(str(error))
            continue
        if addendum_id not in retired:
            problems.extend(_content_problems(project, doc, frozen))
    return problems


def freeze_addendum(project: Project, file: str | Path, *, plan: str | Path | None = None,
                    changelog: str | Path | None = None) -> dict[str, Any]:
    """Freeze one addendum (see the module docstring). Refuses unless the base prereg verifies
    clean, the file names the live frozen prereg as its base, every ``requires`` is frozen and
    its checked sections hold; a frozen id with another hash is refused (immutable)."""
    path = project.paths.resolve(str(file))
    if path.parent != project.paths.resolve(ADDENDA_DIR):
        raise AddendumError(f"an addendum lives in {ADDENDA_DIR}/, not {path.parent}")
    doc = load_addendum(path)
    addendum_id = doc["addendum"]
    sha = guard.sha256_file(path)
    frozen = frozen_addenda(project)
    if addendum_id in frozen and frozen[addendum_id] != sha:
        raise AddendumError(f"{addendum_id} is already frozen with another hash; a frozen "
                            "addendum is immutable (declare a new id)")
    target = doc.get("supersedes")
    state = guard.verify(project)
    blocking = state["problems"]
    if target is not None and frozen.get(addendum_id) != sha:
        if target not in frozen:
            raise AddendumError(f"supersedes {target}, which has no frozen hash")
        if target in superseded(project):
            raise AddendumError(f"{target} is already superseded by "
                                f"{superseded(project)[target]}")
        used = _accesses_of(project, target)
        if used:
            raise AddendumError(f"{target} has {len(used)} journalled access(es); an addendum "
                                "that has been used cannot be superseded")
        mine = set(addendum_problems(project))       # recomputed with the target retired
        blocking = [p for p in blocking if p not in mine] + \
            addendum_problems(project, superseding=(str(target),))
    if blocking:
        raise AddendumError("cannot freeze an addendum: " + "; ".join(blocking))
    if frozen.get(addendum_id) == sha:
        return {"addendum": addendum_id, "sha256": sha, "status": "already frozen"}
    live_version = str(guard.load_prereg(project).get("version") or guard.DEFAULT_VERSION)
    if doc["base_version"] != live_version:
        raise AddendumError(f"base_version {doc['base_version']} is not the live prereg "
                            f"{live_version}")
    problems = _content_problems(project, doc, frozen)
    if problems:
        raise AddendumError("cannot freeze: " + "; ".join(problems))
    if plan is not None and not Path(plan).is_file():
        raise AddendumError(f"--plan {plan} is not a file")
    if changelog is not None and (not Path(changelog).is_file()
                                  or not Path(changelog).read_text(encoding="utf-8").strip()):
        raise AddendumError("--changelog must be a non-empty markdown file")
    if changelog is not None and addendum_id not in Path(changelog).read_text(encoding="utf-8"):
        raise AddendumError(f"the changelog must name {addendum_id}")
    return _write(project, doc, path, sha, state, plan, changelog)


def _write(project: Project, doc: dict[str, Any], path: Path, sha: str, state: dict[str, Any],
           plan: str | Path | None, changelog: str | Path | None) -> dict[str, Any]:
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

    addendum_id = doc["addendum"]
    journal = guard.read_test_journal(project)
    run = create_run(project, "prereg", ResolvedConfig(
        values={"command": "prereg-addendum", "addendum": addendum_id, "sha256": sha},
        source=f"{ADDENDA_DIR}/{addendum_id}.yaml"))
    record: dict[str, Any] = {
        "addendum": addendum_id, "sha256": sha, "base_version": doc["base_version"],
        "base_sha256": doc["base_sha256"], "requires": list(doc.get("requires") or []),
        "supersedes": doc.get("supersedes"), "supersede_reason": doc.get("supersede_reason"),
        "tests": {k: v["access"] for k, v in doc["tests"].items()},
        "plan_sha256": guard.sha256_file(plan) if plan is not None else None,
        "test_journal_entries": len(journal),
        "last_journal_utc": journal[-1].get("utc") if journal else None,
        "split_sha256": state["split_sha256"], "run": run.ref, "utc": utc_now()}
    index = read_index(project) + [record]
    files = [(f"{addendum_id}.sha256", f"{ARTIFACT_DIR}/{addendum_id}.sha256"),
             (f"{addendum_id}.yaml", f"{ARTIFACT_DIR}/{addendum_id}.yaml"),
             ("index.json", INDEX)]
    with run.step("freeze-addendum"):
        atomic_write_text(run.path / f"{addendum_id}.sha256",
                          f"{sha}  {ADDENDA_DIR}/{addendum_id}.yaml\n")
        atomic_write_bytes(run.path / f"{addendum_id}.yaml", path.read_bytes())
        atomic_write_json(run.path / "index.json", index)
        if plan is not None:
            atomic_write_bytes(run.path / f"{addendum_id}.plan.md", Path(plan).read_bytes())
            files.append((f"{addendum_id}.plan.md", f"{ARTIFACT_DIR}/{addendum_id}.plan.md"))
        if changelog is not None:      # appended: earlier entries are never rewritten
            current = project.paths.artifacts / guard.CHANGELOG
            old = current.read_text(encoding="utf-8") if current.is_file() else ""
            text = Path(changelog).read_text(encoding="utf-8")
            atomic_write_text(run.path / "changelog.md",
                              (old.rstrip("\n") + "\n\n" + text) if old else text)
            files.append(("changelog.md", guard.CHANGELOG))
        run.write_summary(record)
    for name, dest in files:
        promote(run.path / name, dest, project.paths, replace=True)
    return {**record, "status": "frozen"}


# --- access of a declared addendum test -----------------------------------------------------


def _note(addendum_id: str, test: str) -> str:
    return f"{addendum_id}:{test}"


def prior_accesses(project: Project, addendum_id: str, test: str) -> list[dict[str, Any]]:
    """Journal entries of ``test`` of ``addendum_id`` (retries included)."""
    note = _note(addendum_id, test)
    return [e for e in guard.read_test_journal(project)
            if str(e.get("note") or "") in (note, f"{note} retry")]


def require_addendum(project: Project, addendum_id: str, test: str, *,
                     retry: bool = False) -> dict[str, Any]:
    """Refuse unless the prereg and every addendum verify clean, ``addendum_id`` is frozen and
    declares ``test``, and (for ``access: test``) the test has not run yet (or, with ``retry``,
    ran exactly once with no retry)."""
    state = guard.verify(project)               # includes every addendum's problems
    if state["problems"]:
        raise AddendumError("addendum test refused: " + "; ".join(state["problems"]))
    frozen = frozen_addenda(project)
    if addendum_id not in frozen:
        raise AddendumError(f"addendum {addendum_id} has no frozen hash")
    if addendum_id in superseded(project):
        raise AddendumError(f"{addendum_id} is superseded by {superseded(project)[addendum_id]}; "
                            "its tests can no longer run")
    doc = load_addendum(addendum_path(project, addendum_id))
    spec = doc["tests"].get(test)
    if spec is None:
        raise AddendumError(f"{addendum_id} declares no test {test!r}")
    prior = prior_accesses(project, addendum_id, test) if spec["access"] == "test" else []
    if prior and not retry:
        raise AddendumError(f"{addendum_id}:{test} already ran ({len(prior)} journal "
                            "entry); each declared test runs once")
    if retry and (len(prior) != 1 or str(prior[0].get("note", "")).endswith(" retry")):
        raise AddendumError(f"{addendum_id}:{test}: a retry needs exactly one earlier, failed "
                            "access and is allowed once")
    return {**state, "addendum": addendum_id, "addendum_sha256": frozen[addendum_id],
            "test": test, "access": spec["access"], "prior_accesses": len(prior)}


def record_addendum_access(project: Project, addendum_id: str, test: str, run_ref: str,
                           eval_cfghash: str, *, retry: bool = False) -> dict[str, Any]:
    """Check with :func:`require_addendum`, then journal the access before any scoring. The
    run-once check and the append hold :func:`guard.journal_lock` together, so two concurrent
    runners of the same test cannot both pass the check."""
    with guard.journal_lock(project):
        state = require_addendum(project, addendum_id, test, retry=retry)
        if state["access"] != "test":
            raise AddendumError(f"{addendum_id}:{test} is declared access: none; it reads no "
                                "test label and is not journalled")
        note = _note(addendum_id, test) + (" retry" if retry else "")
        return guard.record_test_access(project, run_ref, eval_cfghash, state["prereg_sha256"],
                                        state["split_sha256"], note=note,
                                        extra={"addendum_sha256": state["addendum_sha256"]})
