"""Pre-registration guard and test access journal (plan 005 Steps 2 and 5).

- ``configs/prereg.yaml`` is frozen by ``nakkhu prereg freeze``: its SHA-256 is written to a run
  and promoted to ``artifacts/prereg/<version>.sha256`` (``version`` field of the prereg, e.g.
  ``prereg_v2``). Earlier versions' hash files stay as history; ``prereg/history.json`` lists
  every freeze and ``prereg/changelog.md`` (``--changelog``) says what changed and why.
- :func:`require_frozen` (called before every test scoring) refuses unless the live prereg file
  has exactly the hash frozen for its version, the split file has the hash recorded in the
  prereg, and every checked section still holds:
  ``mirror_of: <file>#<dotted.key>`` + ``values`` (a mapping matches key by key, recursively;
  the live config may hold more keys), ``sha256_of: <file>`` + ``sha256`` (file bytes), and
  ``constant_of: <module>:<NAME>`` + ``values`` (a Python constant, tuples read as lists;
  ``NAME`` may be dotted, e.g. ``f.__defaults__``).
- Addenda (:mod:`nakkhu.evaluate.addendum`, plan 014) declare follow-up tests after test access
  without changing this file; :func:`verify` also checks every frozen addendum.
- :func:`record_test_access` appends (run, eval config hash, prereg hash, split hash, UTC) to
  ``runs/eval/test_journal.jsonl`` *before* scoring and flags a run scored again with a different
  config. Once the journal has an entry no new freeze is accepted, except one audited
  amendment (:func:`freeze` with ``amend=True``, plan 013): the previous frozen file is given
  and hash-checked, every changed dotted path must lie under an explicit ``allow`` prefix and
  outside :data:`AMEND_LOCKED` (the sections the journalled test accesses used), no pending
  analysis (:func:`pending_analyses`) may be in the journal, and a changelog is appended.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nakkhu.config import Project, ResolvedConfig, load_yaml
from nakkhu.runs import atomic_write_bytes, utc_now

PREREG = "configs/prereg.yaml"
PREREG_DIR = "prereg"                         # under artifacts/
DEFAULT_VERSION = "prereg_v1"
PREREG_SHA_ARTIFACT = f"{PREREG_DIR}/{DEFAULT_VERSION}.sha256"
HISTORY = f"{PREREG_DIR}/history.json"
CHANGELOG = f"{PREREG_DIR}/changelog.md"
TEST_JOURNAL = "eval/test_journal.jsonl"      # under runs/
SPLIT_FILE = "configs/splits/split_v1.yaml"


class GuardError(RuntimeError):
    """Test scoring would break the pre-registration."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicate_keys(text: str) -> list[str]:
    """Every mapping key that appears twice in the same mapping, at any level, as
    ``"<dotted path> (lines a, b)"``. YAML keeps only the last value of a duplicate key, so a
    duplicate silently drops a section (prereg v5/v6 lost ``uncertainty.block_bootstrap``)."""
    import yaml

    out: list[str] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, yaml.MappingNode):
            seen: dict[Any, int] = {}
            for key_node, value_node in node.value:
                key = key_node.value if isinstance(key_node, yaml.ScalarNode) else str(key_node)
                line = key_node.start_mark.line + 1
                path = f"{where}.{key}" if where else str(key)
                if key in seen and key != "<<":
                    out.append(f"{path} (lines {seen[key]}, {line})")
                else:
                    seen[key] = line
                walk(value_node, path)
        elif isinstance(node, yaml.SequenceNode):
            for i, child in enumerate(node.value):
                walk(child, f"{where}[{i}]")

    for doc in yaml.compose_all(text, Loader=yaml.SafeLoader):
        walk(doc, "")
    return out


def load_prereg(project: Project) -> dict[str, Any]:
    """The prereg mapping; refuses a file with a duplicate key at any level."""
    path = project.paths.resolve(PREREG)
    if not path.is_file():
        raise GuardError(f"{PREREG} does not exist")
    dups = duplicate_keys(path.read_text(encoding="utf-8"))
    if dups:
        raise GuardError(f"{PREREG} has duplicate keys (YAML keeps only the last): "
                         + "; ".join(dups))
    return load_yaml(path)


def sha_artifact(version: str) -> str:
    return f"{PREREG_DIR}/{version}.sha256"


def _version(project: Project) -> str:
    try:
        return str(load_prereg(project).get("version") or DEFAULT_VERSION)
    except (GuardError, ValueError):
        return DEFAULT_VERSION


def frozen_sha(project: Project, version: str | None = None) -> str | None:
    path = project.paths.artifacts / sha_artifact(version or _version(project))
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").split()[0]


def frozen_versions(project: Project) -> dict[str, str]:
    """Every frozen version under ``artifacts/prereg/`` -> its hash."""
    folder = project.paths.artifacts / PREREG_DIR
    if not folder.is_dir():
        return {}
    return {p.stem: p.read_text(encoding="utf-8").split()[0]
            for p in sorted(folder.glob("*.sha256"))}


def _lookup(values: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(values, dict) or part not in values:
            raise KeyError(dotted)
        values = values[part]
    return values


def _checks(node: Any, where: str = "") -> list[tuple[str, dict[str, Any]]]:
    out = []
    if isinstance(node, dict):
        if any(k in node for k in ("mirror_of", "sha256_of", "constant_of")):
            out.append((where, node))
        for key, child in node.items():
            out.extend(_checks(child, f"{where}.{key}" if where else str(key)))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            out.extend(_checks(child, f"{where}[{i}]"))
    return out


def _plain(value: Any) -> Any:
    """JSON round trip: tuples become lists, keys strings (for constants)."""
    return json.loads(json.dumps(value))


def _diff(expected: Any, live: Any, path: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(live, dict):
        out = []
        for k, v in expected.items():
            if k not in live:
                out.append(f"{path}.{k} missing, prereg {v!r}")
            else:
                out.extend(_diff(v, live[k], f"{path}.{k}"))
        return out
    return [] if live == expected else [f"{path} is {live!r}, prereg {expected!r}"]


def mirror_problems(project: Project, prereg: dict[str, Any]) -> list[str]:
    """Differences between the prereg's checked sections and the live files."""
    problems = []
    for where, node in _checks(prereg):
        if "sha256_of" in node:
            path = project.paths.resolve(str(node["sha256_of"]))
            if not path.is_file():
                problems.append(f"{where}: {node['sha256_of']} missing")
            elif sha256_file(path) != node.get("sha256"):
                problems.append(f"{where}: {node['sha256_of']} sha256 {sha256_file(path)[:12]} "
                                f"!= prereg {str(node.get('sha256'))[:12]}")
            continue
        if "constant_of" in node:
            module, _, name = str(node["constant_of"]).partition(":")
            try:
                live = importlib.import_module(module)
                for part in name.split("."):    # dotted: e.g. f.__defaults__
                    live = getattr(live, part)
                live = _plain(live)
            except (ImportError, AttributeError) as error:
                problems.append(f"{where}: cannot read {node['constant_of']} ({error})")
                continue
            problems.extend(f"{where}: {p}" for p in _diff(node.get("values"), live,
                                                           str(node["constant_of"])))
            continue
        target = str(node["mirror_of"])
        file, _, key = target.partition("#")
        try:
            live = load_yaml(project.paths.resolve(file))
            live = _lookup(live, key) if key else live
        except (KeyError, OSError, ValueError) as error:
            problems.append(f"{where}: cannot read {target} ({error})")
            continue
        problems.extend(f"{where}: {p}" for p in _diff(node.get("values"), live, target))
    return problems


def verify(project: Project) -> dict[str, Any]:
    """State of the freeze; ``problems`` is empty when test scoring is allowed."""
    problems: list[str] = []
    out: dict[str, Any] = {"prereg_sha256": None, "frozen_sha256": frozen_sha(project),
                           "split_sha256": None, "problems": problems}
    try:
        prereg = load_prereg(project)
    except GuardError as error:
        problems.append(str(error))
        return out
    out["prereg_sha256"] = sha256_file(project.paths.resolve(PREREG))
    if out["frozen_sha256"] is None:
        problems.append(f"prereg not frozen (artifacts/{sha_artifact(_version(project))} "
                        "missing)")
    elif out["frozen_sha256"] != out["prereg_sha256"]:
        problems.append(f"{PREREG} changed after the freeze (sha256 {out['prereg_sha256'][:12]} "
                        f"!= frozen {out['frozen_sha256'][:12]})")
    split = prereg.get("split") or {}
    split_path = project.paths.resolve(split.get("file", SPLIT_FILE))
    if not split_path.is_file():
        problems.append(f"split file {split_path} missing")
    else:
        out["split_sha256"] = sha256_file(split_path)
        if split.get("sha256") != out["split_sha256"]:
            problems.append(f"split file sha256 {out['split_sha256'][:12]} != prereg "
                            f"{str(split.get('sha256'))[:12]}")
    problems.extend(mirror_problems(project, prereg))
    from nakkhu.evaluate.addendum import addendum_problems, frozen_addenda, superseded

    out["addenda"] = frozen_addenda(project)
    out["addenda_superseded"] = superseded(project)
    problems.extend(addendum_problems(project))
    return out


def require_frozen(project: Project) -> dict[str, Any]:
    state = verify(project)
    if state["problems"]:
        raise GuardError("test scoring refused: " + "; ".join(state["problems"]))
    return state


def check_access(project: Project, split: str) -> dict[str, Any]:
    """Guard of ``nakkhu eval --split``: test needs the frozen prereg; val only records it."""
    if split == "test":
        return require_frozen(project)
    if split != "val":
        raise GuardError(f"unknown split {split!r}; scoring is val or test only")
    return verify(project)


# --- test access journal --------------------------------------------------------------------


def journal_path(project: Project) -> Path:
    return project.paths.runs / TEST_JOURNAL


JOURNAL_LOCK_TIMEOUT_S = 120.0
_held: dict[str, int] = {}                     # lock path -> re-entry depth in this process
_held_guard = threading.RLock()


def _os_lock(handle: Any) -> bool:
    """Non-blocking exclusive OS lock on byte 0 of ``handle``; False if another holds it."""
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _os_unlock(handle: Any) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def journal_lock(project: Project, timeout_s: float | None = None,
                 poll_s: float = 0.05) -> Iterator[Path]:
    """Exclusive cross-process lock of the test journal (``test_journal.jsonl.lock``).

    An OS byte-range lock (``msvcrt`` on Windows, ``flock`` elsewhere) on a persistent lock
    file: it is released by the OS if the holder dies, so a crash cannot leave it stuck. It is
    re-entrant within one process (``record_addendum_access`` holds it across its run-once
    check and the append in :func:`record_test_access`). Raises :class:`GuardError` after
    ``timeout_s`` (default :data:`JOURNAL_LOCK_TIMEOUT_S`) without the lock."""
    path = journal_path(project).with_name(Path(TEST_JOURNAL).name + ".lock")
    key = str(path.resolve())
    with _held_guard:
        if _held.get(key):
            _held[key] += 1
            reentered = True
        else:
            reentered = False
    if reentered:
        try:
            yield path
        finally:
            with _held_guard:
                _held[key] -= 1
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    wait = JOURNAL_LOCK_TIMEOUT_S if timeout_s is None else float(timeout_s)
    deadline = time.monotonic() + wait
    handle = open(path, "a+b")                 # noqa: SIM115 (held for the whole block)
    try:
        while not _os_lock(handle):
            if time.monotonic() >= deadline:
                raise GuardError(f"test journal lock {path} not acquired within {wait:g} s "
                                 "(another writer holds it)")
            time.sleep(poll_s)
        with _held_guard:
            _held[key] = 1
        try:
            yield path
        finally:
            with _held_guard:
                _held.pop(key, None)
            _os_unlock(handle)
    finally:
        handle.close()


def read_test_journal(project: Project) -> list[dict[str, Any]]:
    path = journal_path(project)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _replace_retrying(path: Path, data: bytes, tries: int = 100, wait_s: float = 0.05) -> None:
    """:func:`atomic_write_bytes`, retried while Windows refuses the rename because a reader
    (outside the lock) has the journal open."""
    for attempt in range(tries):
        try:
            atomic_write_bytes(path, data)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait_s)


def record_test_access(project: Project, run_ref: str, eval_cfghash: str, prereg_sha: str,
                       split_sha: str, note: str = "", *,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append one test access; ``flagged`` when the run was scored before with another config.
    ``extra`` adds fields (e.g. ``addendum_sha256``) without replacing the standard ones.
    The read and the append run under :func:`journal_lock` (concurrent writers)."""
    with journal_lock(project):
        prior = [e for e in read_test_journal(project) if e.get("run") == run_ref]
        entry = {**(extra or {}), "run": run_ref, "eval_cfghash": eval_cfghash,
                 "prereg_sha256": prereg_sha, "split_sha256": split_sha, "utc": utc_now(),
                 "note": note}
        path = journal_path(project)
        existing = path.read_bytes() if path.is_file() else b""
        if existing and not existing.endswith(b"\n"):
            existing += b"\n"
        line = json.dumps(entry, sort_keys=True) + "\n"
        _replace_retrying(path, existing + line.encode("utf-8"))
    different = [e for e in prior if e.get("eval_cfghash") != eval_cfghash]
    return {"entry": entry, "prior_accesses": len(prior),
            "flagged": bool(different),
            "flag_reason": (f"run scored on test {len(different)} time(s) before with a different "
                            "eval config") if different else ""}


# --- freezing -------------------------------------------------------------------------------


def _history(project: Project) -> list[dict[str, Any]]:
    path = project.paths.artifacts / HISTORY
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


# --- audited amendment after test access (plan 013) -----------------------------------------

# Sections an amendment may never touch: everything the journalled test accesses (detector
# scoring, C1/C5, C4, the recall estimate) and the shared settings used. A dotted path is
# locked if it equals or lies under one of these, unless it equals or lies under an
# AMEND_UNLOCKED child (fields that describe uses or record an outcome, not a rule).
AMEND_LOCKED = (
    "schema_version", "split", "seeds", "references", "matching", "conf_threshold",
    "detection_postprocess", "metrics", "uncertainty", "labels", "greenhouses", "hand",
    "audit_recall", "detector_selection", "detector_final", "audit_precision_pool",
    "cross_sensor", "detector_comparisons", "falsifiers.C1", "falsifiers.C2", "falsifiers.C4",
    "falsifiers.C5", "falsifiers.G1", "audit.overture_rule", "audit.config.values.schema_version",
    "audit.config.values.seed", "audit.config.values.region", "audit.config.values.registration",
    "audit.config.values.tasks.recall_points", "audit.config.values.tasks.recall_nearmiss",
    "audit.config.values.tasks.precision_pool",
)
AMEND_UNLOCKED = ("detector_final.uses", "falsifiers.C5.outcome")
# Journal entries of analyses that an amendment must precede (run ref, or prefix with ":").
PENDING_RUNS = ("audit/precision_pool", "audit/precision_microsoft", "audit/flood_trace",
                "audit/damage", "uncertainty/run")
PENDING_PREFIXES = ("flood/validate",)
PENDING_NOTES = ("plan 004 G1",)          # `labels g1 --splits test` (nakkhu.labelqa)
PREVIOUS_COPY = f"{PREREG_DIR}/{{version}}.yaml"    # artifacts/prereg/<previous>.yaml


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".")


def changed_paths(old: Any, new: Any, where: str = "") -> list[str]:
    """Dotted paths whose parsed-YAML value differs (added, removed or changed keys; a list or
    scalar compares whole)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            path = f"{where}.{key}" if where else str(key)
            if key not in old or key not in new:
                out.append(path)
            else:
                out.extend(changed_paths(old[key], new[key], path))
        return out
    return [] if old == new else [where or "<root>"]


def locked(path: str) -> bool:
    return any(_under(path, p) for p in AMEND_LOCKED) and \
        not any(_under(path, u) for u in AMEND_UNLOCKED)


def pending_analyses(entries: list[dict[str, Any]]) -> list[str]:
    """Journal entries of analyses that must not precede an amendment."""
    out = []
    for e in entries:
        run, note = str(e.get("run") or ""), str(e.get("note") or "")
        if run in PENDING_RUNS or any(run.startswith(p) for p in PENDING_PREFIXES) \
                or any(n in note for n in PENDING_NOTES):
            out.append(run)
    return out


def _version_number(version: str) -> int:
    tail = version.rsplit("_v", 1)[-1]
    if not tail.isdigit():
        raise GuardError(f"version {version!r} is not of the form prereg_v<N>")
    return int(tail)


def amendment(project: Project, live: dict[str, Any], version: str, *,
              previous_file: str | Path | None, allow: list[str] | tuple[str, ...],
              changelog: str | Path | None) -> dict[str, Any]:
    """Check an amendment after test access (plan 013); returns what the history records.
    Raises :class:`GuardError` on the first violated condition."""
    import yaml

    if changelog is None or not Path(changelog).is_file() \
            or not Path(changelog).read_text(encoding="utf-8").strip():
        raise GuardError("an amendment needs --changelog (a non-empty markdown file)")
    if previous_file is None or not Path(previous_file).is_file():
        raise GuardError("an amendment needs --previous-file (a copy of the frozen previous "
                         "configs/prereg.yaml)")
    if not allow:
        raise GuardError("an amendment needs at least one --allow prefix")
    existing = frozen_versions(project)
    if not existing:
        raise GuardError("nothing is frozen; freeze normally instead of amending")
    previous = max(existing, key=_version_number)
    prev_sha = sha256_file(previous_file)
    if prev_sha != existing[previous]:
        raise GuardError(f"--previous-file sha256 {prev_sha[:12]} != frozen {previous} "
                         f"{existing[previous][:12]}")
    if _version_number(version) <= _version_number(previous):
        raise GuardError(f"version {version} must be above the frozen {previous}")
    text = Path(changelog).read_text(encoding="utf-8")
    if version not in text or existing[previous] not in text:
        raise GuardError(f"the changelog must name {version} and the {previous} hash "
                         f"{existing[previous]}")
    prev_text = Path(previous_file).read_text(encoding="utf-8")
    if duplicate_keys(prev_text):
        raise GuardError("--previous-file has duplicate keys")
    old = yaml.safe_load(prev_text)
    changed = changed_paths(old, live)
    if not changed:
        raise GuardError("the live prereg does not differ from the previous version")
    bad_lock = [p for p in changed if locked(p)]
    if bad_lock:
        raise GuardError("amendment changes locked section(s): " + ", ".join(bad_lock))
    outside = [p for p in changed if not any(_under(p, a) for a in allow)]
    if outside:
        raise GuardError("amendment changes path(s) outside --allow: " + ", ".join(outside))
    journal = read_test_journal(project)
    pending = pending_analyses(journal)
    if pending:
        raise GuardError("the journal already holds pending analyses (" + ", ".join(pending)
                         + "); an amendment must precede them")
    return {"amends": previous, "previous_sha256": existing[previous],
            "test_journal_entries": len(journal),
            "last_journal_utc": journal[-1].get("utc") if journal else None,
            "changed_paths": changed, "allow": list(allow)}


def freeze(project: Project, *, replace: bool = False,
           changelog: str | Path | None = None, amend: bool = False,
           previous_file: str | Path | None = None,
           allow: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """Hash ``configs/prereg.yaml`` into a run; promote ``prereg/<version>.sha256``,
    ``prereg/history.json`` and (if given) ``prereg/changelog.md``.

    Refuses when the split or a checked section disagrees with the prereg, when any test access
    is journalled and a freeze already exists, and (without ``replace``) when another hash is
    already frozen for this or another version. With ``amend`` the journalled case is allowed
    once the checks of :func:`amendment` pass; the changelog is then appended to the promoted
    one and the previous file is promoted as ``prereg/<previous>.yaml``.
    """
    prereg = load_prereg(project)
    version = str(prereg.get("version") or DEFAULT_VERSION)
    state = verify(project)
    blocking = [p for p in state["problems"] if "not frozen" not in p
                and "changed after the freeze" not in p]
    if blocking:
        raise GuardError("cannot freeze: " + "; ".join(blocking))
    sha = state["prereg_sha256"]
    existing = frozen_versions(project)
    if existing.get(version) == sha:
        return {"prereg_sha256": sha, "version": version, "status": "already frozen"}
    if amend:
        if version in existing:
            raise GuardError(f"{version} is already frozen with another hash; an amendment "
                             "needs a new version")
        record = amendment(project, prereg, version, previous_file=previous_file, allow=allow,
                           changelog=changelog)
        return _write_freeze(project, version, sha, state, existing, changelog, record,
                             previous_file)
    if existing:
        accesses = read_test_journal(project)
        if accesses:
            raise GuardError(f"a prereg is frozen ({', '.join(existing)}) and test was accessed "
                             f"{len(accesses)} time(s); no new freeze is allowed")
        if not replace:
            raise GuardError(f"prereg already frozen ({', '.join(existing)}); pass --replace to "
                             "re-freeze (allowed only while the test journal is empty)")
    return _write_freeze(project, version, sha, state, existing, changelog, None, None)


def _write_freeze(project: Project, version: str, sha: str, state: dict[str, Any],
                  existing: dict[str, str], changelog: str | Path | None,
                  amend: dict[str, Any] | None, previous_file: str | Path | None
                  ) -> dict[str, Any]:
    from nakkhu.artifacts import promote
    from nakkhu.runs import atomic_write_json, atomic_write_text, create_run

    run = create_run(project, "prereg", ResolvedConfig(
        values={"command": "prereg-freeze", "prereg_sha256": sha, "version": version,
                "amend": bool(amend)},
        source=PREREG))
    history: list[dict[str, Any]] = _history(project)
    if not history:
        history = [{"version": v, "sha256": h, "run": None, "utc": None, "replaced": None}
                   for v, h in existing.items() if v != version]
    if amend:
        replaced = amend["amends"]
    else:
        replaced = existing.get(version) or (sorted(existing)[-1] if existing else None)
    entry: dict[str, Any] = {"version": version, "sha256": sha, "run": run.ref, "utc": utc_now(),
                             "replaced": replaced,
                             "test_journal_entries": len(read_test_journal(project))}
    if amend:
        entry.update(amend)
    history.append(entry)
    files = [(f"{version}.sha256", sha_artifact(version)), ("history.json", HISTORY)]
    with run.step("freeze"):
        atomic_write_text(run.path / f"{version}.sha256", f"{sha}  {PREREG}\n")
        atomic_write_json(run.path / "history.json", history)
        if changelog is not None:
            text = Path(changelog).read_text(encoding="utf-8")
            if amend:          # appended: earlier entries can never be rewritten by an amendment
                current = project.paths.artifacts / CHANGELOG
                old = current.read_text(encoding="utf-8") if current.is_file() else ""
                text = (old.rstrip("\n") + "\n\n" + text) if old else text
            atomic_write_text(run.path / "changelog.md", text)
            files.append(("changelog.md", CHANGELOG))
        if amend and previous_file is not None:
            name = f"{amend['amends']}.yaml"
            atomic_write_bytes(run.path / name, Path(previous_file).read_bytes())
            files.append((name, PREVIOUS_COPY.format(version=amend["amends"])))
        run.write_summary({**entry, "split_sha256": state["split_sha256"]})
    for name, dest in files:
        promote(run.path / name, dest, project.paths, replace=True)
    return {"prereg_sha256": sha, "version": version, "status": "frozen", "run": run.ref,
            "replaced": entry["replaced"], "kept": sorted(v for v in existing if v != version),
            **({"amends": amend["amends"], "changed_paths": amend["changed_paths"]}
               if amend else {})}
