import hashlib
from pathlib import Path

import pytest
import yaml

from nakkhu.config import Project
from nakkhu.manifest import (
    REQUIRED_FIELDS,
    ManifestError,
    sha256_tree,
    validate_entry,
    verify_manifest,
)


def _entry(path: Path, local_path: str) -> dict:
    data = path.read_bytes()
    return {
        "id": "sample",
        "title": "Sample file",
        "url": "https://example.org/sample.bin",
        "access": "public",
        "licence": "CC0-1.0",
        "attribution": "Example",
        "citation": "example2026",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "retrieved_utc": "2026-10-05T00:00:00Z",
        "local_path": local_path,
        "clip": "extent",
        "notes": "",
    }


def _write_manifest(project: Project, entries: list[dict]) -> Path:
    path = project.paths.root / "configs" / "data" / "manifest.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "entries": entries}), encoding="utf-8")
    return path


@pytest.fixture
def sample(project: Project) -> Path:
    path = project.paths.data_root / "raw" / "sample" / "sample.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(bytes(range(256)) * 16)
    return path


def test_verify_passes_then_fails_on_corruption(project: Project, sample: Path):
    manifest = _write_manifest(project, [_entry(sample, "{data_root}/raw/sample/sample.bin")])
    assert all(r.ok for r in verify_manifest(manifest, project.paths))

    data = bytearray(sample.read_bytes())
    data[100] ^= 0xFF  # flip one byte, same size
    sample.write_bytes(bytes(data))
    [result] = verify_manifest(manifest, project.paths)
    assert not result.ok
    assert any("sha256 mismatch" in problem for problem in result.problems)


def test_missing_field_fails(project: Project, sample: Path):
    entry = _entry(sample, "data/raw/sample/sample.bin")
    del entry["licence"]
    [result] = verify_manifest(_write_manifest(project, [entry]), project.paths)
    assert "missing field 'licence'" in result.problems


def test_missing_file_fails(project: Project, sample: Path):
    entry = _entry(sample, "data/raw/sample/absent.bin")
    [result] = verify_manifest(_write_manifest(project, [entry]), project.paths)
    assert any("does not exist" in problem for problem in result.problems)


def test_path_outside_roots_fails(project: Project, sample: Path):
    entry = _entry(sample, "../../outside.bin")
    [result] = verify_manifest(_write_manifest(project, [entry]), project.paths)
    assert not result.ok


def test_schema_enums_and_types(sample: Path):
    entry = _entry(sample, "x") | {"access": "open", "clip": "basin", "bytes": -1, "sha256": "X"}
    problems = validate_entry(entry)
    assert len(problems) == 4
    assert validate_entry(_entry(sample, "x")) == []
    assert len(REQUIRED_FIELDS) == 13


def test_duplicate_ids_fail(project: Project, sample: Path):
    entry = _entry(sample, "data/raw/sample/sample.bin")
    results = verify_manifest(_write_manifest(project, [entry, dict(entry)]), project.paths)
    assert [r.ok for r in results] == [True, False]


def test_empty_manifest_passes(project: Project):
    path = project.paths.root / "configs" / "data" / "manifest.yaml"
    assert verify_manifest(path, project.paths) == []


def test_malformed_manifest(project: Project):
    path = project.paths.root / "bad.yaml"
    path.write_text("entries: 3\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        verify_manifest(path, project.paths)


def test_directory_entry_uses_tree_digest(project: Project, sample: Path):
    (sample.parent / "b.txt").write_text("b", encoding="utf-8")
    size, digest = sha256_tree(sample.parent)
    entry = _entry(sample, "data/raw/sample") | {"bytes": size, "sha256": digest}
    [result] = verify_manifest(_write_manifest(project, [entry]), project.paths)
    assert result.ok
    (sample.parent / "b.txt").write_text("c", encoding="utf-8")
    [result] = verify_manifest(_write_manifest(project, [entry]), project.paths)
    assert not result.ok
