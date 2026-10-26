from pathlib import Path
from typing import Any

import pytest

from nakkhu.config import ConfigError, apply_overrides, config_hash, load_config, load_project

BASE: dict[str, Any] = {
    "crs": "EPSG:32645",
    "paths": {"data_root": "data"},
    "seeds": {"arms": [0, 1, 2]},
}


def test_override_nested_value_parsed_as_yaml():
    values = apply_overrides(BASE, ["seeds.arms=[3, 4]", "paths.data_root=E:/data"])
    assert values["seeds"]["arms"] == [3, 4]
    assert values["paths"]["data_root"] == "E:/data"
    assert BASE["seeds"]["arms"] == [0, 1, 2]  # input is not mutated


def test_override_scalar_types():
    values = apply_overrides({"a": {"n": 1, "flag": True}}, ["a.n=0.5", "a.flag=false"])
    assert values["a"] == {"n": 0.5, "flag": False}


@pytest.mark.parametrize(
    "override", ["seeds.typo=1", "nope.arms=1", "crs.inner=1", "noequals", "=1"]
)
def test_override_rejects_unknown_or_malformed(override):
    with pytest.raises(ConfigError):
        apply_overrides(BASE, [override])


def test_hash_stable_under_key_reordering(tmp_path: Path):
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("x: 1\ny:\n  b: 2\n  a: [1, 2]\n", encoding="utf-8")
    second.write_text("y:\n  a: [1, 2]\n  b: 2\nx: 1\n", encoding="utf-8")
    a, b = load_config(first), load_config(second)
    assert a.cfghash == b.cfghash
    assert len(a.cfghash) == 8
    assert a.sha256.startswith(a.cfghash)


def test_hash_changes_with_values_and_overrides():
    assert config_hash(BASE) != config_hash(apply_overrides(BASE, ["seeds.arms=[0]"]))
    assert config_hash({"a": [1, 2]}) != config_hash({"a": [2, 1]})


def test_hash_handles_yaml_dates(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("date: 2024-01-13\n", encoding="utf-8")
    assert len(load_config(path).cfghash) == 8


def test_load_project_uses_data_root_override(root: Path, tmp_path: Path):
    external = tmp_path / "external"
    project = load_project(root, overrides=[f"paths.data_root={external.as_posix()}"])
    assert project.paths.data_root == external.resolve()
    assert project.config.source == "configs/project.yaml"


def test_missing_config(root: Path):
    with pytest.raises(ConfigError):
        load_project(root, config="configs/missing.yaml")
