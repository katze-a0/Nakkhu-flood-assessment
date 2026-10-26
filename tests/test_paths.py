from pathlib import Path

import pytest

from nakkhu.paths import PathError, ProjectPaths


def test_relative_resolves_under_root(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path)
    assert paths.resolve("configs/project.yaml") == (tmp_path / "configs/project.yaml").resolve()
    assert paths.data_root == (tmp_path / "data").resolve()


@pytest.mark.parametrize("value", ["../outside.txt", "configs/../../outside.txt", ""])
def test_rejects_escape_and_empty(tmp_path: Path, value: str):
    paths = ProjectPaths.create(tmp_path / "root")
    with pytest.raises(PathError):
        paths.resolve(value)


def test_rejects_absolute_outside_root(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path / "root")
    with pytest.raises(PathError):
        paths.resolve(tmp_path / "elsewhere" / "file.tif")


def test_rejects_colab_content_path(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path / "root")
    with pytest.raises(PathError):
        paths.resolve("/content/drive/MyDrive/mosaic.tif")


def test_absolute_inside_root_allowed(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path)
    target = tmp_path / "runs" / "x.json"
    assert paths.resolve(target) == target.resolve()


def test_external_data_root_allowed(tmp_path: Path):
    external = tmp_path / "drive" / "nakkhu-data"
    paths = ProjectPaths.create(tmp_path / "root", external)
    target = external / "raw" / "dem.tif"
    assert paths.resolve(target) == target.resolve()
    assert paths.resolve("{data_root}/raw/dem.tif") == target.resolve()
    assert paths.relative(target) == "{data_root}/raw/dem.tif"
    with pytest.raises(PathError):
        paths.resolve("{data_root}/../escape.tif")


def test_relative_form_under_root(tmp_path: Path):
    paths = ProjectPaths.create(tmp_path)
    assert paths.relative(tmp_path / "configs" / "a.yaml") == "configs/a.yaml"
