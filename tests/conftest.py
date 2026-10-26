from pathlib import Path

import pytest

from nakkhu.config import Project, load_project

PROJECT_YAML = """\
schema_version: 1
crs: EPSG:32645
paths:
  data_root: data
  manifest: configs/data/manifest.yaml
seeds:
  arms: [0, 1, 2]
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "nakkhu"
    (root / "configs" / "data").mkdir(parents=True)
    (root / "configs" / "project.yaml").write_text(PROJECT_YAML, encoding="utf-8")
    (root / "configs" / "data" / "manifest.yaml").write_text(
        "schema_version: 1\nentries: []\n", encoding="utf-8"
    )
    (root / "artifacts").mkdir()
    return root


@pytest.fixture
def project(root: Path) -> Project:
    return load_project(root)
