from pathlib import Path

import pytest

from nakkhu.cli import build_parser, main
from nakkhu.config import load_project
from nakkhu.runs import create_run


def test_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("data", "runs", "artifacts"):
        assert command in out


def test_data_verify_empty_manifest(root: Path, capsys):
    assert main(["--root", str(root), "data", "verify"]) == 0
    assert "0 entries, 0 failed" in capsys.readouterr().out


def test_promote_and_list_round_trip(root: Path, capsys):
    run = create_run(load_project(root), "evaluate")
    with run.step("score"):
        (run.path / "out.json").write_text("{}\n", encoding="utf-8")
    assert main(["--root", str(root), "runs", "list"]) == 0
    assert f"{run.ref}\tcomplete" in capsys.readouterr().out
    assert main(["--root", str(root), "artifacts", "promote", f"{run.ref}/out.json", "o.json"]) == 0
    assert main(["--root", str(root), "artifacts", "list"]) == 0
    assert "1 artifacts" in capsys.readouterr().out


def test_errors_exit_2(root: Path, capsys):
    assert main(["--root", str(root), "--set", "paths.typo=1", "runs", "list"]) == 2
    assert "unknown config path" in capsys.readouterr().err
