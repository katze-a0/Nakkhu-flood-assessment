import json
from pathlib import Path

import pytest
import yaml

from nakkhu.artifacts import promote
from nakkhu.cli import main
from nakkhu.config import load_project
from nakkhu.paper import (
    NumbersError,
    build_numbers,
    format_value,
    parse_selector,
    run_numbers,
    select,
)
from nakkhu.runs import create_run

SUMMARY = {
    "aoi_km2": {"analysis_extent": 13.344122927629066},
    "tile_sets": [
        {"purpose": "infer", "scene": "wv02", "tiles": 277},
        {"purpose": "train", "scene": "wv02", "tiles": 1483},
    ],
    "history": [{"version": "v1", "sha256": "aa" * 32}, {"version": "v2", "sha256": "bb" * 32}],
}
CSV = "inventory,n_final\ngoogle_v3,14202\nosm,10154\n"


def _setup(root: Path) -> None:
    """A complete run with a JSON and a CSV output, both promoted; one config file."""
    project = load_project(root)
    run = create_run(project, "demo")
    with run.step("write"):
        (run.path / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
        (run.path / "filters.csv").write_text(CSV, encoding="utf-8")
    promote(f"{run.ref}/summary.json", "demo/summary.json", project.paths)
    promote(f"{run.ref}/filters.csv", "demo/filters.csv", project.paths)
    (root / "configs" / "demo.yaml").write_text("radius: {r_m: 8.0}\npixel_m: 0.5\ncap_px: 24\n",
                                                encoding="utf-8")
    (root / "run_ref.txt").write_text(run.ref, encoding="utf-8")


def _table(root: Path, numbers: dict, output: str = "out/numbers.tex") -> Path:
    path = root / "configs" / "paper" / "numbers.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"schema_version": 1, "output": output, "numbers": numbers}),
                    encoding="utf-8")
    return path


ENTRIES = {
    "numExtentKm": {"source": "artifacts/demo/summary.json", "select": "aoi_km2.analysis_extent",
                    "format": "{:.2f}", "status": "final"},
    "numTiles": {"source": "artifacts/demo/summary.json",
                 "select": "tile_sets[purpose=train,scene=wv02].tiles", "format": "{:d}",
                 "status": "final"},
    "numFpGoogle": {"source": "artifacts/demo/filters.csv",
                    "select": "[inventory=google_v3].n_final", "format": "{:,}",
                    "status": "final"},
    "numSha": {"source": "artifacts/demo/summary.json", "select": "history[-1].sha256",
               "format": "{:.8}", "status": "final"},
    "numVersions": {"source": "artifacts/demo/summary.json", "select": "history",
                    "aggregate": "count", "format": "{:d}", "status": "final"},
    "numCapM": {"source": "configs/demo.yaml", "select": "cap_px",
                "multiply_by": {"source": "configs/demo.yaml", "select": "pixel_m"},
                "format": "{:g}", "status": "final"},
    "numRadius": {"source": "configs/demo.yaml", "select": "radius.r_m", "format": "{:.1f}",
                  "status": "val"},
    "numReach": {"source": None, "status": "missing", "note": "no artifact yet"},
}


def test_selectors():
    assert parse_selector("a.b[2]") == [("key", "a"), ("key", "b"), ("index", 2)]
    assert parse_selector("[k=v, j=w].x") == [("match", {"k": "v", "j": "w"}), ("key", "x")]
    assert parse_selector(".") == []
    doc = {"rows": [{"k": "a", "v": 1}, {"k": "b", "v": 2}]}
    assert select(doc, "rows[k=b].v") == 2
    assert select(doc, "rows[-1].k") == "b"
    assert select(doc, ".") is doc
    with pytest.raises(NumbersError, match="matches 0"):
        select(doc, "rows[k=c].v")
    with pytest.raises(NumbersError, match="not found"):
        select(doc, "missing")
    with pytest.raises(NumbersError, match="out of range"):
        select(doc, "rows[5]")


def test_format_value_latex():
    assert format_value(14202, "{:,}") == "14{,}202"
    assert format_value(13.344, "{:.2f}") == "13.34"
    assert format_value("prereg_v5", "{}") == r"prereg\_v5"
    assert format_value("a,b", "{}") == "a,b"  # commas only become {,} in numbers


def test_final_values_and_provenance(root: Path):
    _setup(root)
    _table(root, ENTRIES)
    path, numbers, up_to_date = run_numbers(load_project(root))
    assert not up_to_date
    text = path.read_text(encoding="utf-8")
    assert path == (root / "out" / "numbers.tex").resolve()
    assert r"\newcommand{\numExtentKm}{13.34}" in text
    assert r"\newcommand{\numTiles}{1483}" in text
    assert r"\newcommand{\numFpGoogle}{14{,}202}" in text
    assert r"\newcommand{\numSha}{bbbbbbbb}" in text
    assert r"\newcommand{\numVersions}{2}" in text
    assert r"\newcommand{\numCapM}{12}" in text
    # every macro has a comment with its source, hash prefix, selector, raw value and status
    lines = text.splitlines()
    i = lines.index(r"\newcommand{\numFpGoogle}{14{,}202}")
    assert lines[i - 1].startswith("% numFpGoogle | final | artifacts/demo/filters.csv @")
    assert "[inventory=google_v3].n_final = 14202" in lines[i - 1]
    # val and missing never print a value
    assert r"\newcommand{\numRadius}{\todo{\texttt{numRadius} (val)}}" in text
    assert r"\newcommand{\numReach}{\todo{\texttt{numReach}: no logged artifact}}" in text
    assert {n.name: n.status for n in numbers}["numReach"] == "missing"
    # deterministic: a second run is a no-op
    assert run_numbers(load_project(root), check=True)[2]


def test_draft_marks_non_final(root: Path):
    _setup(root)
    _table(root, ENTRIES)
    path, _, _ = run_numbers(load_project(root), draft=True)
    text = path.read_text(encoding="utf-8")
    assert r"\newcommand{\numRadius}{\draftnum{8.0}{val}}" in text
    assert r"\newcommand{\numExtentKm}{13.34}" in text
    assert r"\todo{\texttt{numReach}: no logged artifact}" in text  # still a todo


def test_check_reports_stale(root: Path, capsys):
    _setup(root)
    _table(root, ENTRIES)
    args = ["--root", str(root), "paper", "numbers"]
    assert main([*args, "--check"]) == 1
    assert "STALE" in capsys.readouterr().out
    assert not (root / "out" / "numbers.tex").exists()
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_artifact_hash_must_match_index(root: Path):
    _setup(root)
    _table(root, ENTRIES)
    (root / "artifacts" / "demo" / "filters.csv").write_text(CSV + "ms,1\n", encoding="utf-8")
    with pytest.raises(NumbersError, match="differs from INDEX"):
        run_numbers(load_project(root))


def test_run_source_needs_complete_journal(root: Path):
    _setup(root)
    project = load_project(root)
    run = create_run(project, "open")
    run.start("write")
    (run.path / "summary.json").write_text('{"x": 1}', encoding="utf-8")
    entry = {"numX": {"source": f"runs/{run.ref}/summary.json", "select": "x",
                      "status": "final"}}
    with pytest.raises(NumbersError, match="running"):
        build_numbers(project.paths, {"numbers": entry}, draft=False)
    run.end("write", 0)
    assert build_numbers(project.paths, {"numbers": entry}, draft=False)[0].tex == "1"


@pytest.mark.parametrize("entry, message", [
    ({"source": "configs/demo.yaml", "select": "cap_px", "status": "done"}, "status must be"),
    ({"source": None, "status": "final"}, "missing"),
    ({"source": "configs/demo.yaml", "select": "cap_px", "status": "missing"}, "missing"),
    ({"source": "configs/demo.yaml", "status": "final"}, "needs `select`"),
    ({"source": "README.md", "select": "x", "status": "final"}, "must lie under"),
    ({"source": "configs/nope.yaml", "select": "x", "status": "final"}, "does not exist"),
])
def test_invalid_entries(root: Path, entry: dict, message: str):
    _setup(root)
    (root / "README.md").write_text("x", encoding="utf-8")
    with pytest.raises(NumbersError, match=message):
        build_numbers(load_project(root).paths, {"numbers": {"numX": entry}}, draft=False)


def test_macro_names_are_letters_only(root: Path):
    _setup(root)
    entry = {"source": "configs/demo.yaml", "select": "cap_px", "status": "final"}
    with pytest.raises(NumbersError, match="letters only"):
        build_numbers(load_project(root).paths, {"numbers": {"num_1": entry}}, draft=False)


def test_repository_table_is_valid():
    """The shipped table resolves against the real artifacts and keeps statuses honest."""
    project = load_project()
    table = yaml.safe_load(
        (project.paths.configs / "paper" / "numbers.yaml").read_text(encoding="utf-8"))
    try:
        numbers = build_numbers(project.paths, table, draft=False)
    except NumbersError as error:
        if "journal is empty" in str(error):  # run journals are local (gitignored)
            pytest.skip(str(error))
        raise
    by_name = {n.name: n for n in numbers}
    for number in numbers:
        if number.status != "final":
            assert number.tex.startswith(r"\todo{"), number.name
    assert by_name["numExtentKm"].tex == "13.34"
    assert by_name["numSplitTrain"].tex == "973"


def test_values_aggregates_field_and_expr(root: Path):
    """`values` concatenates selections; mean/sd/min/max aggregate; `expr` derives a number."""
    _setup(root)
    (root / "configs" / "seeds.yaml").write_text(
        "a: {f1: 0.10}\nb: {f1: 0.12}\nc: {f1: 0.14}\n"
        "per_seed: [{x: 2.0, y: 1.0}, {x: 4.0, y: 1.0}]\n", encoding="utf-8")
    seeds = [{"source": "configs/seeds.yaml", "select": f"{k}.f1"} for k in "abc"]
    entries = {
        "numMean": {"values": seeds, "aggregate": "mean", "format": "{:.3f}", "status": "final"},
        "numSd": {"values": seeds, "aggregate": "sd", "format": "{:.3f}", "status": "final"},
        "numMin": {"values": seeds, "aggregate": "min", "format": "{:.2f}", "status": "final"},
        "numMax": {"values": seeds, "aggregate": "max", "format": "{:.2f}", "status": "final"},
        "numX": {"source": "configs/seeds.yaml", "select": "per_seed", "field": "x",
                 "aggregate": "mean", "format": "{:g}", "status": "final"},
        "numVal": {"source": "configs/demo.yaml", "select": "pixel_m", "status": "val"},
        "numRel": {"expr": "100 * (1 - numMin / numMax)", "format": "{:.0f}",
                   "status": "final"},
    }
    tex = {n.name: n.tex for n in build_numbers(load_project(root).paths,
                                                  {"numbers": entries}, draft=False)}
    assert tex["numMean"] == "0.120"
    assert tex["numSd"] == "0.020"          # sample SD (ddof 1)
    assert (tex["numMin"], tex["numMax"]) == ("0.10", "0.14")
    assert tex["numX"] == "3"
    assert tex["numRel"] == "29"
    bad = {**entries, "numBad": {"expr": "numVal * 2", "status": "final"}}
    with pytest.raises(NumbersError, match="non-final"):
        build_numbers(load_project(root).paths, {"numbers": bad}, draft=False)
    bad = {**entries, "numBad": {"expr": "__import__('os')", "status": "final"}}
    with pytest.raises(NumbersError, match="may only use"):
        build_numbers(load_project(root).paths, {"numbers": bad}, draft=False)
    bad = {"numBad": {"values": seeds[:1], "aggregate": "sd", "status": "final"}}
    with pytest.raises(NumbersError, match="at least two"):
        build_numbers(load_project(root).paths, {"numbers": bad}, draft=False)
