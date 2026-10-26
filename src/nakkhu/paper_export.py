"""``nakkhu paper export``: a standalone submission copy of the manuscript.

The working manuscript takes every number from the ``\\num...`` macros that ``nakkhu paper
numbers`` writes into ``numbers.tex``, whose comments carry artifact paths and hashes. The
export reads ``Main.tex`` and ``numbers.tex`` and writes ``<out>/Main.tex`` with

* LaTeX comments removed: a comment-only line is dropped, a trailing comment is cut back to a
  bare ``%`` (which still swallows the end of line, so the typeset text does not change);
* every ``\\num...`` macro replaced by its defined value, copied verbatim, with TeX's spacing
  rules kept: a space (or one end of line) after the control word is swallowed as TeX would,
  ``\\numX{}`` becomes the value, and ``~``, ``\\,``, ``}``, punctuation and the rest stay;
* the ``\\input{numbers}`` line removed.

It then copies what the build needs: the document class's directory (``Definitions/``,
unchanged), the ``\\bibliography`` files (``%`` lines between entries dropped) and every
``\\includegraphics`` file (PDF Subject/Keywords blanked: floodfig records its source artifacts
there), and checks that no internal path or hash marker is left in them. It fails on a ``\\num...`` macro
that ``numbers.tex`` does not define, on a used value that is still a ``\\todo``/``\\draftnum``,
on any other ``\\input``/``\\include`` and on verbatim material the comment stripper cannot read.
The working files are never written.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nakkhu.config import Project
from nakkhu.paper import NUMBERS_CONFIG, NumbersError, load_table, output_path

EXPORT_DIR = "submission"
# TeX/LaTeX commands that start with "num" but are not generated numbers.
TEX_NUM_COMMANDS = frozenset({"number", "numexpr", "numberwithin", "numberline"})
GRAPHIC_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
PDF_INFO_KEYS = (b"/Subject", b"/Keywords")
# Path and provenance fragments that must not reach the journal (checked in Main.tex, the
# bibliography and the figures; the template's Definitions/ are copied as published).
INTERNAL_MARKERS = ("artifacts/", "runs/", "configs/", "plans/", ".tmp/", "data/raw/",
                    "FINDINGS", "sha256", "prereg_")
_NEWCOMMAND = re.compile(r"\\newcommand\{\\(?P<name>[A-Za-z]+)\}\{(?P<value>.*)\}")
_PROVIDE = re.compile(r"\\providecommand\{\\(todo|draftnum)\}")
_CONTROL = re.compile(r"\\(?P<word>[A-Za-z]+)|\\.", re.S)
_ENDS_IN_WORD = re.compile(r"\\[A-Za-z]+$")
_VERBATIM = re.compile(r"\\verb\b|\\begin\{(verbatim|lstlisting|minted|comment)\*?\}")
_URL_PERCENT = re.compile(r"\\(url|href)\{[^}]*(?<!\\)%")
_INPUT = re.compile(r"\\(input|include)\{(?P<name>[^}]*)\}")
_GRAPHICS = re.compile(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}")
_BIBLIOGRAPHY = re.compile(r"\\bibliography\{(?P<names>[^}]+)\}")
_CLASS = re.compile(r"\\documentclass\s*(\[[^\]]*\])?\s*\{(?P<path>[^}]+)\}")


class ExportError(RuntimeError):
    """The manuscript cannot be exported faithfully."""


@dataclass
class ExportResult:
    main: Path
    macros_used: int
    replacements: int
    files: list[Path] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)


# --- numbers.tex ----------------------------------------------------------------------------


def load_macros(text: str) -> dict[str, str]:
    """Macro name -> value from a generated numbers file (one ``\\newcommand`` per line)."""
    macros: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or _PROVIDE.match(stripped):
            continue
        match = _NEWCOMMAND.fullmatch(stripped)
        if match is None:
            raise ExportError(f"numbers file line {number}: not a one-line \\newcommand: {line}")
        name = match.group("name")
        if name in macros:
            raise ExportError(f"numbers file defines \\{name} twice")
        macros[name] = match.group("value")
    return macros


# --- comments -------------------------------------------------------------------------------


def _comment_start(line: str) -> int | None:
    """Index of the first unescaped ``%`` in ``line``; ``\\%`` is text, ``\\\\%`` a comment."""
    i = 0
    while i < len(line):
        if line[i] == "\\":
            i += 2  # a control symbol, or the first letter of a control word (never a %)
            continue
        if line[i] == "%":
            return i
        i += 1
    return None


def strip_comments(tex: str) -> str:
    """Drop comment-only lines; cut trailing comments back to a bare ``%``."""
    if (match := _VERBATIM.search(tex)) or (match := _URL_PERCENT.search(tex)):
        raise ExportError(f"cannot strip comments safely around {match.group(0)!r}")
    out: list[str] = []
    for line in tex.splitlines(keepends=True):
        start = _comment_start(line)
        if start is None:
            out.append(line)
        elif line[:start].strip():
            out.append(line[:start] + "%" + ("\n" if line.endswith("\n") else ""))
    return "".join(out)


# --- macros ---------------------------------------------------------------------------------


def expand_numbers(tex: str, macros: dict[str, str]) -> tuple[str, set[str], int]:
    """Replace each ``\\num...`` macro by its value. Returns (text, names used, replacements)."""
    out: list[str] = []
    used: set[str] = set()
    undefined: set[str] = set()
    pos = 0
    replacements = 0
    previous_word_end = -1
    for match in _CONTROL.finditer(tex):
        if match.start() < pos:
            continue  # inside text this function already consumed (a swallowed `{}`)
        name = match.group("word")
        if name is None or not name.startswith("num") or name in TEX_NUM_COMMANDS:
            if name is not None:
                previous_word_end = match.end()
            continue
        if name not in macros:
            undefined.add(name)
            continue
        value = macros[name]
        if previous_word_end == match.start() and re.match(r"[A-Za-z]", value):
            raise ExportError(f"\\{name} follows a control word and its value {value!r} starts "
                              "with a letter; add {} or a space before it in Main.tex")
        out.append(tex[pos:match.start()])
        end = match.end()
        while end < len(tex) and tex[end] in " \t":
            end += 1  # TeX skips spaces after a control word
        if tex.startswith("{}", end):
            end += 2
        elif end < len(tex) and tex[end] in "\r\n":
            value += "%"  # the end of line after a control word yields no space either
        if _ENDS_IN_WORD.search(value) and end < len(tex) and tex[end].isalpha():
            value += "{}"
        out.append(value)
        used.add(name)
        replacements += 1
        pos = end
    out.append(tex[pos:])
    if undefined:
        raise ExportError("undefined number macros: " + ", ".join(f"\\{n}" for n in
                                                                  sorted(undefined)))
    drafts = sorted(n for n in used if "\\todo" in macros[n] or "\\draftnum" in macros[n])
    if drafts:
        raise ExportError("number macros still without a final value: " + ", ".join(
            f"\\{n} = {macros[n]}" for n in drafts))
    return "".join(out), used, replacements


def remove_numbers_input(tex: str, stem: str) -> str:
    """Remove the ``\\input{numbers}`` line; any other ``\\input``/``\\include`` is an error."""
    lines = tex.splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    for line in lines:
        inputs = list(_INPUT.finditer(line))
        names = {Path(m.group("name").strip()).with_suffix("").as_posix() for m in inputs}
        if inputs and names == {stem} and not _INPUT.sub("", line).strip():
            removed += 1
            continue
        if inputs:
            raise ExportError(f"cannot inline {line.strip()!r}: only \\input{{{stem}}} is handled")
        kept.append(line)
    if removed != 1:
        raise ExportError(f"expected one \\input{{{stem}}} line in the manuscript, found {removed}")
    return "".join(kept)


# --- support files --------------------------------------------------------------------------


def _relative(base: Path, name: str, what: str) -> Path:
    path = (base / name).resolve()
    if base.resolve() not in path.parents:
        raise ExportError(f"{what} {name!r} lies outside {base}")
    return path


def support_files(tex: str, base: Path) -> tuple[list[Path], list[Path]]:
    """(directories, files) the build needs, relative to the manuscript directory ``base``."""
    dirs: list[Path] = []
    files: list[Path] = []
    for match in _CLASS.finditer(tex):
        parent = Path(match.group("path").strip()).parent
        if parent != Path("."):
            dirs.append(_relative(base, parent.as_posix(), "class directory"))
    for match in _BIBLIOGRAPHY.finditer(tex):
        for name in match.group("names").split(","):
            name = name.strip()
            files.append(_relative(base, name if name.endswith(".bib") else f"{name}.bib",
                                   "bibliography"))
    if "\\graphicspath" in tex:
        raise ExportError("\\graphicspath is not supported by the export")
    for match in _GRAPHICS.finditer(tex):
        name = match.group("path").strip()
        candidates = [name] if Path(name).suffix else [name + s for s in GRAPHIC_SUFFIXES]
        found = [p for p in (_relative(base, c, "figure") for c in candidates) if p.is_file()]
        if not found:
            raise ExportError(f"figure {name!r} not found under {base}")
        files.append(found[0])
    missing = [p for p in [*dirs, *files] if not p.exists()]
    if missing:
        raise ExportError("missing build inputs: " + ", ".join(str(p) for p in missing))
    return dirs, list(dict.fromkeys(files))


def _prepare_out(out: Path, main: Path) -> None:
    """Empty ``out`` for a fresh export; refuse a directory that is not a previous export."""
    out = out.resolve()
    if out == main.parent.resolve() or out in main.resolve().parents:
        raise ExportError(f"output {out} would overwrite the working manuscript")
    if out.exists():
        if not out.is_dir():
            raise ExportError(f"output {out} is not a directory")
        if any(out.iterdir()) and not (out / main.name).is_file():
            raise ExportError(f"output {out} is not empty and holds no previous export; "
                              "remove it or choose another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def strip_bib_comments(bib: str) -> str:
    """Drop ``%`` lines between entries: BibTeX ignores them, and they carry curation notes."""
    out: list[str] = []
    depth = 0
    for number, line in enumerate(bib.splitlines(keepends=True), 1):
        if depth == 0 and line.lstrip().startswith("%"):
            continue
        depth += line.count("{") - line.count("}")  # BibTeX matches braces the same way
        if depth < 0:
            raise ExportError(f"bibliography line {number}: unbalanced closing brace")
        out.append(line)
    return "".join(out)


def _pdf_string_end(data: bytes, start: int) -> int:
    """Index just past the PDF literal string that opens at ``data[start]`` (``(``)."""
    depth = 0
    i = start
    while i < len(data):
        if data[i] == 0x5C:  # backslash escape
            i += 2
            continue
        if data[i] == 0x28:
            depth += 1
        elif data[i] == 0x29:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ExportError("unterminated string in a PDF document-information entry")


def scrub_pdf_info(data: bytes) -> bytes:
    """Blank the Subject/Keywords document information (floodfig writes source paths there).

    Each string keeps its length, so the cross-reference offsets stay valid.
    """
    out = bytearray(data)
    for key in PDF_INFO_KEYS:
        for match in re.finditer(re.escape(key) + rb"\s*\(", data):
            start = match.end() - 1
            end = _pdf_string_end(data, start)
            out[start + 1:end - 1] = b" " * (end - start - 2)
    return bytes(out)


def internal_markers(data: bytes) -> list[str]:
    """Internal path or provenance markers found in a file's bytes."""
    return [m for m in INTERNAL_MARKERS if m.encode() in data]


# --- export ---------------------------------------------------------------------------------


def export(main: Path, numbers: Path, out: Path) -> ExportResult:
    """Write the submission copy of ``main`` into ``out``. The working files are only read."""
    for path in (main, numbers):
        if not path.is_file():
            raise ExportError(f"{path} does not exist")
    base = main.parent.resolve()
    macros = load_macros(numbers.read_text(encoding="utf-8"))
    tex = strip_comments(main.read_text(encoding="utf-8"))
    tex = remove_numbers_input(tex, numbers.stem)
    tex, used, replacements = expand_numbers(tex, macros)
    leftover = sorted({m.group("word") for m in _CONTROL.finditer(tex)
                       if (m.group("word") or "") in macros})
    if leftover:
        raise ExportError("number macros left after expansion: " + ", ".join(leftover))
    dirs, files = support_files(tex, base)
    _prepare_out(out, main)
    target = out / main.name
    target.write_text(tex, encoding="utf-8", newline="\n")
    checked = [target]
    copied = [target]
    for directory in dirs:  # the journal's own template files, copied unchanged
        shutil.copytree(directory, out / directory.relative_to(base))
        copied.append(out / directory.relative_to(base))
    for path in files:
        destination = out / path.relative_to(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".bib":
            destination.write_text(strip_bib_comments(path.read_text(encoding="utf-8")),
                                   encoding="utf-8", newline="\n")
        elif path.suffix == ".pdf":
            destination.write_bytes(scrub_pdf_info(path.read_bytes()))
        else:
            shutil.copy2(path, destination)
        checked.append(destination)
        copied.append(destination)
    leaks = {p.relative_to(out).as_posix(): found for p in checked
             if (found := internal_markers(p.read_bytes()))}
    if leaks:
        raise ExportError("internal markers left in the export: " + "; ".join(
            f"{name}: {', '.join(found)}" for name, found in leaks.items()))
    todos = re.findall(r"\\todo\{[^{}]*\}", tex)
    return ExportResult(main=target, macros_used=len(used), replacements=replacements,
                        files=copied, todos=todos)


# --- CLI ------------------------------------------------------------------------------------


def _export_cmd(project: Project, args: argparse.Namespace) -> int:
    try:
        if args.numbers:
            numbers = Path(args.numbers).resolve()
        else:
            table = load_table(project.paths.resolve(args.table))
            numbers = output_path(project.paths, table, None)
        main = Path(args.main).resolve() if args.main else numbers.parent / "Main.tex"
        out = Path(args.out).resolve() if args.out else main.parent / EXPORT_DIR
        result = export(main, numbers, out)
    except (ExportError, NumbersError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {result.main}: {result.replacements} number macros inlined "
          f"({result.macros_used} distinct), comments stripped")
    print(f"copied {len(result.files) - 1} support paths into {result.main.parent}")
    for todo in result.todos:
        print(f"  warning: left in the manuscript text: {todo}")
    return 0


def register(sub: Any) -> None:
    """Add ``export`` to the ``nakkhu paper`` subcommands."""
    parser = sub.add_parser("export", help="write a standalone submission copy of the manuscript "
                            f"(numbers inlined, comments stripped) into <manuscript>/{EXPORT_DIR}/")
    parser.add_argument("--table", default=NUMBERS_CONFIG,
                        help=f"numbers table naming numbers.tex (default: {NUMBERS_CONFIG})")
    parser.add_argument("--numbers", help="numbers .tex (default: the table's `output`)")
    parser.add_argument("--main", help="manuscript .tex (default: Main.tex beside numbers.tex)")
    parser.add_argument("--out", help=f"output directory (default: {EXPORT_DIR}/ beside Main.tex;"
                        " emptied first, and only if it holds a previous export)")
    parser.set_defaults(func=_export_cmd)
