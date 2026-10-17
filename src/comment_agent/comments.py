"""Turning review suggestions into comment lines and placing them in source text."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Tag = Literal["TODO", "FIXME", "PERF", "WARN", "NOTE"]

# Every comment the agent writes carries this marker, so they are easy to find
# (`:g/\[ai\]/d` in Neovim removes them all) and the model can recognise its own.
MARKER = "[ai]"

# Lines this short ("}", "end", ")") are too ambiguous to re-anchor if they move.
MIN_ANCHOR_LENGTH = 4
ANCHOR_SEARCH_WINDOW = 200

_PREFIXES = {
    "#": [".py", ".pyi", ".rb", ".sh", ".bash", ".zsh", ".fish", ".pl", ".r", ".jl", ".ex", ".exs", ".nix", ".tf"],
    "//": [
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".cs",
        ".swift",
        ".dart",
        ".php",
        ".zig",
    ],
    "--": [".lua", ".sql", ".hs", ".elm"],
}
COMMENT_PREFIXES = {ext: prefix for prefix, exts in _PREFIXES.items() for ext in exts}

# Task comments a code action can hand to Claude Code: the agent's own and the developer's.
TASK_COMMENT = re.compile(r"^\s*(?:#|//|--)\s*(TODO|FIXME|FIX|BUG|PERF|WARN|HACK|NOTE)\b\S*:?\s*(.+)$")


class Suggestion(BaseModel):
    """One improvement the model wants flagged, as it returns it."""

    line: int = Field(description="1-based line number of the code line the comment should be placed directly above.")
    tag: Tag
    message: str = Field(description="One short, specific sentence (under ~100 characters).")


class Review(BaseModel):
    """The model's structured output for one file."""

    suggestions: list[Suggestion] = Field(
        description="Improvements worth flagging. Empty if nothing is worth the developer's attention.",
    )


@dataclass(frozen=True)
class Edit:
    """Comment lines to insert above the line that read `expect` at review time."""

    line: int
    expect: str
    comments: list[str]


@dataclass(frozen=True)
class Placement:
    """An edit resolved against the current text: insert `lines` before 1-based line `line`."""

    line: int
    lines: list[str]


def comment_prefix(path: Path) -> str | None:
    """Return the line-comment syntax for `path`'s language, or None if it isn't supported."""
    assert path.name, f"comment_prefix got {path!r}, not a file; pass the document's file path"
    prefix = COMMENT_PREFIXES.get(path.suffix.lower())
    assert prefix is None or prefix in _PREFIXES, f"COMMENT_PREFIXES maps {path.suffix!r} to {prefix!r}; fix _PREFIXES"
    return prefix


def render(prefix: str, suggestion: Suggestion) -> str:
    """Format a suggestion as a single, unindented comment line carrying the agent's marker."""
    assert prefix in _PREFIXES, f"render got prefix {prefix!r}; take it from comment_prefix()"
    message = " ".join(suggestion.message.split())
    assert message, f"render got the empty suggestion {suggestion!r}; plan_edits must filter these out first"
    comment = f"{prefix} {suggestion.tag}: {MARKER} {message}"
    assert parse_task_comment(comment), f"rendered {comment!r}, which TASK_COMMENT can't parse; align the two"
    return comment


def detect_newline(text: str) -> str:
    """Return the line ending `text` uses, so inserted lines match it."""
    newline = "\r\n" if "\r\n" in text else "\n"
    assert newline in text or "\n" not in text, (
        f"picked {newline!r} for text containing {text[:80]!r}; fix the detection"
    )
    assert newline == "\r\n" or "\r\n" not in text, f"picked LF for CRLF text {text[:80]!r}; fix the detection"
    return newline


def plan_edits(text: str, suggestions: list[Suggestion], prefix: str) -> list[Edit]:
    """Group suggestions by target line, dropping invalid lines, empty messages and comments already present."""
    assert prefix in _PREFIXES, f"plan_edits got prefix {prefix!r}; take it from comment_prefix()"
    lines = text.split(detect_newline(text))
    by_line: dict[int, list[str]] = {}
    for suggestion in suggestions:
        if not 1 <= suggestion.line <= len(lines) or not suggestion.message.strip():
            continue
        comment = render(prefix, suggestion)
        existing = by_line.setdefault(suggestion.line, [])
        if comment in existing or _already_above(lines, suggestion.line, comment):
            continue
        existing.append(comment)
    edits = [Edit(line, lines[line - 1], comments) for line, comments in sorted(by_line.items()) if comments]
    assert all(1 <= e.line <= len(lines) and e.comments for e in edits), (
        f"plan_edits produced {edits!r} for a {len(lines)}-line file; its range and empty-group filters are broken"
    )
    return edits


def _already_above(lines: list[str], line: int, comment: str) -> bool:
    assert 1 <= line <= len(lines), f"_already_above got line {line} of {len(lines)}; plan_edits must range-check first"
    assert comment.strip() == comment, f"_already_above got {comment!r}; pass the unindented output of render()"
    prefix = comment.split(" ", 1)[0]
    i = line - 2
    while i >= 0 and lines[i].lstrip().startswith(prefix):
        if lines[i].strip() == comment:
            return True
        i -= 1
    return False


def find_anchor(lines: list[str], line: int, expect: str) -> int | None:
    """Locate the line an edit targets, even if edits made during the review shifted it."""
    assert line >= 1, f"find_anchor got line {line}; Edit.line is 1-based, so convert before calling"
    assert "\n" not in expect, f"find_anchor got {expect!r}; Edit.expect must be one line without its newline"
    anchor = _search_anchor(lines, line, expect)
    assert anchor is None or lines[anchor - 1] == expect, (
        f"_search_anchor returned line {anchor}, which doesn't hold {expect!r}; fix its index arithmetic"
    )
    return anchor


def _search_anchor(lines: list[str], line: int, expect: str) -> int | None:
    assert line >= 1, f"_search_anchor got line {line}; call it through find_anchor"
    assert ANCHOR_SEARCH_WINDOW > 0, f"ANCHOR_SEARCH_WINDOW is {ANCHOR_SEARCH_WINDOW}; set it to a positive line count"
    if 1 <= line <= len(lines) and lines[line - 1] == expect:
        return line
    if len(expect.strip()) < MIN_ANCHOR_LENGTH:
        return None
    for distance in range(1, ANCHOR_SEARCH_WINDOW + 1):
        for candidate in (line - distance, line + distance):
            if 1 <= candidate <= len(lines) and lines[candidate - 1] == expect:
                return candidate
    return None


def place_edits(text: str, edits: list[Edit]) -> list[Placement]:
    """Resolve edits against `text`, indenting comments to match their anchor line. Sorted bottom-up."""
    assert all(e.comments for e in edits), f"place_edits got {edits!r} with an empty edit; build edits with plan_edits"
    lines = text.split(detect_newline(text))
    placements: list[Placement] = []
    for edit in edits:
        if (anchor := find_anchor(lines, edit.line, edit.expect)) is None:
            continue
        target = lines[anchor - 1]
        indent = target[: len(target) - len(target.lstrip())]
        placements.append(Placement(anchor, [indent + comment for comment in edit.comments]))
    placements.sort(key=lambda p: p.line, reverse=True)
    assert len(placements) <= len(edits), (
        f"place_edits made {len(placements)} placements from {len(edits)} edits; each edit may place at most once"
    )
    return placements


def apply_placements(text: str, placements: list[Placement]) -> str:
    """Insert placed comment lines into `text`."""
    newline = detect_newline(text)
    lines = text.split(newline)
    assert all(1 <= p.line <= len(lines) for p in placements), (
        f"apply_placements got {placements!r} for {len(lines)} lines; resolve them against this text with place_edits"
    )
    for placement in sorted(placements, key=lambda p: p.line, reverse=True):
        lines[placement.line - 1 : placement.line - 1] = placement.lines
    result = newline.join(lines)
    added = sum(len(p.lines) for p in placements)
    assert result.count(newline) == text.count(newline) + added, (
        f"apply_placements should add {added} lines but went {text.count(newline)} -> {result.count(newline)}; "
        "fix the slice insertion"
    )
    return result


def parse_task_comment(line: str) -> tuple[str, str] | None:
    """(tag, message) if `line` is a TODO-style comment, else None."""
    assert "\n" not in line, f"parse_task_comment got {line!r}; strip the line ending before calling"
    if match := TASK_COMMENT.match(line):
        tag, message = match.group(1), match.group(2).removeprefix(MARKER).strip()
        assert tag, f"TASK_COMMENT matched {line!r} without a tag; check its first group"
        assert message, f"TASK_COMMENT matched {line!r} with an empty message; require text after the tag"
        return tag, message
    return None
