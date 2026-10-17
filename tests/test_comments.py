"""Tests for placing review comments in source text."""

from pathlib import Path

import pytest

from comment_agent.comments import (
    Suggestion,
    apply_placements,
    comment_prefix,
    parse_task_comment,
    place_edits,
    plan_edits,
)
from comment_agent.review import History

SOURCE = "def f(xs):\n    for x in xs:\n        print(sorted(xs))\n    return 1\n"


def test_comment_is_indented_to_match_target_line() -> None:
    """A comment sits at its target's indentation, so it stays inside the same block in Python."""
    edits = plan_edits(SOURCE, [Suggestion(line=3, tag="PERF", message="Sort once  outside\nthe loop")], "#")
    result = apply_placements(SOURCE, place_edits(SOURCE, edits)).splitlines()
    assert result[2] == "        # PERF: [ai] Sort once outside the loop", (
        f"comment line is {result[2]!r}; check render() whitespace folding and place_edits() indentation"
    )
    assert result[3] == "        print(sorted(xs))", (
        f"target line became {result[3]!r}; apply_placements() must insert above it, not replace it"
    )


def test_out_of_range_and_duplicate_suggestions_are_dropped() -> None:
    """Hallucinated line numbers and comments already in the file never produce edits."""
    already = "def f():\n    # FIXME: [ai] Handle None\n    return x\n"
    suggestions = [
        Suggestion(line=99, tag="TODO", message="nope"),
        Suggestion(line=3, tag="FIXME", message="Handle None"),
    ]
    edits = plan_edits(already, suggestions, "#")
    assert edits == [], f"got {edits!r}; check plan_edits() range filter and _already_above()"


def test_edits_follow_their_line_when_code_shifts_during_review() -> None:
    """Lines added above the target while the model runs don't misplace the comment."""
    edits = plan_edits(SOURCE, [Suggestion(line=3, tag="PERF", message="Hoist sort")], "#")
    edited = "import os\n\n" + SOURCE
    result = apply_placements(edited, place_edits(edited, edits)).splitlines()
    assert result[4] == "        # PERF: [ai] Hoist sort", (
        f"line 5 is {result[4]!r} in {result!r}; find_anchor() should have followed the target down two lines"
    )
    assert result[5] == "        print(sorted(xs))", f"target moved to {result!r}; check find_anchor()"


def test_short_lines_that_moved_are_not_reanchored() -> None:
    """A moved `}` is ambiguous, so the edit is dropped rather than guessed."""
    text = "a = 1\n}\n"
    edits = plan_edits(text, [Suggestion(line=2, tag="NOTE", message="x")], "//")
    placements = place_edits("x\n" + text, edits)
    assert placements == [], f"got {placements!r}; find_anchor() must not search for lines under MIN_ANCHOR_LENGTH"


def test_crlf_is_preserved() -> None:
    """Windows line endings survive, so the edit doesn't rewrite every line of the file."""
    text = "a = 1\r\nb = 2\r\n"
    edits = plan_edits(text, [Suggestion(line=2, tag="TODO", message="name b")], "#")
    result = apply_placements(text, place_edits(text, edits))
    assert result == "a = 1\r\n# TODO: [ai] name b\r\nb = 2\r\n", f"got {result!r}; check detect_newline()"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("    # PERF: [ai] Hoist sort", ("PERF", "Hoist sort")),
        ("// TODO(ben): handle errors", ("TODO", "handle errors")),
        ("-- FIXME check bounds", ("FIXME", "check bounds")),
        ("x = 1  # TODO later", None),
        ("# just a comment", None),
    ],
)
def test_parse_task_comment(line: str, expected: tuple[str, str] | None) -> None:
    """Whole-line task comments in each syntax get a code action; trailing and plain comments don't."""
    parsed = parse_task_comment(line)
    assert parsed == expected, f"parse_task_comment({line!r}) gave {parsed!r}; check the TASK_COMMENT pattern"


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.py", "#"), ("a.TS", "//"), ("a.lua", "--"), ("README.md", None)],
)
def test_comment_prefix(name: str, expected: str | None) -> None:
    """Each language gets its own comment syntax, case-insensitively; unknown files are skipped."""
    prefix = comment_prefix(Path(name))
    assert prefix == expected, f"comment_prefix({name!r}) gave {prefix!r}; check _PREFIXES"


def test_history_reports_deleted_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A comment the developer deleted is reported as rejected after a restart, so it isn't raised again."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    History().record(Path("/x.py"), ["# TODO: [ai] a", "# PERF: [ai] b"])
    dismissed = History().dismissed(Path("/x.py"), "code\n# PERF: [ai] b\n")
    assert dismissed == ["# TODO: [ai] a"], f"got {dismissed!r}; check History persistence and dismissed()"
