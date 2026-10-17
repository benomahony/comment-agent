"""The Pydantic AI review agent and the state that keeps it quiet."""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import cast

import pydantic_ai
from pydantic_ai import Agent, NativeOutput

from .comments import MARKER, Review

DEFAULT_MODEL = os.environ.get("COMMENT_AGENT_MODEL", "anthropic:claude-opus-5-5")
MAX_LINES = 3000
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
    "target",
    ".next",
    "vendor",
}

INSTRUCTIONS = f"""\
You are a quiet, senior code reviewer who leaves comments in a developer's source file while they work.
Each suggestion becomes a single comment line inserted directly above the line you choose.

Tags:
- FIXME: a bug or likely bug (wrong logic, unhandled edge case, resource leak, race).
- PERF: a concrete performance improvement (needless repeated work, wrong data structure, N+1 calls).
- WARN: a security or robustness risk (injection, unsafe input handling, swallowed errors).
- TODO: a missing piece or a worthwhile refactor (duplication, unclear naming, missing validation).
- NOTE: something non-obvious the developer should know (a subtle API behaviour, a footgun).

Guidelines:
- Be selective. Only flag things a strong reviewer would actually raise; zero suggestions is a fine answer.
  Never comment on style a formatter or linter would handle, and never praise.
- Be specific: name the problem and the fix, e.g. "Called per item; hoist the regex compile out of the loop".
- Point at the first line of the statement or block concerned, never at a line inside a multi-line
  string, docstring or expression continuation, since the comment must keep the file valid.
- Lines containing `{MARKER}` are comments you left earlier. Do not repeat or rephrase them.
- The developer has deleted some of your earlier comments; they are listed when present. Treat those as
  rejected and do not raise them again.
- The file is shown with line numbers as `<n> | <code>`; the numbers are not part of the code.
"""


def build_agent(model: str) -> Agent[None, Review]:
    """Build the review agent. The model is resolved on first run, so a missing API key surfaces there."""
    assert model.strip(), f"build_agent got model {model!r}; pass a Pydantic AI model string such as {DEFAULT_MODEL!r}"
    # The language server speaks JSON-RPC on stdout, so Pydantic AI's first-run banner must never print.
    pydantic_ai.BANNER_ENABLED = False
    agent = Agent(model, output_type=NativeOutput(Review), instructions=INSTRUCTIONS, defer_model_check=True)
    assert isinstance(agent.output_type, NativeOutput), (
        f"agent output type is {agent.output_type!r}; keep NativeOutput so Opus doesn't need forced tool use"
    )
    return agent


def review_prompt(display_path: str, text: str, dismissed: list[str]) -> str:
    """Build the user prompt: the line-numbered file, plus any comments the developer rejected."""
    assert display_path, f"review_prompt got display path {display_path!r}; pass CommentAgentServer.display(path)"
    assert all("\n" not in d for d in dismissed), f"review_prompt got multi-line {dismissed!r}; use History.dismissed"
    lines = text.splitlines()
    width = len(str(len(lines)))
    numbered = "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1))
    parts = [f"File: {display_path}", f"<file>\n{numbered}\n</file>"]
    if dismissed:
        parts.append("Comments the developer deleted (do not raise again):\n" + "\n".join(f"- {d}" for d in dismissed))
    prompt = "\n\n".join(parts)
    assert display_path in prompt, f"review_prompt lost the path {display_path!r}; keep it in the parts list"
    assert "<file>" in prompt, f"review_prompt for {display_path!r} lost the file block; keep it in the parts list"
    return prompt


class History:
    """Comments the agent has inserted, per file, so deleted ones can be treated as rejected.

    Stored at a fixed location under the user's cache directory ($XDG_CACHE_HOME or ~/.cache).
    """

    LIMIT = 100

    def __init__(self) -> None:
        """Load the history, starting empty if the file is missing or unreadable."""
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.path = cache / "comment-agent" / "history.json"
        assert self.path.parent.parent == cache, f"history path {self.path} is outside {cache}; keep it fixed there"
        self._data = self._load()
        assert all(isinstance(c, str) for cs in self._data.values() for c in cs), (
            f"{self.path} loaded non-string comments; fix History._load()'s filtering"
        )

    def dismissed(self, file: Path, text: str) -> list[str]:
        """Comments recorded for `file` that are no longer in `text`: the developer deleted them."""
        assert file.is_absolute(), f"History.dismissed got {file}; pass the resolved path so keys match record()"
        dismissed = [c for c in self._data.get(str(file), []) if c not in text]
        assert len(dismissed) <= self.LIMIT, (
            f"{len(dismissed)} dismissed comments for {file} exceeds {self.LIMIT}; record() must trim the history"
        )
        return dismissed

    def record(self, file: Path, comments: list[str]) -> None:
        """Remember comments inserted into `file`, keeping the most recent `LIMIT`."""
        assert file.is_absolute(), f"History.record got {file}; pass the resolved path so keys match dismissed()"
        assert all(c and c.strip() == c for c in comments), f"History.record got {comments!r}; strip indentation first"
        # dict.fromkeys de-duplicates in order, so merging is one step rather than a check per comment.
        merged = list(dict.fromkeys([*self._data.get(str(file), []), *comments]))[-self.LIMIT :]
        self._data[str(file)] = merged
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=1))
        assert set(comments) <= set(merged) or len(comments) > self.LIMIT, (
            f"History.record dropped new comments for {file}; trim the oldest entries, not the newest"
        )

    def _load(self) -> dict[str, list[str]]:
        assert self.path.suffix == ".json", f"history path {self.path} isn't JSON; keep History.path as history.json"
        try:
            raw: object = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        entries = cast("dict[object, object]", raw)
        data = {
            str(file): [c for c in cast("list[object]", comments) if isinstance(c, str)]
            for file, comments in entries.items()
            if isinstance(comments, list)
        }
        assert len(data) <= len(entries), (
            f"History._load made {len(data)} entries from {len(entries)}; it may only drop"
        )
        return data


async def is_ignored(path: Path, root: Path | None) -> bool:
    """Whether `path` is in a vendored/build directory or ignored by git."""
    assert path.is_absolute(), f"is_ignored got {path}; resolve the document path first"
    assert root is None or root.is_absolute(), f"is_ignored got root {root}; use CommentAgentServer.root"
    parts = path.relative_to(root).parts if root and path.is_relative_to(root) else path.parts
    if IGNORED_DIRS.intersection(parts[:-1]):
        return True
    if (git := shutil.which("git")) is None:
        return False
    try:
        process = await asyncio.create_subprocess_exec(
            git,
            "check-ignore",
            "-q",
            path.name,
            cwd=path.parent,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await asyncio.wait_for(process.wait(), timeout=5) == 0
    except (OSError, TimeoutError):
        return False
