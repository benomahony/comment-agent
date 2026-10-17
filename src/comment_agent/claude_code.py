"""Handing a single task comment to a headless Claude Code run."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]


@dataclass(frozen=True)
class Outcome:
    """How a Claude Code run ended, and its one-sentence summary or error."""

    ok: bool
    summary: str


def task_prompt(display_path: str, line: int, comment: str) -> str:
    """Write the instructions for Claude Code to address `comment` at `display_path:line`."""
    assert display_path, f"task_prompt got display path {display_path!r}; pass CommentAgentServer.display(path)"
    assert line >= 1, f"task_prompt got line {line}; pass the 1-based line number"
    assert "\n" not in comment, f"task_prompt got {comment!r}; pass the comment without its newline"
    prompt = (
        f"In {display_path} at line {line} there is this comment:\n\n"
        f"    {comment.strip()}\n\n"
        "Make the change it asks for. Keep the change minimal and focused on that comment; "
        "do not touch unrelated code. Once it is fully addressed, delete the comment line. "
        "Finish with a one-sentence summary of what you changed."
    )
    assert f"{display_path} at line {line}" in prompt, (
        f"task_prompt dropped the location {display_path}:{line}; keep it in the prompt text"
    )
    return prompt


async def run(claude: str, prompt: str, cwd: Path, extra_args: Sequence[str] = ()) -> Outcome:
    """Run `claude -p` with edit permissions in `cwd` and wait for it to finish."""
    assert claude, "claude_code.run got no executable; pass Config.claude"
    assert prompt.strip(), f"claude_code.run got prompt {prompt!r}; build it with task_prompt()"
    try:
        process = await asyncio.create_subprocess_exec(
            claude,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            *ALLOWED_TOOLS,
            "--no-session-persistence",
            *extra_args,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:  # the executable or the working directory is missing or unusable
        return Outcome(ok=False, summary=f"couldn't start `{claude}` in {cwd}: {e}; check --claude and the workspace")
    stdout, stderr = await process.communicate()
    assert process.returncode is not None, (
        f"claude still running (pid {process.pid}) after communicate(); await process.wait() before reading results"
    )
    try:
        raw: object = json.loads(stdout)
    except ValueError:
        return Outcome(ok=False, summary=(stderr or stdout).decode(errors="replace").strip()[-500:] or "no output")
    if not isinstance(raw, dict):
        return Outcome(ok=False, summary=f"unexpected output: {stdout[:200]!r}")
    result = cast("dict[str, object]", raw)
    summary = str(result.get("result") or result.get("subtype") or "").strip()
    return Outcome(ok=process.returncode == 0 and not result.get("is_error"), summary=summary)
