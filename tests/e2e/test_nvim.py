"""End-to-end: a real headless Neovim talking to the language server, with a stub review agent."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

NVIM = shutil.which("nvim")
pytestmark = pytest.mark.skipif(NVIM is None, reason="needs nvim")

INIT = """
local dir, server = ...
vim.lsp.start({ name = 'comment_agent', cmd = { %(python)r, server, dir }, root_dir = dir })
"""

SCRIPT = """
local out = {}
local client
vim.wait(5000, function() client = vim.lsp.get_clients({ bufnr = 0 })[1]; return client ~= nil end)
vim.wait(1000, function() return client.initialized end)
vim.cmd('write')
vim.wait(5000, function() return vim.api.nvim_buf_line_count(0) > 4 end)
out.lines = vim.api.nvim_buf_get_lines(0, 0, -1, false)
out.modified = vim.bo.modified
local params = {
  textDocument = { uri = vim.uri_from_bufnr(0) },
  range = { start = { line = 2, character = 0 }, ['end'] = { line = 2, character = 0 } },
  context = { diagnostics = {} },
}
local resp = client:request_sync("textDocument/codeAction", params, 5000, vim.api.nvim_get_current_buf())
out.actions = vim.tbl_map(function(a) return a.title end, resp.result or {})
vim.fn.writefile({ vim.json.encode(out) }, %(result)r)
vim.cmd('qa!')
"""


def test_review_on_save_inserts_comment_into_buffer(tmp_path: Path) -> None:
    """Saving in Neovim puts the review's comment in the buffer, and offers the Claude Code action on it."""
    assert NVIM is not None, "skipif should have skipped this test when nvim isn't on PATH"
    source = tmp_path / "sample.py"
    source.write_text("def f(xs):\n    for x in xs:\n        print(sorted(xs))\n    return 1\n")
    result = tmp_path / "result.json"
    init = tmp_path / "init.lua"
    init.write_text(INIT % {"python": sys.executable})
    script = tmp_path / "script.lua"
    script.write_text(SCRIPT % {"result": str(result)})
    server = Path(__file__).with_name("fake_server.py")

    subprocess.run(  # noqa: S603 - fixed argv, only test-controlled paths
        [
            NVIM,
            "--headless",
            "--clean",
            "-c",
            f"lua loadfile({str(init)!r})({str(tmp_path)!r}, {str(server)!r})",
            "-c",
            f"luafile {script}",
            str(source),
        ],
        cwd=tmp_path,
        timeout=30,
        check=True,
        capture_output=True,
    )

    out: dict[str, Any] = json.loads(result.read_text())
    lines, actions = out["lines"], out["actions"]
    assert lines[2] == "        # PERF: [ai] Sort once outside the loop", (
        f"buffer is {lines!r}; the review's workspace/applyEdit didn't land above line 3 (see :LspLog)"
    )
    assert lines[3] == "        print(sorted(xs))", f"buffer is {lines!r}; the edit must insert, not replace"
    assert out["modified"], "buffer is unmodified, so the edit went to disk behind Neovim's back, not to the buffer"
    on_disk = source.read_text()
    assert "[ai]" not in on_disk, f"disk has {on_disk!r}; edits must only reach disk when the user saves"
    assert actions == ["Claude Code: PERF Sort once outside the loop"], (
        f"code actions on the comment were {actions!r}; check task_actions() and that didChange reached the server"
    )
