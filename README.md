# comment-agent

A language server that quietly reviews your code as you work. Each time you save, it asks a
[Pydantic AI](https://ai.pydantic.dev) agent (Claude Opus 5.5 by default) for improvements and inserts them
as comments directly above the relevant line:

```python
def find_users(db_path, names):
    results = []
    for name in names:
        # PERF: [ai] Opens a new connection per name; connect once before the loop
        conn = sqlite3.connect(db_path)
        # WARN: [ai] f-string SQL is injectable; use a ? placeholder
        rows = conn.execute(f"SELECT * FROM users WHERE name = '{name}'").fetchall()
```

On any `TODO`/`FIXME`/`PERF`/`WARN`/`NOTE` comment (the agent's or your own), the code action
**"Claude Code: …"** hands that task and its location to a background `claude -p` run, which makes the
change and deletes the comment.

## How it behaves

- Comments arrive as LSP edits to your **buffer**, not the file on disk, so there are no "file changed" prompts.
  `u` undoes them, and they're saved with your next `:w`.
- The agent only flags what a strong reviewer would raise. Delete a comment you disagree with and it's
  remembered as rejected (`~/.cache/comment-agent/history.json`) and won't come back.
- Reviews wait `--debounce` seconds after a save (default 2) and at most once per `--cooldown` per file
  (default 60). Saves that don't change anything, including saving the agent's own comments, are skipped.
- Git-ignored files, vendored/build directories and files over 3000 lines are skipped.
- Remove every agent comment in a buffer with `:g/\[ai\]/d`.

## Install

```sh
uv tool install -e .
export ANTHROPIC_API_KEY=...   # must be visible to Neovim
```

Any Pydantic AI model works: `--model openai:gpt-5` or `COMMENT_AGENT_MODEL=...`.
The Claude Code action uses `claude` from your `PATH` (or `--claude=/path/to/claude`); pass extra flags with
`--claude-arg`, e.g. `--claude-arg=--model=sonnet`.

## Neovim (0.11+)

```lua
vim.lsp.config('comment_agent', {
  cmd = { 'comment-agent' },
  filetypes = { 'python', 'lua', 'javascript', 'typescript', 'typescriptreact', 'go', 'rust', 'sh', 'c', 'cpp', 'java' },
  root_markers = { '.git' },
})
vim.lsp.enable('comment_agent')

-- Claude Code edits files on disk; reload buffers when it's done.
vim.o.autoread = true
vim.api.nvim_create_autocmd({ 'FocusGained', 'BufEnter', 'CursorHold' }, { command = 'silent! checktime' })
```

Put the cursor on a comment and press `gra` (or run `:lua vim.lsp.buf.code_action()`) to delegate it.
Save the buffer first: the action refuses to start on unsaved changes, since Claude Code works on the file on disk.
Review progress and Claude Code runs show up as LSP progress (e.g. in fidget.nvim or your statusline),
and results arrive as notifications. `:LspLog` has the details.

## Development

```sh
uv run pytest                 # includes an end-to-end test against headless Neovim with a scripted model
uv run ruff check src tests   # every rule; the one exception is explained in pyproject.toml
uv run pyright --warnings     # strict mode
```
