"""The language server: review on save, delegate task comments to Claude Code via a code action."""

import asyncio
import hashlib
import math
import uuid
from collections import Counter
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from lsprotocol import types
from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError, UserError
from pygls.exceptions import JsonRpcException
from pygls.lsp.server import LanguageServer

from . import claude_code
from .comments import (
    Edit,
    Review,
    apply_placements,
    comment_prefix,
    detect_newline,
    parse_task_comment,
    place_edits,
    plan_edits,
)
from .review import DEFAULT_MODEL, MAX_LINES, History, build_agent, is_ignored, review_prompt

DELEGATE_COMMAND = "commentAgent.delegate"
SHA256_HEX_LENGTH = 64
HEX_DIGITS = frozenset("0123456789abcdef")
MAX_ACTION_TITLE = 80


@dataclass
class Config:
    """Command-line settings for the server."""

    model: str = DEFAULT_MODEL
    claude: str = "claude"
    debounce: float = 2.0
    cooldown: float = 60.0
    claude_args: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class _Target:
    """A document chosen for review."""

    path: Path
    name: str
    prefix: str
    text: str


def _digest(text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert len(digest) == SHA256_HEX_LENGTH, (
        f"_digest made {digest!r}; keep it a sha256 hex digest so seen-checks match"
    )
    assert set(digest) <= HEX_DIGITS, f"_digest made non-hex {digest!r}; use hexdigest(), not digest()"
    return digest


class CommentAgentServer(LanguageServer):
    """Reviews documents on save and offers task comments to Claude Code."""

    def __init__(
        self,
        config: Config,
        agent: Agent[None, Review] | None = None,
        history: History | None = None,
    ) -> None:
        """Set up review state; the agent defaults to one built from `config.model`."""
        assert config.debounce >= 0, f"server got debounce {config.debounce}; validate --debounce before creating it"
        assert config.cooldown >= 0, f"server got cooldown {config.cooldown}; validate --cooldown before creating it"
        super().__init__("comment-agent", "0.1.0")  # pyright: ignore[reportUnknownMemberType] - pygls leaves some params untyped
        self.config = config
        self.agent = agent or build_agent(config.model)
        self.history = history or History()
        # Text we last reviewed or produced, so no-op saves and our own edits don't trigger reviews.
        self.seen: dict[str, str] = {}
        self.pending: dict[str, asyncio.Task[None]] = {}
        self.last_review: dict[str, float] = {}
        self.review_slots = asyncio.Semaphore(2)
        self.review_locks: dict[str, asyncio.Lock] = {}
        self.file_locks: dict[Path, asyncio.Lock] = {}
        self._error_counts: Counter[str] = Counter()

    @property
    def root(self) -> Path | None:
        """The workspace root, if the editor sent one."""
        root_path = self.workspace.root_path
        assert root_path is None or Path(root_path).is_absolute(), (
            f"workspace root {root_path!r} is relative; the editor must send rootUri or workspaceFolders as file URIs"
        )
        # Resolved like document paths are (e.g. /var -> /private/var on macOS), so relative_to() works.
        root = Path(root_path).resolve() if root_path else None
        assert root is None or root.is_absolute(), f"resolved root {root} is relative; check Path.resolve()"
        return root

    def display(self, path: Path) -> str:
        """`path` relative to the workspace root when it's inside it, for messages and prompts."""
        assert path.is_absolute(), f"display got {path}; pass the document's absolute path"
        root = self.root
        shown = str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path)
        assert shown != ".", f"display turned {path} into {shown!r} under root {root}; pass a file, not the root"
        return shown

    def log(self, message: str) -> None:
        """Write to the editor's LSP log (`:LspLog` in Neovim)."""
        assert message, f"log got {message!r}; pass the text to write to :LspLog"
        assert "comment-agent:" not in message, f"log got pre-prefixed {message!r}; only notify() adds the prefix"
        self.window_log_message(types.LogMessageParams(type=types.MessageType.Log, message=message))

    def notify(self, message: str, kind: types.MessageType = types.MessageType.Info) -> None:
        """Show a message to the user (`vim.notify` in Neovim)."""
        assert message, f"notify got {message!r}; pass the text to show the user"
        assert kind in types.MessageType, f"notify got kind {kind!r}; pass a types.MessageType member"
        self.window_show_message(types.ShowMessageParams(type=kind, message=f"comment-agent: {message}"))

    def report_error(self, message: str) -> None:
        """Log every error, but pop up each distinct one only once per session."""
        assert message, f"report_error got {message!r}; pass the error text"
        self.log(message)
        self._error_counts[message] += 1
        count = self._error_counts[message]
        if count == 1:
            self.notify(message, types.MessageType.Error)
        assert count >= 1, f"report_error counted {message!r} {count} times; increment before deciding to notify"

    @asynccontextmanager
    async def reporting_progress(self, title: str, message: str) -> AsyncGenerator[None]:
        """Show LSP work-done progress for the duration of the block, if the editor supports it."""
        assert title, f"progress got title {title!r}; pass the name shown in the progress UI"
        assert message, f"progress {title!r} got message {message!r}; say what is in progress"
        token: str | None = str(uuid.uuid4())
        try:
            await self.work_done_progress.create_async(token)  # pyright: ignore[reportUnknownMemberType] - untyped in pygls
            self.work_done_progress.begin(token, types.WorkDoneProgressBegin(title=title, message=message))
        except JsonRpcException:
            token = None  # client doesn't support progress; carry on without it
        try:
            yield
        finally:
            if token:
                self.work_done_progress.end(token, types.WorkDoneProgressEnd())

    # Reviews

    def schedule_review(self, uri: str) -> None:
        """Review `uri` after the debounce, and no sooner than the cooldown since its last review."""
        assert uri, f"schedule_review got uri {uri!r}; pass params.text_document.uri"
        if task := self.pending.pop(uri, None):
            task.cancel()
        loop = asyncio.get_running_loop()
        ready_at = self.last_review.get(uri, -math.inf) + self.config.cooldown
        delay = max(self.config.debounce, ready_at - loop.time())
        self.pending[uri] = loop.create_task(self._review_later(uri, delay))
        assert delay >= self.config.debounce, (
            f"review of {uri} delayed {delay}s, under the {self.config.debounce}s debounce"
        )

    async def _review_later(self, uri: str, delay: float) -> None:
        assert delay >= 0, f"_review_later got delay {delay} for {uri}; schedule_review must clamp it to the debounce"
        await asyncio.sleep(delay)
        self.pending.pop(uri, None)
        self.last_review[uri] = asyncio.get_running_loop().time()
        assert uri not in self.pending, f"review of {uri} is still pending after starting; pop it before running"
        await self.review(uri)

    async def review(self, uri: str) -> None:
        """Review a document and insert comments into it. One review per document at a time."""
        assert uri, f"review got uri {uri!r}; pass params.text_document.uri"
        lock = self.review_locks.setdefault(uri, asyncio.Lock())
        async with lock:
            await self._review(uri)
        assert self.review_locks[uri] is lock, f"review lock for {uri} was replaced mid-review; never reassign it"

    async def _review(self, uri: str) -> None:
        assert self.review_locks[uri].locked(), f"_review of {uri} ran without its lock; call it through review()"
        target = await self._review_target(uri)
        if target is None:
            return
        self.log(f"reviewing {target.name}")
        review = await self._ask_model(target)
        if review is None:
            return
        edits = plan_edits(target.text, review.suggestions, target.prefix)
        if not edits:
            self.log(f"{target.name}: nothing to flag")
            return
        added = await self._insert_comments(uri, target, edits)
        assert added <= sum(len(e.comments) for e in edits), (
            f"inserted {added} comments from {edits!r}; check placement"
        )

    async def _review_target(self, uri: str) -> "_Target | None":
        """Pick the document to review, or None if it's unsupported, unchanged, too big or ignored."""
        assert uri in self.review_locks, f"_review_target({uri!r}) called outside review(); go through review()"
        doc = self.workspace.get_text_document(uri)
        path = await asyncio.to_thread(Path(doc.path).resolve)
        prefix = comment_prefix(path)
        text = doc.source
        digest = _digest(text)
        # Swap in the new digest and read the old one in a single statement.
        previous, self.seen[uri] = self.seen.get(uri), digest
        if prefix is None or not text.strip() or previous == digest:
            return None
        if text.count("\n") > MAX_LINES or await is_ignored(path, self.root):
            return None
        target = _Target(path=path, name=self.display(path), prefix=prefix, text=text)
        assert self.seen[uri] == digest, f"seen digest for {uri} changed while choosing it; reviews must hold the lock"
        return target

    async def _ask_model(self, target: "_Target") -> Review | None:
        """Run the review agent on `target`; None (after telling the user) if the model call failed."""
        assert target.text.strip(), f"_ask_model got an empty {target.name}; _review_target should have skipped it"
        prompt = review_prompt(target.name, target.text, self.history.dismissed(target.path, target.text))
        try:
            async with self.review_slots, self.reporting_progress("comment-agent", f"reviewing {target.name}"):
                result = await self.agent.run(prompt)
        except (UserError, AgentRunError) as e:  # missing API key / model or API failure
            self.report_error(
                f"review failed: {e}. Check ANTHROPIC_API_KEY (or your provider's key) and --model "
                f"({self.config.model}); :LspLog has details",
            )
            return None
        assert isinstance(result.output, Review), (
            f"agent returned {result.output!r}; build it with NativeOutput(Review)"
        )
        return result.output

    async def _insert_comments(self, uri: str, target: "_Target", edits: list[Edit]) -> int:
        """Insert `edits` into the editor's buffer; return how many comment lines were added."""
        assert edits, f"_insert_comments got no edits for {target.name}; return early when there's nothing to flag"
        # The buffer may have changed while the model was thinking; anchor against what it holds now.
        # The edit carries the document version, so the editor rejects it if the buffer changes again.
        doc = self.workspace.get_text_document(uri)
        current = doc.source
        placements = place_edits(current, edits)
        if not placements:
            self.log(f"{target.name}: code changed too much since the review; dropped suggestions")
            return 0
        newline = detect_newline(current)
        text_edits: list[types.TextEdit | types.AnnotatedTextEdit | types.SnippetTextEdit] = [
            types.TextEdit(
                range=types.Range(start=types.Position(p.line - 1, 0), end=types.Position(p.line - 1, 0)),
                new_text="".join(line + newline for line in p.lines),
            )
            for p in placements
        ]
        identifier = types.OptionalVersionedTextDocumentIdentifier(uri=uri, version=doc.version)
        edit = types.WorkspaceEdit(
            document_changes=[types.TextDocumentEdit(text_document=identifier, edits=text_edits)],
        )
        response = await self.workspace_apply_edit_async(
            types.ApplyWorkspaceEditParams(label="comment-agent review", edit=edit),
        )
        if not response.applied:
            self.log(f"{target.name}: editor rejected the edit ({response.failure_reason or 'buffer changed'})")
            return 0
        self.seen[uri] = _digest(apply_placements(current, placements))
        comments = [line.strip() for p in placements for line in p.lines]
        self.history.record(target.path, comments)
        self.log(f"{target.name}: added {len(comments)} comment(s)")
        assert all(c.startswith(target.prefix) for c in comments), f"inserted non-comments {comments!r}; use render()"
        return len(comments)

    # Delegation to Claude Code

    def task_actions(self, uri: str, selection: types.Range) -> list[types.CodeAction]:
        """Offer a "Claude Code: …" action for each task comment in `selection`."""
        assert uri, f"task_actions got uri {uri!r}; pass params.text_document.uri"
        lines = self.workspace.get_text_document(uri).lines
        start, end = selection.start, selection.end
        actions: list[types.CodeAction] = []
        for n in range(start.line, min(end.line, len(lines) - 1) + 1):
            if parsed := parse_task_comment(lines[n].rstrip("\r\n")):
                tag, message = parsed
                title = f"Claude Code: {tag} {message}"
                title = title if len(title) <= MAX_ACTION_TITLE else title[: MAX_ACTION_TITLE - 1] + "…"
                actions.append(
                    types.CodeAction(
                        title=title,
                        kind=types.CodeActionKind.QuickFix,
                        command=types.Command(title=title, command=DELEGATE_COMMAND, arguments=[uri, n]),
                    ),
                )
        assert all(len(a.title) <= MAX_ACTION_TITLE for a in actions), (
            f"code action titles {[a.title for a in actions]!r} exceed {MAX_ACTION_TITLE} characters; fix truncation"
        )
        return actions

    def delegate(self, uri: str, line: int) -> asyncio.Task[None]:
        """Hand the task comment on 0-based `line` of `uri` to Claude Code in the background."""
        assert uri, f"delegate got uri {uri!r}; pass the uri from the code action"
        assert line >= 0, f"delegate got line {line}; pass the 0-based line from the code action"
        task = asyncio.get_running_loop().create_task(self._run_claude_code(uri, line))
        assert not task.done(), f"delegation of {uri}:{line} finished synchronously; it must run as a background task"
        return task

    async def _run_claude_code(self, uri: str, line: int) -> None:
        assert line >= 0, f"_run_claude_code got line {line}; pass the 0-based line from the code action"
        path = await asyncio.to_thread(Path(self.workspace.get_text_document(uri).path).resolve)
        name = f"{self.display(path)}:{line + 1}"
        lock = self.file_locks.setdefault(path, asyncio.Lock())
        async with lock:
            assert self.file_locks[path] is lock, (
                f"file lock for {name} was replaced; never reassign file_locks entries"
            )
            # Checked under the lock, right before starting, so a run that queued behind another
            # sees the file as it is now.
            doc = self.workspace.get_text_document(uri)
            lines = doc.lines
            comment = lines[line].rstrip("\r\n") if line < len(lines) else ""
            if parse_task_comment(comment) is None:
                self.notify(f"{name} is no longer a task comment", types.MessageType.Warning)
                return
            try:
                on_disk = await asyncio.to_thread(path.read_text)
            except OSError:
                on_disk = None
            if on_disk != doc.source:
                # Claude Code edits the file on disk; unsaved buffer changes would conflict with it.
                self.notify(f"save {path.name} first so Claude Code sees what you see", types.MessageType.Warning)
                return
            self.notify(f"Claude Code is working on {name}; avoid editing {path.name} until it finishes")
            prompt = claude_code.task_prompt(self.display(path), line + 1, comment)
            async with self.reporting_progress("Claude Code", name):
                outcome = await claude_code.run(
                    self.config.claude,
                    prompt,
                    self.root or path.parent,
                    self.config.claude_args,
                )
        if outcome.ok:
            self.notify(f"Claude Code finished {name}: {outcome.summary}")
        else:
            self.notify(f"Claude Code failed on {name}: {outcome.summary}", types.MessageType.Error)


def create_server(
    config: Config,
    agent: Agent[None, Review] | None = None,
    history: History | None = None,
) -> CommentAgentServer:
    """Build the server and register its LSP handlers."""
    server = CommentAgentServer(config, agent, history)

    @server.feature(types.TEXT_DOCUMENT_DID_SAVE)
    async def did_save(ls: CommentAgentServer, params: types.DidSaveTextDocumentParams) -> None:
        assert ls is server, f"didSave reached {ls!r}, not the server it was registered on; create one per process"
        uri = params.text_document.uri
        ls.schedule_review(uri)
        assert uri in ls.pending, f"save of {uri} scheduled nothing; schedule_review must add it to pending"

    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
    async def did_close(ls: CommentAgentServer, params: types.DidCloseTextDocumentParams) -> None:
        assert ls is server, f"didClose reached {ls!r}, not the server it was registered on; create one per process"
        uri = params.text_document.uri
        if task := ls.pending.pop(uri, None):
            task.cancel()
        assert uri not in ls.pending, f"closed {uri} still has review {ls.pending.get(uri)!r} pending; pop it on close"

    @server.feature(
        types.TEXT_DOCUMENT_CODE_ACTION,
        types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
    )
    def code_action(ls: CommentAgentServer, params: types.CodeActionParams) -> list[types.CodeAction]:
        assert ls is server, f"codeAction reached {ls!r}, not the server it was registered on; create one per process"
        actions = ls.task_actions(params.text_document.uri, params.range)
        assert all(a.command and a.command.command == DELEGATE_COMMAND for a in actions), (
            f"code actions {actions!r} don't all run {DELEGATE_COMMAND}; register any new command before offering it"
        )
        return actions

    @server.command(DELEGATE_COMMAND)
    async def delegate(ls: CommentAgentServer, uri: str, line: int) -> None:
        assert ls is server, f"{DELEGATE_COMMAND} reached {ls!r}, not its server; create one server per process"
        # The line comes back from the editor, so validate it rather than assert.
        if line < 0:
            ls.notify(f"ignoring {DELEGATE_COMMAND} for line {line}", types.MessageType.Warning)
            return
        task = ls.delegate(uri, line)
        assert task in asyncio.all_tasks(), f"delegation of {uri}:{line} isn't scheduled on the running loop"

    assert server.config is config, f"server runs with {server.config!r}, not {config!r}; pass the config through"
    assert not server.pending, f"new server already has reviews pending for {sorted(server.pending)}"
    return server
