"""comment-agent: a language server that reviews code on save and leaves TODO/FIXME/PERF comments."""

import argparse
import shutil

from .review import DEFAULT_MODEL
from .server import Config, create_server


def main() -> None:
    """Parse the command line and serve LSP over stdio."""
    parser = argparse.ArgumentParser(
        prog="comment-agent",
        description="Language server that reviews files on save and leaves TODO/FIXME/PERF comments.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pydantic AI model string (default: %(default)s)")
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="seconds to wait after a save (default: %(default)s)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=60.0,
        help="minimum seconds between reviews of a file (default: %(default)s)",
    )
    parser.add_argument(
        "--claude",
        default=shutil.which("claude"),
        help="path to the Claude Code CLI for code-action runs (default: `claude` on PATH, %(default)s)",
    )
    parser.add_argument(
        "--claude-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument for `claude -p` code-action runs, e.g. --claude-arg=--model=sonnet (repeatable)",
    )
    parser.add_argument("--stdio", action="store_true", help="accepted for compatibility; stdio is always used")
    args = parser.parse_args()
    if args.debounce < 0 or args.cooldown < 0:
        parser.error("--debounce and --cooldown must be non-negative")
    if not args.model.strip():
        parser.error("--model must not be empty")

    config = Config(
        model=args.model,
        claude=args.claude or "claude",
        debounce=args.debounce,
        cooldown=args.cooldown,
        claude_args=args.claude_arg,
    )
    assert config.debounce >= 0, f"debounce {config.debounce} is negative after validation; keep parser.error above"
    assert config.cooldown >= 0, f"cooldown {config.cooldown} is negative after validation; keep parser.error above"
    server = create_server(config)
    assert server.config is config, f"server runs with {server.config!r}, not the parsed {config!r}; pass it through"
    server.start_io()
