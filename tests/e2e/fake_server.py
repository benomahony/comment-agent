"""Runs the real language server with a scripted model, for the Neovim end-to-end test."""

import json
import os
import sys

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from comment_agent.comments import Review
from comment_agent.review import INSTRUCTIONS
from comment_agent.server import Config, create_server

SUGGESTION = {"line": 3, "tag": "PERF", "message": "Sort once outside the loop"}


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer every review with one PERF suggestion on line 3, as Claude would in JSON."""
    assert messages, "the server sent the model no messages; check review_prompt and agent.run"
    mode = info.model_request_parameters.output_mode
    assert mode == "native", f"agent asked for {mode!r} output; the server must build agents with NativeOutput"
    return ModelResponse(parts=[TextPart(json.dumps({"suggestions": [SUGGESTION]}))])


if __name__ == "__main__":
    os.environ["XDG_CACHE_HOME"] = sys.argv[1]
    agent = Agent(FunctionModel(respond), output_type=NativeOutput(Review), instructions=INSTRUCTIONS)
    create_server(Config(debounce=0.05, cooldown=0), agent=agent).start_io()
