"""A deterministic, scripted `LLMProvider` for tests.

`FakeLLMProvider` never makes a network call. A test constructs it with a
list of pre-scripted `LLMResponse`s (helpers below make this concise) and
each `chat()` call pops and returns the next one, recording what it was
sent so a test can assert the agent built the expected conversation state
(system prompt present, prior tool results included, etc.) — this is what
makes it possible to prove the agent orchestration loop genuinely reacts
to model-selected tool calls rather than running a hard-coded workflow:
the *fake model* decides which tool to call next, and the test asserts
the agent actually called it.
"""
from __future__ import annotations

import uuid

from llm.base import ChatMessage, LLMProviderError, LLMResponse, ToolCall, ToolDefinition


def tool_call_turn(name: str, arguments: dict, call_id: str | None = None) -> LLMResponse:
    """Build a scripted 'model wants to call this tool' response."""
    call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
    return LLMResponse(
        message=ChatMessage(role="assistant", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]),
        finish_reason="tool_calls",
    )


def final_turn(content: str) -> LLMResponse:
    """Build a scripted 'model has a final answer' response."""
    return LLMResponse(message=ChatMessage(role="assistant", content=content), finish_reason="stop")


class FakeLLMProvider:
    """Deterministic scripted provider. See module docstring."""

    model = "fake-model"

    def __init__(self, scripted_responses: list[LLMResponse]) -> None:
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools) if tools else []})
        if not self._responses:
            raise LLMProviderError(
                "FakeLLMProvider ran out of scripted responses "
                f"(received {len(self.calls)} chat() calls; check the test's script length)."
            )
        return self._responses.pop(0)
