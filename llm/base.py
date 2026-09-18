"""Provider-agnostic LLM interface.

`agent/` talks only to this interface (`LLMProvider.chat`), never to a
specific vendor SDK, so swapping providers (or using `FakeLLMProvider` in
tests) never touches orchestration code. The shapes here (`ChatMessage`,
`ToolDefinition`, `ToolCall`, `LLMResponse`) are a deliberately small,
OpenAI-tool-calling-shaped common denominator, since that shape is what
Groq (this project's primary provider) and most other OpenAI-compatible
providers already speak natively — a provider implementation's job is
just to translate to/from its SDK's exact request/response objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """One tool invocation the model is requesting."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatMessage:
    """One turn in the conversation sent to/received from the model.

    - A `"tool"` message reports one tool's result back to the model and
      must set `tool_call_id` to the `ToolCall.id` it answers.
    - An `"assistant"` message either has `content` (a final answer) or
      `tool_calls` (the model wants tools invoked before it can answer) —
      never both in this project's usage.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolDefinition:
    """A tool the model may choose to call, in JSON-Schema-described form —
    the same shape MCP's `tools/list` already returns, so the agent can
    pass discovered MCP tools straight through without translation beyond
    field renaming."""

    name: str
    description: str
    input_schema: dict


@dataclass
class LLMResponse:
    """One model turn."""

    message: ChatMessage
    finish_reason: str
    raw: object = field(default=None, repr=False)


class LLMConfigError(RuntimeError):
    """Raised for a provider misconfiguration (e.g. a missing API key),
    at provider construction time — before any network call is made."""


class LLMProviderError(RuntimeError):
    """Raised for a runtime provider failure: timeout, authentication
    rejected by the API, rate limit, or any other API error. Always
    carries a clean, specific message; never a raw SDK exception leaks
    past a provider implementation."""


class LLMProvider(Protocol):
    """The interface `agent/` depends on. Implementations: `GroqProvider`
    (production) and `FakeLLMProvider` (deterministic tests)."""

    model: str

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Send the conversation so far (plus available tool schemas) and
        get back the model's next turn — either a final answer or a
        request to call one or more tools."""
        ...
