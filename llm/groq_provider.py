"""Groq provider, implemented via the official `openai` SDK pointed at
Groq's OpenAI-compatible endpoint (https://api.groq.com/openai/v1).

Groq was chosen as the primary provider because it offers a genuinely
free tier with fast inference for open-weight tool-calling-capable models
(e.g. Llama 3.3), which fits this project's free-tier-compatible
requirement well. Using the `openai` SDK (rather than a bespoke HTTP
client, or Groq's own SDK) rather than gives a second benefit for free:
any other OpenAI-compatible provider (OpenAI itself, Together, Fireworks,
a local vLLM/Ollama server exposing an OpenAI-compatible endpoint, etc.)
can be added later by adding another thin subclass that only changes
`base_url`/the env vars it reads — `GroqProvider` below is a template for
that, not a one-off.
"""
from __future__ import annotations

import os

import openai

from llm.base import ChatMessage, LLMConfigError, LLMProvider, LLMProviderError, LLMResponse, ToolCall, ToolDefinition

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Overridable via GROQ_MODEL; this default is a current (as of writing)
# Groq-hosted model with reliable tool-calling support. Per CLAUDE.md/this
# phase's instructions, this is a fallback default, not a hard-coded
# requirement — set GROQ_MODEL to use a different one without code changes.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2  # low/deterministic-leaning, appropriate for HR policy answers
DEFAULT_TIMEOUT_SECONDS = 30.0


def _to_openai_message(message: ChatMessage) -> dict:
    out: dict = {"role": message.role}
    if message.content is not None:
        out["content"] = message.content
    elif message.role == "assistant" and not message.tool_calls:
        out["content"] = ""
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": _dumps(tc.arguments)},
            }
            for tc in message.tool_calls
        ]
    if message.tool_call_id:
        out["tool_call_id"] = message.tool_call_id
    if message.name:
        out["name"] = message.name
    return out


def _dumps(arguments: dict) -> str:
    import json

    return json.dumps(arguments)


def _to_openai_tool(tool: ToolDefinition) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _from_openai_message(message: object) -> ChatMessage:
    import json

    tool_calls = None
    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls:
        tool_calls = []
        for tc in raw_tool_calls:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    return ChatMessage(role="assistant", content=message.content, tool_calls=tool_calls)


class GroqProvider:
    """`LLMProvider` implementation calling Groq's chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and set GROQ_API_KEY to a "
                "key from https://console.groq.com/keys, or set it as an environment variable."
            )
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self.temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._client = openai.AsyncOpenAI(api_key=resolved_key, base_url=GROQ_BASE_URL, timeout=timeout_seconds)

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[_to_openai_message(m) for m in messages],
                tools=[_to_openai_tool(t) for t in tools] if tools else openai.NOT_GIVEN,
                temperature=self.temperature,
            )
        except openai.APITimeoutError as exc:
            raise LLMProviderError(
                f"Groq API request timed out after {self._timeout_seconds}s."
            ) from exc
        except openai.AuthenticationError as exc:
            raise LLMProviderError(
                "Groq API rejected the configured GROQ_API_KEY (authentication failed)."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMProviderError("Groq API rate limit exceeded.") from exc
        except openai.APIError as exc:
            raise LLMProviderError(f"Groq API error: {exc}") from exc

        choice = response.choices[0]
        return LLMResponse(
            message=_from_openai_message(choice.message),
            finish_reason=choice.finish_reason or "stop",
            raw=response,
        )
