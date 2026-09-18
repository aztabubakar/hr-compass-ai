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

import logging
import os
import re

import openai

from llm.base import ChatMessage, LLMConfigError, LLMProvider, LLMProviderError, LLMResponse, ToolCall, ToolDefinition

logger = logging.getLogger("hr_compass_ai.llm.groq")

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


_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9._\-]+")


def _redact(text: str | None, secret: str | None) -> str:
    """Strip anything that could be the configured API key, or a bearer
    token, out of a string before it's ever logged. Applied defensively to
    every error message this module logs, even though `openai`'s status
    errors are built from the API *response body* (Groq's own error text),
    never from our outgoing request headers -- see `_make_status_error_from_response`
    in the `openai` SDK -- so the key should never actually appear here.
    This is belt-and-suspenders, not a sign the key was observed leaking.
    """
    if not text:
        return "" if text is None else text
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return _BEARER_PATTERN.sub(r"\1[REDACTED]", text)


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
        # Kept only so error messages can be redacted before logging (see
        # _log_provider_error/_redact below) -- never logged itself, and
        # openai.AsyncOpenAI already holds this same value internally.
        self._api_key = resolved_key
        self._client = openai.AsyncOpenAI(api_key=resolved_key, base_url=GROQ_BASE_URL, timeout=timeout_seconds)

    def _log_provider_error(self, exc: Exception, category: str) -> None:
        """Log a Groq/provider failure server-side with enough detail to
        actually diagnose it (exception type, HTTP status code, API error
        code, the configured model, and a redacted message) -- without ever
        logging the API key, an Authorization header, or raw environment
        variables. This is a deliberate fix for exactly the failure mode
        where every /chat request comes back as a clean 200 with the
        generic "trouble reaching the language model" answer and nothing
        appears in server logs to say why: that graceful-degradation path
        (agent/orchestrator.py catching LLMProviderError) is correct
        user-facing behavior, but it must not be the only place this ever
        surfaces.
        """
        status_code = getattr(exc, "status_code", None)
        error_code = getattr(exc, "code", None)
        raw_message = getattr(exc, "message", None) or str(exc)
        message = _redact(raw_message, self._api_key)
        logger.error(
            "Groq provider error (%s): exception_type=%s status_code=%s error_code=%s model=%s message=%s",
            category, type(exc).__name__, status_code, error_code, self.model, message,
        )

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
            self._log_provider_error(exc, "timeout")
            raise LLMProviderError(
                f"Groq API request timed out after {self._timeout_seconds}s."
            ) from exc
        except openai.AuthenticationError as exc:
            self._log_provider_error(exc, "authentication_failed")
            raise LLMProviderError(
                "Groq API rejected the configured GROQ_API_KEY (authentication failed)."
            ) from exc
        except openai.RateLimitError as exc:
            self._log_provider_error(exc, "rate_limited")
            raise LLMProviderError("Groq API rate limit exceeded.") from exc
        except openai.APIError as exc:
            self._log_provider_error(exc, "api_error")
            raise LLMProviderError(f"Groq API error: {exc}") from exc

        choice = response.choices[0]
        return LLMResponse(
            message=_from_openai_message(choice.message),
            finish_reason=choice.finish_reason or "stop",
            raw=response,
        )
