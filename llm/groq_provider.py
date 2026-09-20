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
import time

import anyio
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

# Bounded retry policy for *transient* failures only -- a dropped
# connection, a request that timed out, or a rate limit that may well have
# cleared a second later. DEFAULT_MAX_RETRIES=2 means up to 3 total
# attempts. Deliberately small: this project's per-attempt timeout is
# already 30s, so a full retry budget can add real, user-visible latency;
# it exists to smooth over a transient blip, not to paper over a
# persistently overloaded or misconfigured endpoint.
#
# Never retried: AuthenticationError (a bad key will be bad on attempt 2
# just as much as attempt 1) and the generic APIError catch-all, which is
# what a permanent 4xx like an invalid model name or a request that
# exceeds the model's context window surfaces as -- retrying an
# unchanged, too-large request fails identically every time; the fix for
# that is a smaller request (see agent/tool_results.py's citation
# filtering and LLM-payload trimming), not a retry.
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_BACKOFF_MULTIPLIER = 2.0


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
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
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
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        # Kept only so error messages can be redacted before logging (see
        # _log_provider_error/_redact below) -- never logged itself, and
        # openai.AsyncOpenAI already holds this same value internally.
        self._api_key = resolved_key
        self._client = openai.AsyncOpenAI(api_key=resolved_key, base_url=GROQ_BASE_URL, timeout=timeout_seconds)

    def _log_provider_error(self, exc: Exception, category: str, attempt: int, duration_ms: float) -> None:
        """Log a Groq/provider failure server-side with enough detail to
        actually diagnose it (exception type, HTTP status code, API error
        code, the configured model, how long the attempt took, and a
        redacted message) -- without ever logging the API key, an
        Authorization header, or raw environment variables. This is a
        deliberate fix for exactly the failure mode where every /chat
        request comes back as a clean 200 with the generic "trouble
        reaching the language model" answer and nothing appears in server
        logs to say why: that graceful-degradation path
        (agent/orchestrator.py catching LLMProviderError) is correct
        user-facing behavior, but it must not be the only place this ever
        surfaces. `duration_ms` in particular is what separates "this hung
        for the full timeout" from "this failed instantly" (e.g. an
        immediate 429 or a context-length-exceeded 400) in the log record,
        without needing to guess from the exception type alone.
        """
        status_code = getattr(exc, "status_code", None)
        error_code = getattr(exc, "code", None)
        raw_message = getattr(exc, "message", None) or str(exc)
        message = _redact(raw_message, self._api_key)
        logger.error(
            "Groq provider error (%s): exception_type=%s status_code=%s error_code=%s model=%s "
            "attempt=%d/%d duration_ms=%.0f message=%s",
            category, type(exc).__name__, status_code, error_code, self.model,
            attempt, self._max_retries + 1, duration_ms, message,
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        """Honor the API's own Retry-After header when it gives one
        (common on a 429); otherwise fall back to exponential backoff from
        `retry_base_delay_seconds`. `attempt` is 1-based (the attempt that
        just failed)."""
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass
        return self._retry_base_delay_seconds * (_RETRY_BACKOFF_MULTIPLIER ** (attempt - 1))

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        openai_messages = [_to_openai_message(m) for m in messages]
        openai_tools = [_to_openai_tool(t) for t in tools] if tools else openai.NOT_GIVEN

        attempt = 0
        while True:
            attempt += 1
            attempt_start = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    tools=openai_tools,
                    temperature=self.temperature,
                )
            except (openai.APIConnectionError, openai.RateLimitError) as exc:
                # APIConnectionError's subclass APITimeoutError is caught
                # here too (isinstance matches subclasses) -- both a bare
                # connection failure and a timeout are equally transient.
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                if isinstance(exc, openai.APITimeoutError):
                    category = "timeout"
                elif isinstance(exc, openai.RateLimitError):
                    category = "rate_limited"
                else:
                    category = "connection_error"
                self._log_provider_error(exc, category, attempt, duration_ms)

                if attempt > self._max_retries:
                    if category == "timeout":
                        raise LLMProviderError(f"Groq API request timed out after {self._timeout_seconds}s.") from exc
                    if category == "rate_limited":
                        raise LLMProviderError("Groq API rate limit exceeded.") from exc
                    raise LLMProviderError(f"Groq API connection error: {exc}") from exc

                delay = self._retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Retrying Groq request after a %s error (attempt %d/%d, waiting %.1fs)...",
                    category, attempt, self._max_retries + 1, delay,
                )
                await anyio.sleep(delay)
                continue
            except openai.AuthenticationError as exc:
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                self._log_provider_error(exc, "authentication_failed", attempt, duration_ms)
                # Never retried: a bad key is bad on every attempt.
                raise LLMProviderError(
                    "Groq API rejected the configured GROQ_API_KEY (authentication failed)."
                ) from exc
            except openai.APIError as exc:
                # Every other 4xx/5xx (e.g. an invalid model name, or a
                # request too large for the model's context window) lands
                # here. Never retried: retrying an unchanged request that
                # was rejected as malformed/too-large fails identically
                # every time -- the fix is a smaller/corrected request, not
                # a retry (see agent/tool_results.py's citation filtering
                # and LLM-payload trimming for the actual fix to the
                # too-large-request case).
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                self._log_provider_error(exc, "api_error", attempt, duration_ms)
                raise LLMProviderError(f"Groq API error: {exc}") from exc

            choice = response.choices[0]
            return LLMResponse(
                message=_from_openai_message(choice.message),
                finish_reason=choice.finish_reason or "stop",
                raw=response,
            )
