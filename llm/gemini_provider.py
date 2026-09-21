"""Gemini provider, implemented via Google's current `google-genai` Python
SDK (not the older, deprecated `google-generativeai` package).

Gemini has a different content model than the OpenAI-compatible shape
`llm/base.py` is built around: there is no "system" or "tool" role in its
`contents` list. A system prompt goes into a separate `system_instruction`
config field; an assistant turn (text or tool calls) is sent as
`role="model"`; and a tool's result is sent back as `role="user"` content
wrapping a `function_response` part. All of that translation happens in
this module only -- `agent/orchestrator.py` and the rest of the project
never see a Gemini-specific type, exactly as `GroqProvider` never leaks an
`openai` type past `llm/groq_provider.py`.

MCP tool schemas are handed to Gemini via
`types.FunctionDeclaration(parameters_json_schema=...)`, which accepts a
raw JSON Schema dict directly -- so this adapter is a thin pass-through
(name, description, schema) rather than a schema-rewriting layer. MCP
remains the sole execution authority; no business logic is duplicated
here.

Gemini 3 models attach an opaque `thought_signature` to the response
`Part` that carries a function call, and require that exact signature to
be replayed on that same function-call Part when its result is sent back
for the next turn -- omitting it fails multi-step tool calling with
"Function call is missing a thought_signature in functionCall parts."
`thought_signature` lives on `types.Part` itself (a sibling of
`function_call`, not a field of `FunctionCall`), so it has to be carried
alongside our own `ToolCall` explicitly: `_from_gemini_response` reads it
off each function-call `Part` into `ToolCall.provider_metadata` (a
provider-opaque field on the shared `llm.base.ToolCall`, unused by every
other provider), and `_parts_for` reattaches it, unchanged, to the
reconstructed `Part` when that same tool call is replayed as history. This
is the smallest fix for the round trip: nothing here decodes, rewrites, or
logs the signature's contents -- it is treated as an opaque blob the whole
way through, exactly like Gemini's own docs describe it (not
chain-of-thought).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid

import anyio
import httpx
from google import genai
from google.genai import errors, types

from llm.base import ChatMessage, LLMConfigError, LLMProvider, LLMProviderError, LLMResponse, ToolCall, ToolDefinition

logger = logging.getLogger("hr_compass_ai.llm.gemini")

# Overridable via GEMINI_MODEL; this default is the official Gemini 2.5
# Flash-Lite model ID, chosen for free-tier quota headroom (10 RPM / 250K
# TPM / 20 RPD, vs. a stricter free-tier quota on other Gemini models). A
# fallback default, not a hard-coded requirement -- set GEMINI_MODEL to use
# a different one without code changes.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SECONDS = 30.0

# Bounded retry policy for genuinely transient failures only -- see
# llm/groq_provider.py's module docstring for the general rationale, which
# applies identically here. DEFAULT_MAX_RETRIES=2 means up to 3 total
# attempts.
#
# Gemini's error hierarchy classifies a 429 (quota/rate-limit) as a
# `ClientError`, unlike Groq/openai's dedicated `RateLimitError` -- so
# "never retry a 4xx" can't be applied as a blanket rule the way it is for
# Groq. Retried: RESOURCE_EXHAUSTED/429 (may clear briefly; still capped
# at the same small retry budget so this can never turn into unbounded
# polling of an exhausted quota) and a 5xx server error. Never retried:
# UNAUTHENTICATED/PERMISSION_DENIED (401/403 -- a bad key is bad on every
# attempt) and INVALID_ARGUMENT/other 4xx (e.g. a malformed request or a
# request too large for the model's context window -- retrying an
# unchanged bad request fails identically every time).
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_BACKOFF_MULTIPLIER = 2.0


_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9._\-]+")


def _redact(text: str | None, secret: str | None) -> str:
    """Strip anything that could be the configured API key, or a bearer
    token, out of a string before it's ever logged. See
    llm/groq_provider.py's `_redact` for the identical rationale."""
    if not text:
        return "" if text is None else text
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return _BEARER_PATTERN.sub(r"\1[REDACTED]", text)


def _role_for(message: ChatMessage) -> str:
    """Gemini's `contents` list only has "user" and "model" roles. A
    "tool" message (a function result reported back to the model) is sent
    as role="user", per Gemini's function-calling convention -- there is
    no dedicated "tool"/"function" role."""
    if message.role in ("user", "tool"):
        return "user"
    return "model"  # assistant


def _parts_for(message: ChatMessage) -> list[types.Part]:
    if message.role == "tool":
        try:
            response_payload = json.loads(message.content) if message.content else {}
        except json.JSONDecodeError:
            response_payload = {"raw": message.content}
        if not isinstance(response_payload, dict):
            response_payload = {"result": response_payload}
        return [types.Part(function_response=types.FunctionResponse(
            id=message.tool_call_id, name=message.name, response=response_payload,
        ))]
    if message.tool_calls:
        # thought_signature (see module docstring) lives on the Part, not
        # on FunctionCall -- reattach it, unchanged, to the exact Part this
        # tool call originally came from. `None` for a tool call that never
        # had one (e.g. from GroqProvider/FakeLLMProvider history, or a
        # Gemini call Gemini itself didn't attach one to) is passed through
        # as-is; never fabricated.
        return [
            types.Part(
                function_call=types.FunctionCall(id=tc.id, name=tc.name, args=tc.arguments),
                thought_signature=(tc.provider_metadata or {}).get("gemini_thought_signature"),
            )
            for tc in message.tool_calls
        ]
    return [types.Part(text=message.content or "")]


def _to_gemini_contents(messages: list[ChatMessage]) -> tuple[str | None, list[types.Content]]:
    """Split the project's uniform `ChatMessage` history into Gemini's
    `system_instruction` (a plain string) plus a `contents` list.

    Consecutive messages that map to the same Gemini role are merged into
    one `Content` with multiple parts -- this is what lets several
    back-to-back `role="tool"` messages (one MCP tool call producing
    several results in the same orchestrator turn) become the single
    multi-`function_response` user turn Gemini expects to answer a
    single multi-`function_call` model turn.
    """
    system_instruction = "\n\n".join(m.content for m in messages if m.role == "system" and m.content) or None

    contents: list[types.Content] = []
    for message in messages:
        if message.role == "system":
            continue
        role = _role_for(message)
        parts = _parts_for(message)
        if contents and contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=parts))
    return system_instruction, contents


def _to_gemini_tool(tool: ToolDefinition) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.input_schema,
    )


def _from_gemini_response(response: types.GenerateContentResponse) -> LLMResponse:
    if not response.candidates:
        raise LLMProviderError("Gemini API returned no candidates (the response may have been blocked).")
    candidate = response.candidates[0]

    if candidate.finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL:
        raise LLMProviderError("Gemini returned a malformed function call it could not complete.")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in (candidate.content.parts if candidate.content else None) or []:
        if part.function_call is not None:
            args = part.function_call.args if part.function_call.args is not None else {}
            if not isinstance(args, dict):
                raise LLMProviderError(
                    f"Gemini returned malformed function-call arguments for '{part.function_call.name}'."
                )
            call_id = part.function_call.id or f"call_{uuid.uuid4().hex[:8]}"
            # Capture this Part's thought_signature (opaque bytes, or None
            # if Gemini didn't attach one to this particular call) so it
            # can be replayed unchanged on this exact tool call -- see the
            # module docstring. Never decoded/inspected/logged in full.
            provider_metadata = (
                {"gemini_thought_signature": part.thought_signature} if part.thought_signature else None
            )
            tool_calls.append(ToolCall(
                id=call_id, name=part.function_call.name, arguments=args, provider_metadata=provider_metadata,
            ))
        elif part.text:
            text_parts.append(part.text)

    if tool_calls:
        # Presence/absence only -- never the signature value itself (see
        # ToolCall.provider_metadata's docstring in llm/base.py).
        signed = sum(1 for tc in tool_calls if tc.provider_metadata)
        logger.debug(
            "Gemini requested %d tool call(s), %d carrying a thought_signature.", len(tool_calls), signed,
        )
        return LLMResponse(
            message=ChatMessage(role="assistant", tool_calls=tool_calls), finish_reason="tool_calls", raw=response,
        )
    return LLMResponse(
        message=ChatMessage(role="assistant", content="".join(text_parts)), finish_reason="stop", raw=response,
    )


class GeminiProvider:
    """`LLMProvider` implementation calling the Gemini API via `google-genai`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and set GEMINI_API_KEY to a "
                "key from https://aistudio.google.com/apikey, or set it as an environment variable."
            )
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        # Kept only so error messages can be redacted before logging -- see
        # _log_provider_error/_redact below -- never logged itself.
        self._api_key = resolved_key
        # retry_options=HttpRetryOptions(attempts=1) disables the SDK's own
        # default retry behavior (which otherwise defaults to 5 attempts).
        # Application-level retry below is the *only* retry layer, which is
        # deliberate: this is the exact "nested retry" problem (SDK
        # retries stacking with application retries, requests pending for
        # minutes after a 429) already observed and fixed for Groq.
        http_options = types.HttpOptions(
            timeout=int(timeout_seconds * 1000),  # google-genai's HttpOptions.timeout is milliseconds
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        self._client = genai.Client(api_key=resolved_key, http_options=http_options)

    def _categorize(self, exc: errors.APIError) -> str:
        code = exc.code
        status = exc.status or ""
        if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
            return "authentication_failed"
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return "rate_limited"
        if isinstance(exc, errors.ServerError) or code in (500, 502, 503, 504):
            return "server_error"
        return "invalid_request"

    def _log_provider_error(self, exc: Exception, category: str, attempt: int, duration_ms: float) -> None:
        """Log a Gemini provider failure server-side with enough detail to
        diagnose it (exception type, HTTP status code, Gemini status
        string, the configured model, attempt count, latency, and a
        redacted message) -- without ever logging the API key. See
        llm/groq_provider.py's `_log_provider_error` for the identical
        rationale (this is the fix for /chat requests silently failing
        with nothing in server logs to say why)."""
        status_code = getattr(exc, "code", None)
        error_code = getattr(exc, "status", None)
        raw_message = getattr(exc, "message", None) or str(exc)
        message = _redact(raw_message, self._api_key)
        logger.error(
            "Gemini provider error (%s): exception_type=%s status_code=%s error_code=%s model=%s "
            "attempt=%d/%d duration_ms=%.0f message=%s",
            category, type(exc).__name__, status_code, error_code, self.model,
            attempt, self._max_retries + 1, duration_ms, message,
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        """Honor the API's own Retry-After header when it gives one
        (common on a 429); otherwise fall back to exponential backoff.
        `attempt` is 1-based (the attempt that just failed)."""
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
        system_instruction, contents = _to_gemini_contents(messages)
        gemini_tools = [types.Tool(function_declarations=[_to_gemini_tool(t) for t in tools])] if tools else None
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            tools=gemini_tools,
            # This project drives its own manual tool-calling loop
            # (agent/orchestrator.py) against MCP -- Gemini must never try
            # to auto-invoke a Python callable on our behalf.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        attempt = 0
        while True:
            attempt += 1
            attempt_start = time.perf_counter()
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except errors.APIError as exc:
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                category = self._categorize(exc)
                self._log_provider_error(exc, category, attempt, duration_ms)

                if category == "authentication_failed":
                    raise LLMProviderError(
                        "Gemini API rejected the configured GEMINI_API_KEY (authentication failed)."
                    ) from exc
                if category == "invalid_request":
                    # Never retried: retrying an unchanged malformed/too-large
                    # request fails identically every time.
                    raise LLMProviderError(f"Gemini API error: {_redact(exc.message, self._api_key)}") from exc

                # Only "rate_limited" and "server_error" reach here -- both
                # transient enough to be worth a small bounded retry.
                if attempt > self._max_retries:
                    if category == "rate_limited":
                        raise LLMProviderError("Gemini API rate limit or quota exceeded.") from exc
                    raise LLMProviderError(f"Gemini API error: {_redact(exc.message, self._api_key)}") from exc

                delay = self._retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Retrying Gemini request after a %s error (attempt %d/%d, waiting %.1fs)...",
                    category, attempt, self._max_retries + 1, delay,
                )
                await anyio.sleep(delay)
                continue
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                # A network-level failure with no HTTP response at all
                # (unlike errors.APIError above) -- equally transient.
                duration_ms = (time.perf_counter() - attempt_start) * 1000
                category = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection_error"
                self._log_provider_error(exc, category, attempt, duration_ms)

                if attempt > self._max_retries:
                    if category == "timeout":
                        raise LLMProviderError(f"Gemini API request timed out after {self._timeout_seconds}s.") from exc
                    raise LLMProviderError(f"Gemini API connection error: {exc}") from exc

                delay = self._retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Retrying Gemini request after a %s error (attempt %d/%d, waiting %.1fs)...",
                    category, attempt, self._max_retries + 1, delay,
                )
                await anyio.sleep(delay)
                continue

            return _from_gemini_response(response)
