"""Tests for the fix to the "every live /chat request returns the generic
apology, and Render only logs a 200 OK" bug: a Groq/provider exception was
being caught and turned into a graceful `AgentResponse` (correct
user-facing behavior) without ever being logged server-side (the actual
defect). These tests prove:

1. `GroqProvider.chat()` logs a sanitized error record (exception type,
   HTTP status code, API error code, model, attempt count, duration, and
   message) for every openai exception category it handles, without ever
   logging the API key. Tests in this section use `max_retries=0` (a
   single attempt) so they test the *logging format*, not retry behavior
   -- retry behavior has its own section below.
2. `_redact()` strips a secret/bearer token out of arbitrary text.
3. `AgentOrchestrator.run()` itself logs an ERROR when the LLM provider
   raises, independent of which provider implementation it's driving --
   so this can't regress even if `GroqProvider`'s own logging is ever
   changed.
4. The user-facing graceful failure message and `escalated`/trace
   behavior are unchanged by any of this.
5. Bounded retry/backoff applies only to transient failures (timeout,
   rate limit, connection error) and never to permanent ones
   (authentication, or the generic 4xx/5xx catch-all that covers things
   like an invalid model name or a context-length-exceeded request).

No real network call is made anywhere in this file -- the openai client's
`chat.completions.create` is monkeypatched to raise a real, pre-built
`openai` exception instance (or a scripted sequence of them), never
invoked over the network.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import anyio
import openai
import pytest

from agent.orchestrator import AgentOrchestrator
from llm.base import LLMProviderError
from llm.fake_provider import FakeLLMProvider
from llm.groq_provider import GroqProvider, _redact
from mcp_client.client import MCPClient

FAKE_API_KEY = "gsk_test_super_secret_value_do_not_log_me"


class _FakeHttpResponse:
    """Minimal duck-typed stand-in for `httpx2.Response`, carrying only
    the attributes `openai.APIStatusError.__init__` actually reads
    (`.status_code`, `.headers.get(...)`, `.request`) -- avoids this test
    module depending directly on `httpx2`, which requirements.txt notes is
    a transitive pin of the `openai` SDK, not something this project's own
    code imports."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.request = None


def _status_error(cls, status_code: int, message: str, code: str | None = None) -> openai.APIStatusError:
    body = {"message": message, **({"code": code} if code else {})}
    response = _FakeHttpResponse(status_code)
    return cls(f"Error code: {status_code} - {{'error': {body}}}", response=response, body=body)


# ---------------------------------------------------------------------------
# _redact
# ---------------------------------------------------------------------------
def test_redact_strips_the_configured_secret() -> None:
    text = f"request failed, key was {FAKE_API_KEY}, try again"
    redacted = _redact(text, FAKE_API_KEY)
    assert FAKE_API_KEY not in redacted
    assert "[REDACTED]" in redacted


def test_redact_strips_bearer_tokens_even_without_a_known_secret() -> None:
    text = "rejected: Authorization: Bearer abc123.def-456_ghi"
    redacted = _redact(text, None)
    assert "abc123.def-456_ghi" not in redacted
    assert "[REDACTED]" in redacted


def test_redact_handles_none_and_empty() -> None:
    assert _redact(None, FAKE_API_KEY) == ""
    assert _redact("", FAKE_API_KEY) == ""


# ---------------------------------------------------------------------------
# GroqProvider.chat() sanitized logging
# ---------------------------------------------------------------------------
def _provider(max_retries: int = 0, retry_base_delay_seconds: float = 0.0) -> GroqProvider:
    # max_retries=0 by default: these tests check the sanitized-logging
    # *format* for a single failed attempt, not retry behavior (see the
    # dedicated "Retry/backoff" section below for that).
    return GroqProvider(
        api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile",
        max_retries=max_retries, retry_base_delay_seconds=retry_base_delay_seconds,
    )


@pytest.mark.parametrize(
    ("build_exc", "expected_category"),
    [
        (lambda: _status_error(openai.AuthenticationError, 401, "Invalid API Key", code="invalid_api_key"), "authentication_failed"),
        (lambda: _status_error(openai.RateLimitError, 429, "Rate limit reached", code="rate_limit_exceeded"), "rate_limited"),
        (lambda: _status_error(openai.NotFoundError, 404, "The model does not exist", code="model_not_found"), "api_error"),
        (lambda: _status_error(openai.BadRequestError, 400, "model_decommissioned", code="model_decommissioned"), "api_error"),
    ],
)
def test_provider_errors_are_logged_with_sanitized_detail(caplog, build_exc, expected_category) -> None:
    provider = _provider()
    exc = build_exc()

    async def _raise(*args, **kwargs):
        raise exc

    provider._client.chat.completions.create = _raise  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert f"exception_type={type(exc).__name__}" in record.message
    assert f"status_code={exc.status_code}" in record.message
    assert f"error_code={exc.code}" in record.message
    assert "model=llama-3.3-70b-versatile" in record.message
    assert expected_category in record.message
    # The whole point of this bug fix: never log the key.
    assert FAKE_API_KEY not in caplog.text


def test_timeout_error_is_logged_without_status_code(caplog) -> None:
    provider = _provider()
    exc = openai.APITimeoutError(request=None)

    async def _raise(*args, **kwargs):
        raise exc

    provider._client.chat.completions.create = _raise  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert len(caplog.records) == 1
    assert "exception_type=APITimeoutError" in caplog.records[0].message
    assert "status_code=None" in caplog.records[0].message
    assert FAKE_API_KEY not in caplog.text


def test_provider_never_logs_the_api_key_even_if_it_appears_in_the_error_body(caplog) -> None:
    # A defense-in-depth check: even if a hypothetical error body echoed the
    # key back (it shouldn't -- Groq's error responses never do this), the
    # redaction pass must still strip it before it reaches the log.
    provider = _provider()
    exc = _status_error(openai.BadRequestError, 400, f"bad request, saw key {FAKE_API_KEY}")

    async def _raise(*args, **kwargs):
        raise exc

    provider._client.chat.completions.create = _raise  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert FAKE_API_KEY not in caplog.text
    assert "[REDACTED]" in caplog.text


# ---------------------------------------------------------------------------
# Retry/backoff: only for transient failures, bounded, never for permanent ones
# ---------------------------------------------------------------------------
def _fake_success_response(content: str) -> object:
    """A minimal duck-typed stand-in for the openai SDK's ChatCompletion
    response shape -- only what `_from_openai_message` actually reads
    (`choices[0].message.content`/`.tool_calls`, `choices[0].finish_reason`)."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def test_rate_limit_error_is_retried_and_eventually_succeeds(caplog) -> None:
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise _status_error(openai.RateLimitError, 429, "Rate limit reached", code="rate_limit_exceeded")
        return _fake_success_response("All good now.")

    provider._client.chat.completions.create = _flaky  # type: ignore[method-assign]

    async def body():
        return await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        result = anyio.run(body)

    assert result.message.content == "All good now."
    assert calls["count"] == 3  # 2 failed attempts + 1 successful retry
    assert len(caplog.records) == 2  # only the 2 failed attempts are logged as errors
    assert all("rate_limited" in r.message for r in caplog.records)


def test_rate_limit_error_exhausts_retries_and_raises(caplog) -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    exc = _status_error(openai.RateLimitError, 429, "Rate limit reached", code="rate_limit_exceeded")

    async def _always_fails(*args, **kwargs):
        raise exc

    provider._client.chat.completions.create = _always_fails  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="rate limit"):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert len(caplog.records) == 2  # max_retries=1 -> 2 total attempts, both logged
    assert "attempt=1/2" in caplog.records[0].message
    assert "attempt=2/2" in caplog.records[1].message


def test_connection_error_is_retried(caplog) -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise openai.APIConnectionError(request=None)
        return _fake_success_response("Recovered.")

    provider._client.chat.completions.create = _flaky  # type: ignore[method-assign]

    async def body():
        return await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        result = anyio.run(body)

    assert result.message.content == "Recovered."
    assert calls["count"] == 2
    assert len(caplog.records) == 1
    assert "connection_error" in caplog.records[0].message


def test_authentication_error_is_never_retried(caplog) -> None:
    # Even with retries configured, a bad key is bad on every attempt --
    # retrying it wastes time and produces identical failures.
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _always_fails(*args, **kwargs):
        calls["count"] += 1
        raise _status_error(openai.AuthenticationError, 401, "Invalid API Key", code="invalid_api_key")

    provider._client.chat.completions.create = _always_fails  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="authentication failed"):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert calls["count"] == 1  # no retry attempted
    assert len(caplog.records) == 1
    assert "attempt=1/3" in caplog.records[0].message


def test_context_length_exceeded_style_error_is_never_retried(caplog) -> None:
    # A permanent 4xx (invalid model, or a request too large for the
    # model's context window) surfaces as a generic APIError/BadRequestError
    # -- retrying the same oversized request would fail identically.
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _always_fails(*args, **kwargs):
        calls["count"] += 1
        raise _status_error(openai.BadRequestError, 400, "This model's maximum context length is 8192 tokens.",
                             code="context_length_exceeded")

    provider._client.chat.completions.create = _always_fails  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.groq"):
        anyio.run(body)

    assert calls["count"] == 1  # no retry attempted
    assert len(caplog.records) == 1
    assert "context_length_exceeded" in caplog.records[0].message


# ---------------------------------------------------------------------------
# Orchestrator-level logging (provider-agnostic backstop)
# ---------------------------------------------------------------------------
def test_orchestrator_logs_llm_provider_errors_and_keeps_the_graceful_response(
    caplog, mcp_server_env: dict[str, str]
) -> None:
    fake = FakeLLMProvider([])  # raises LLMProviderError on the first chat() call

    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            orchestrator = AgentOrchestrator(fake, client)
            with caplog.at_level(logging.ERROR, logger="hr_compass_ai.agent.orchestrator"):
                return await orchestrator.run("hello")

    result = anyio.run(body)

    # User-facing behavior unchanged: still the generic, graceful message.
    assert result.answer == "I'm having trouble reaching the language model right now. Please try again shortly."
    assert result.escalated is True

    # The client-facing operational trace (what the "Agent Activity" panel
    # renders) must be sanitized -- never the raw provider/API exception
    # text -- even though the full detail is available server-side.
    trace_text = json.dumps(result.trace)
    assert "ran out of scripted responses" not in trace_text
    decision_steps = [s for s in result.trace if s["type"] == "decision"]
    assert any(s["escalation"] == "llm_unavailable" for s in decision_steps)

    # But the full detail is actually visible server-side, in logs.
    assert any("LLM provider error during agent turn" in r.message for r in caplog.records)
    assert any("ran out of scripted responses" in r.message for r in caplog.records)
