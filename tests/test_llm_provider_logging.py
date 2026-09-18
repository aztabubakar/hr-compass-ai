"""Tests for the fix to the "every live /chat request returns the generic
apology, and Render only logs a 200 OK" bug: a Groq/provider exception was
being caught and turned into a graceful `AgentResponse` (correct
user-facing behavior) without ever being logged server-side (the actual
defect). These tests prove:

1. `GroqProvider.chat()` logs a sanitized error record (exception type,
   HTTP status code, API error code, model, message) for every openai
   exception category it handles, without ever logging the API key.
2. `_redact()` strips a secret/bearer token out of arbitrary text.
3. `AgentOrchestrator.run()` itself logs an ERROR when the LLM provider
   raises, independent of which provider implementation it's driving --
   so this can't regress even if `GroqProvider`'s own logging is ever
   changed.
4. The user-facing graceful failure message and `escalated`/trace
   behavior are unchanged by any of this.

No real network call is made anywhere in this file -- the openai client's
`chat.completions.create` is monkeypatched to raise a real, pre-built
`openai` exception instance, never invoked over the network.
"""
from __future__ import annotations

import logging

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
def _provider() -> GroqProvider:
    return GroqProvider(api_key=FAKE_API_KEY, model="llama-3.3-70b-versatile")


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

    # But now it's actually visible server-side.
    assert any("LLM provider error during agent turn" in r.message for r in caplog.records)
