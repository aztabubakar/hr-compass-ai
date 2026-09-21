"""Tests for `GeminiProvider` (llm/gemini_provider.py), the Gemini-backed
`LLMProvider` implementation added when migrating the production LLM from
Groq to Google Gemini.

No real network call is made anywhere in this file -- `google.genai`'s
`Client.aio.models.generate_content` is monkeypatched to return a real,
hand-built `types.GenerateContentResponse` (or raise a real
`google.genai.errors.APIError`/`httpx` exception), the same pattern
tests/test_llm_provider_logging.py already uses for `openai` exceptions
against `GroqProvider`. This also proves `GeminiProvider` never actually
calls a live API from CI.

Covers (see the migration task's 18 required scenarios):
  1.  normal text completion
  2.  system instruction handling
  3.  Gemini requests one MCP tool
  4.  structured function arguments parsed correctly
  5.  multiple tool calls in one turn
  6.  MCP tool result sent back to Gemini
  7.  final response after a tool call (multi-turn continuation)
  8.  malformed tool arguments / MALFORMED_FUNCTION_CALL
  9.  Gemini authentication/configuration failure
  10. rate-limit/quota failure
  11. transient (server) error + bounded retry
  12. timeout
  13. provider selection via LLM_PROVIDER
  14. /health reports Gemini correctly
  15. API key never appears in logs
  16. create_mock_hr_ticket confirmation gate still works with Gemini
  17. existing Groq tests still pass -- verified by running the full suite,
      not re-tested here (see tests/test_llm_provider_logging.py).
  18. existing FakeLLMProvider tests still pass -- likewise (see
      tests/test_agent_orchestrator.py, tests/test_evaluation.py).
"""
from __future__ import annotations

import logging

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

import app.main as app_main
from agent.orchestrator import AgentOrchestrator
from app.main import app, get_llm_provider
from llm.base import ChatMessage, LLMConfigError, LLMProviderError, ToolCall, ToolDefinition
from llm.config import LLMConfig, get_provider
from llm.gemini_provider import GeminiProvider
from llm.groq_provider import GroqProvider
from mcp_client.client import MCPClient

FAKE_API_KEY = "AIzaSy_test_super_secret_value_do_not_log_me"


def _provider(max_retries: int = 0, retry_base_delay_seconds: float = 0.0) -> GeminiProvider:
    return GeminiProvider(
        api_key=FAKE_API_KEY, model="gemini-2.5-flash-lite",
        max_retries=max_retries, retry_base_delay_seconds=retry_base_delay_seconds,
    )


def _text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        finish_reason=types.FinishReason.STOP,
    )])


def _tool_call_response(*calls: tuple[str, dict]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls
        ]),
        finish_reason=types.FinishReason.STOP,
    )])


def _client_error(code: int, status: str, message: str, retry_after: str | None = None) -> errors.ClientError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(status_code=code, headers=headers, request=httpx.Request("POST", "https://example.com"))
    return errors.ClientError(code=code, response_json={"error": {"status": status, "message": message}}, response=response)


def _server_error(code: int, message: str) -> errors.ServerError:
    response = httpx.Response(status_code=code, request=httpx.Request("POST", "https://example.com"))
    return errors.ServerError(code=code, response_json={"error": {"status": "UNAVAILABLE", "message": message}}, response=response)


# ---------------------------------------------------------------------------
# 1. Normal text completion
# ---------------------------------------------------------------------------
def test_normal_text_completion() -> None:
    provider = _provider()

    async def _generate(*args, **kwargs):
        return _text_response("Solara observes 10 fixed holidays, per POL-HOL-001.")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([ChatMessage(role="user", content="What is the holiday policy?")])

    result = anyio.run(body)
    assert result.finish_reason == "stop"
    assert result.message.content == "Solara observes 10 fixed holidays, per POL-HOL-001."
    assert result.message.tool_calls is None


# ---------------------------------------------------------------------------
# 2. System instruction handling
# ---------------------------------------------------------------------------
def test_system_instruction_is_passed_separately_not_as_a_content_entry() -> None:
    provider = _provider()
    captured = {}

    async def _generate(*, model, contents, config):
        captured["contents"] = contents
        captured["system_instruction"] = config.system_instruction
        return _text_response("ok")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([
            ChatMessage(role="system", content="You are HR Compass AI."),
            ChatMessage(role="user", content="hello"),
        ])

    anyio.run(body)
    assert captured["system_instruction"] == "You are HR Compass AI."
    # The system message must not appear in `contents` at all -- Gemini has
    # no "system" role in that list.
    assert all(c.role != "system" for c in captured["contents"])
    assert len(captured["contents"]) == 1
    assert captured["contents"][0].role == "user"


# ---------------------------------------------------------------------------
# 3. Gemini requests one MCP tool
# ---------------------------------------------------------------------------
def test_single_tool_call_is_requested() -> None:
    provider = _provider()

    async def _generate(*args, **kwargs):
        return _tool_call_response(("check_pto_balance", {"employee_id": "EMP-1001"}))

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat(
            [ChatMessage(role="user", content="How much PTO do I have? EMP-1001")],
            tools=[ToolDefinition(name="check_pto_balance", description="d", input_schema={"type": "object"})],
        )

    result = anyio.run(body)
    assert result.finish_reason == "tool_calls"
    assert len(result.message.tool_calls) == 1
    call = result.message.tool_calls[0]
    assert call.name == "check_pto_balance"
    assert call.arguments == {"employee_id": "EMP-1001"}
    assert call.id  # a synthetic id is assigned when Gemini doesn't supply one


# ---------------------------------------------------------------------------
# 4. Structured function arguments parsed correctly
# ---------------------------------------------------------------------------
def test_structured_nested_arguments_are_preserved() -> None:
    provider = _provider()
    nested_args = {"employee_id": "EMP-1002", "scenario_type": "remote_work", "requested_days": 30, "notice_business_days": None}

    async def _generate(*args, **kwargs):
        return _tool_call_response(("check_policy_compliance", nested_args))

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([ChatMessage(role="user", content="six weeks remote from Colorado")])

    result = anyio.run(body)
    assert result.message.tool_calls[0].arguments == nested_args


# ---------------------------------------------------------------------------
# 5. Multiple tool calls in one turn
# ---------------------------------------------------------------------------
def test_multiple_tool_calls_in_one_turn() -> None:
    provider = _provider()

    async def _generate(*args, **kwargs):
        return _tool_call_response(
            ("lookup_employee_profile", {"employee_id": "EMP-1001"}),
            ("check_pto_balance", {"employee_id": "EMP-1001"}),
        )

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([ChatMessage(role="user", content="EMP-1001, 3 days PTO next week")])

    result = anyio.run(body)
    assert [c.name for c in result.message.tool_calls] == ["lookup_employee_profile", "check_pto_balance"]
    # Each call gets its own id, even though Gemini didn't supply one.
    assert len({c.id for c in result.message.tool_calls}) == 2


# ---------------------------------------------------------------------------
# 6. MCP tool result sent back to Gemini
# ---------------------------------------------------------------------------
def test_tool_result_is_sent_back_as_a_function_response_part() -> None:
    provider = _provider()
    captured = {}

    async def _generate(*, model, contents, config):
        captured["contents"] = contents
        return _text_response("You have 48 hours available.")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([
            ChatMessage(role="user", content="How much PTO? EMP-1001"),
            ChatMessage(role="assistant", tool_calls=[ToolCall(id="call_1", name="check_pto_balance", arguments={"employee_id": "EMP-1001"})]),
            ChatMessage(role="tool", tool_call_id="call_1", name="check_pto_balance", content='{"balance_hours": 48}'),
        ])

    anyio.run(body)
    tool_content = captured["contents"][-1]
    assert tool_content.role == "user"  # Gemini has no dedicated "tool" role
    part = tool_content.parts[0]
    assert part.function_response.name == "check_pto_balance"
    assert part.function_response.response == {"balance_hours": 48}


def test_consecutive_tool_results_are_merged_into_one_turn() -> None:
    # Two tool results from the same orchestrator turn (a multi-tool-call
    # round) must be sent back as ONE user-role Content with two
    # function_response parts, not two separate Content entries -- this is
    # what a real multi-call round from scenario 5 needs on its way back.
    provider = _provider()
    captured = {}

    async def _generate(*, model, contents, config):
        captured["contents"] = contents
        return _text_response("done")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([
            ChatMessage(role="user", content="EMP-1001, PTO?"),
            ChatMessage(role="assistant", tool_calls=[
                ToolCall(id="c1", name="lookup_employee_profile", arguments={"employee_id": "EMP-1001"}),
                ToolCall(id="c2", name="check_pto_balance", arguments={"employee_id": "EMP-1001"}),
            ]),
            ChatMessage(role="tool", tool_call_id="c1", name="lookup_employee_profile", content='{"full_name": "Jordan Vance"}'),
            ChatMessage(role="tool", tool_call_id="c2", name="check_pto_balance", content='{"balance_hours": 48}'),
        ])

    anyio.run(body)
    tool_turn = captured["contents"][-1]
    assert tool_turn.role == "user"
    assert len(tool_turn.parts) == 2
    assert {p.function_response.name for p in tool_turn.parts} == {"lookup_employee_profile", "check_pto_balance"}


# ---------------------------------------------------------------------------
# 7. Final response after a tool call (two real chat() calls in sequence)
# ---------------------------------------------------------------------------
def test_final_response_after_a_tool_call_round_trip() -> None:
    provider = _provider()
    responses = [
        _tool_call_response(("check_pto_balance", {"employee_id": "EMP-1001"})),
        _text_response("You have 48 hours of PTO available."),
    ]

    async def _generate(*args, **kwargs):
        return responses.pop(0)

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        first = await provider.chat(
            [ChatMessage(role="user", content="EMP-1001, how much PTO?")],
            tools=[ToolDefinition(name="check_pto_balance", description="d", input_schema={"type": "object"})],
        )
        call = first.message.tool_calls[0]
        history = [
            ChatMessage(role="user", content="EMP-1001, how much PTO?"),
            first.message,
            ChatMessage(role="tool", tool_call_id=call.id, name=call.name, content='{"balance_hours": 48}'),
        ]
        return await provider.chat(history)

    result = anyio.run(body)
    assert result.finish_reason == "stop"
    assert result.message.content == "You have 48 hours of PTO available."


# ---------------------------------------------------------------------------
# 8. Malformed tool arguments
# ---------------------------------------------------------------------------
def test_malformed_function_call_raises_a_clean_error() -> None:
    provider = _provider()

    async def _generate(*args, **kwargs):
        return types.GenerateContentResponse(candidates=[types.Candidate(
            content=None, finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
        )])

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="malformed function call"):
            await provider.chat([ChatMessage(role="user", content="hello")])

    anyio.run(body)


def test_non_dict_function_call_args_raise_a_clean_error() -> None:
    provider = _provider()

    async def _generate(*args, **kwargs):
        response = types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="check_pto_balance"))]),
            finish_reason=types.FinishReason.STOP,
        )])
        # Force a non-dict `args` value the way malformed upstream JSON might
        # -- FunctionCall.args is normally always a dict when set at all.
        response.candidates[0].content.parts[0].function_call.args = "not-a-dict"
        return response

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="malformed function-call arguments"):
            await provider.chat([ChatMessage(role="user", content="hello")])

    anyio.run(body)


# ---------------------------------------------------------------------------
# 9. Authentication/configuration failure
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_config_error_at_construction() -> None:
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key=None)


@pytest.mark.parametrize(
    ("code", "status"),
    [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")],
)
def test_authentication_failure_is_never_retried(caplog, code, status) -> None:
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        raise _client_error(code, status, "API key not valid")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="authentication failed"):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        anyio.run(body)

    assert calls["count"] == 1  # no retry attempted
    assert len(caplog.records) == 1
    assert "authentication_failed" in caplog.records[0].message


# ---------------------------------------------------------------------------
# 10. Rate-limit/quota failure
# ---------------------------------------------------------------------------
def test_rate_limit_is_retried_and_eventually_succeeds(caplog) -> None:
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise _client_error(429, "RESOURCE_EXHAUSTED", "quota exceeded")
        return _text_response("All good now.")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        result = anyio.run(body)

    assert result.message.content == "All good now."
    assert calls["count"] == 3
    assert all("rate_limited" in r.message for r in caplog.records)


def test_rate_limit_exhausts_bounded_retries_and_fails_gracefully(caplog) -> None:
    # Never endlessly retries quota exhaustion -- the retry budget is small
    # and bounded, exactly like GroqProvider's.
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        raise _client_error(429, "RESOURCE_EXHAUSTED", "quota exceeded")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="rate limit or quota"):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        anyio.run(body)

    assert calls["count"] == 2  # max_retries=1 -> 2 total attempts
    assert len(caplog.records) == 2


def test_rate_limit_honors_retry_after_header() -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=99.0)
    exc = _client_error(429, "RESOURCE_EXHAUSTED", "quota exceeded", retry_after="0.01")
    delay = provider._retry_delay_seconds(exc, attempt=1)
    assert delay == 0.01  # honors Retry-After instead of the (much larger) backoff default


def test_invalid_argument_is_never_retried(caplog) -> None:
    # A permanent 4xx (e.g. a request too large for the model's context
    # window) -- retrying an unchanged request fails identically every time.
    provider = _provider(max_retries=2, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        raise _client_error(400, "INVALID_ARGUMENT", "request too large")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        anyio.run(body)

    assert calls["count"] == 1


# ---------------------------------------------------------------------------
# 11. Transient (server) error + bounded retry
# ---------------------------------------------------------------------------
def test_server_error_is_retried_and_recovers(caplog) -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _server_error(503, "model overloaded")
        return _text_response("Recovered.")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        result = anyio.run(body)

    assert result.message.content == "Recovered."
    assert calls["count"] == 2
    assert "server_error" in caplog.records[0].message


# ---------------------------------------------------------------------------
# 12. Timeout
# ---------------------------------------------------------------------------
def test_network_timeout_is_retried_then_raises_a_clean_error(caplog) -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        raise httpx.TimeoutException("timed out")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError, match="timed out"):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        anyio.run(body)

    assert calls["count"] == 2  # max_retries=1 -> 2 total attempts
    assert all("timeout" in r.message for r in caplog.records)


def test_connection_error_is_retried_and_recovers() -> None:
    provider = _provider(max_retries=1, retry_base_delay_seconds=0.01)
    calls = {"count": 0}

    async def _generate(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection refused")
        return _text_response("Recovered.")

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        return await provider.chat([])

    result = anyio.run(body)
    assert result.message.content == "Recovered."
    assert calls["count"] == 2


# ---------------------------------------------------------------------------
# 13. Provider selection via LLM_PROVIDER
# ---------------------------------------------------------------------------
def test_llm_provider_env_selects_gemini(monkeypatch) -> None:
    config = LLMConfig(provider="gemini", gemini_api_key=FAKE_API_KEY, gemini_model="gemini-2.5-flash-lite")
    provider = get_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash-lite"


def test_llm_provider_env_still_selects_groq() -> None:
    config = LLMConfig(provider="groq", groq_api_key="gsk_fake")
    provider = get_provider(config)
    assert isinstance(provider, GroqProvider)


def test_gemini_selected_without_key_fails_clearly_not_silently() -> None:
    config = LLMConfig(provider="gemini", gemini_api_key=None)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        get_provider(config)


def test_fake_llm_provider_still_directly_constructible() -> None:
    # Item 4's explicit requirement: tests must still be able to construct
    # FakeLLMProvider directly regardless of what LLM_PROVIDER is set to.
    from llm.fake_provider import FakeLLMProvider

    fake = FakeLLMProvider([])
    assert fake.model == "fake-model"


def test_is_llm_configured_for_gemini() -> None:
    from llm.config import is_llm_configured

    assert is_llm_configured(LLMConfig(provider="gemini", gemini_api_key=FAKE_API_KEY)) is True
    assert is_llm_configured(LLMConfig(provider="gemini", gemini_api_key=None)) is False


# ---------------------------------------------------------------------------
# 14. /health reports Gemini correctly
# ---------------------------------------------------------------------------
def test_health_reports_gemini_when_selected(real_corpus_hashing_index_dir, monkeypatch) -> None:
    # LLMConfig's fields default from `os.getenv(...)` evaluated once at
    # class-definition time (same characteristic as the rest of this
    # project's env-var-driven config), so setting LLM_PROVIDER via
    # monkeypatch after import wouldn't be observed by a plain `LLMConfig()`
    # call -- in real deployment this is a non-issue (Render sets env vars
    # before the process, and therefore this module, ever starts). Here we
    # monkeypatch the already-imported names `/health` actually calls, which
    # exercises the exact same route logic without relying on reimporting.
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("RAG_INDEX_DIR", str(real_corpus_hashing_index_dir))
    monkeypatch.setattr(app_main, "LLMConfig", lambda: LLMConfig(provider="gemini", gemini_api_key=FAKE_API_KEY))
    monkeypatch.setattr(app_main, "is_llm_configured", lambda: True)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["llm"] == {"configured": True, "provider": "gemini"}
    assert FAKE_API_KEY not in str(data)
    assert "gemini_api_key" not in str(data).lower()


def test_health_reports_gemini_not_configured_without_key(real_corpus_hashing_index_dir, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("RAG_INDEX_DIR", str(real_corpus_hashing_index_dir))
    monkeypatch.setattr(app_main, "LLMConfig", lambda: LLMConfig(provider="gemini", gemini_api_key=None))
    monkeypatch.setattr(app_main, "is_llm_configured", lambda: False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.json()["llm"] == {"configured": False, "provider": "gemini"}


# ---------------------------------------------------------------------------
# 15. API key never appears in logs
# ---------------------------------------------------------------------------
def test_api_key_never_appears_in_logs_on_any_failure_path(caplog) -> None:
    provider = _provider()
    exc = _client_error(400, "INVALID_ARGUMENT", f"bad request, saw key {FAKE_API_KEY}")

    async def _generate(*args, **kwargs):
        raise exc

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        with pytest.raises(LLMProviderError):
            await provider.chat([])

    with caplog.at_level(logging.ERROR, logger="hr_compass_ai.llm.gemini"):
        anyio.run(body)

    assert FAKE_API_KEY not in caplog.text
    assert "[REDACTED]" in caplog.text


# ---------------------------------------------------------------------------
# 16. create_mock_hr_ticket confirmation gate still works with Gemini
# ---------------------------------------------------------------------------
def test_confirmation_gate_downgrades_a_gemini_driven_ticket_creation(mcp_server_env: dict[str, str]) -> None:
    # The code-level gate in agent/orchestrator.py operates on the tool
    # call's name/arguments regardless of which provider produced it -- this
    # proves a Gemini-sourced tool call can't bypass it either.
    provider = _provider()
    responses = [
        _tool_call_response(("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "remote_work", "summary": "Remote work request", "confirmed": True,
        })),
        _text_response("Done."),
    ]

    async def _generate(*args, **kwargs):
        return responses.pop(0)

    provider._client.aio.models.generate_content = _generate  # type: ignore[method-assign]

    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            orchestrator = AgentOrchestrator(provider, client)
            return await orchestrator.run(
                "Create an HR ticket for my remote-work request.",
                prior_turn_requires_confirmation=False,  # no prior CONFIRM turn
            )

    result = anyio.run(body)
    ticket_step = next(s for s in result.trace if s.get("tool") == "create_mock_hr_ticket")
    assert ticket_step["arguments"]["confirmed"] is False  # downgraded, exactly like the Groq path
