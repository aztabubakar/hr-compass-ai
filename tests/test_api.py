"""FastAPI endpoint tests. The real MCP server is used (via the app's own
lifespan, pointed at a hermetic hashing-backend index through
`mcp_server_env`-equivalent env vars set for the test process itself);
the LLM is always `FakeLLMProvider`, injected via FastAPI's dependency
override — no network call, no GROQ_API_KEY needed.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_provider
from llm.fake_provider import FakeLLMProvider, final_turn, tool_call_turn


@pytest.fixture
def api_client(real_corpus_hashing_index_dir, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("RAG_INDEX_DIR", str(real_corpus_hashing_index_dir))
    # raise_server_exceptions=False makes the test client behave like a real
    # deployed server: an exception that escapes every handler comes back
    # as an actual 500 HTTP response (via Starlette's ServerErrorMiddleware)
    # instead of being re-raised into the test process for debugging, which
    # is TestClient's default and not representative of production behavior.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.pop(get_llm_provider, None)


def _use_fake_llm(scripted_responses) -> FakeLLMProvider:
    fake = FakeLLMProvider(scripted_responses)
    app.dependency_overrides[get_llm_provider] = lambda: fake
    return fake


def test_health_returns_expected_shape(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mcp"] == {"ready": True, "tools_discovered": 8}
    assert data["rag"] == {"ready": True}
    assert data["llm"]["configured"] is False  # no GROQ_API_KEY in the test environment
    assert data["llm"]["provider"] == "groq"
    # Never expose secrets.
    assert "api_key" not in str(data).lower()
    assert "groq_api_key" not in str(data).lower()


def test_health_reports_degraded_when_mcp_unavailable(api_client: TestClient) -> None:
    # Simulate the same "MCP unreachable" state the lifespan sets when the
    # server subprocess can't be reached at startup (see app/main.py's
    # lifespan()), without actually breaking the process launch.
    app.state.mcp_client = None
    try:
        response = api_client.get("/health")
    finally:
        pass  # the api_client fixture tears the whole app down after this test anyway
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["mcp"]["ready"] is False
    assert data["mcp"]["tools_discovered"] is None


def test_api_tools_lists_all_required_tools(api_client: TestClient) -> None:
    response = api_client.get("/api/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()["tools"]}
    assert names == {
        "search_policy_documents", "get_policy_section", "lookup_employee_profile",
        "check_pto_balance", "lookup_benefits_status", "check_policy_compliance",
        "draft_hr_email", "create_mock_hr_ticket",
    }
    # Schemas are present and non-trivial.
    for tool in response.json()["tools"]:
        assert tool["input_schema"]["type"] == "object"


def test_chat_request_validation_rejects_empty_message(api_client: TestClient) -> None:
    # Provide a working LLM override so the body-validation failure (the
    # thing under test) isn't masked by an unrelated 503 for "LLM not
    # configured" -- both would fire on an unconfigured environment, and
    # FastAPI resolves the Depends() before body validation surfaces.
    _use_fake_llm([final_turn("unused")])
    response = api_client.post("/chat", json={"message": ""})
    assert response.status_code == 422
    # A structured validation error, not a stack trace.
    assert "Traceback" not in response.text


def test_chat_without_llm_configured_returns_clean_503(api_client: TestClient) -> None:
    app.dependency_overrides.pop(get_llm_provider, None)  # ensure no override is active
    response = api_client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert response.json()["detail"]


def test_chat_happy_path_returns_answer_citations_and_trace(api_client: TestClient) -> None:
    _use_fake_llm([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        final_turn("You have 48 hours of PTO available."),
    ])
    response = api_client.post("/chat", json={
        "message": "My employee ID is EMP-1001. How much PTO do I have?",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "You have 48 hours of PTO available."
    assert isinstance(data["trace"], list)
    # 2 tool_call steps (one per scripted tool call) plus one "decision"
    # step per LLM turn (agent/orchestrator.py's per-call latency
    # instrumentation) -- assert on the meaningful subset rather than the
    # exact total step count, which is an implementation detail of how
    # many LLM turns this specific script happens to take.
    tool_call_steps = [s for s in data["trace"] if s["type"] == "tool_call"]
    assert [s["tool"] for s in tool_call_steps] == ["lookup_employee_profile", "check_pto_balance"]
    assert all(s["duration_ms"] is not None for s in data["trace"])
    assert data["conversation_id"]
    assert data["requires_clarification"] is False
    assert data["requires_confirmation"] is False


def test_chat_returns_citations_for_policy_question(api_client: TestClient) -> None:
    _use_fake_llm([
        tool_call_turn("search_policy_documents", {"query": "holiday schedule", "top_k": 3}),
        final_turn("Solara observes 10 fixed holidays plus 2 floating holidays, per POL-HOL-001."),
    ])
    response = api_client.post("/chat", json={"message": "What is Solara's holiday policy?"})
    assert response.status_code == 200
    data = response.json()
    assert data["citations"]
    assert all({"document_id", "document_title", "section", "snippet"} <= c.keys() for c in data["citations"])
    assert data["snippets"]


def test_chat_conversation_continues_across_requests(api_client: TestClient) -> None:
    _use_fake_llm([final_turn("CLARIFY: What is your employee ID?")])
    first = api_client.post("/chat", json={"message": "How much PTO do I have?"})
    conversation_id = first.json()["conversation_id"]
    assert first.json()["requires_clarification"] is True

    _use_fake_llm([
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        final_turn("You have 48 hours of PTO available."),
    ])
    second = api_client.post("/chat", json={"message": "EMP-1001", "conversation_id": conversation_id})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id


def test_chat_clarification_and_confirmation_flags_surface(api_client: TestClient) -> None:
    _use_fake_llm([final_turn("CONFIRM: I can create a mock HR case for this. Proceed?")])
    response = api_client.post("/chat", json={"message": "I want to report an issue."})
    data = response.json()
    assert data["requires_confirmation"] is True
    assert data["requires_clarification"] is False


def test_llm_provider_failure_is_handled_gracefully_not_as_a_crash(api_client: TestClient) -> None:
    # AgentOrchestrator.run() catches LLMProviderError itself (an empty
    # scripted-response list makes FakeLLMProvider raise one) and returns a
    # normal, graceful AgentResponse rather than propagating the exception
    # — so this is a 200 with an apologetic answer, not a 5xx, and in
    # either case never a raw traceback.
    fake = FakeLLMProvider([])
    app.dependency_overrides[get_llm_provider] = lambda: fake
    response = api_client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert "Traceback" not in response.text
    data = response.json()
    assert data["escalated"] is True
    assert data["answer"]


def test_genuinely_unexpected_error_is_caught_by_global_handler(api_client: TestClient) -> None:
    def _boom():
        raise RuntimeError("something truly unexpected")

    app.dependency_overrides[get_llm_provider] = _boom
    response = api_client.post("/chat", json={"message": "hello"})
    assert response.status_code == 500
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text
    assert response.json()["detail"] == "An unexpected error occurred. Please try again."


def test_root_serves_chat_ui_with_required_elements(api_client: TestClient) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "HR Compass AI" in html
    assert 'id="message-input"' in html
    assert 'id="messages"' in html
    assert 'id="send-button"' in html
