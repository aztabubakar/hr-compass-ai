"""Tests for the deployed-endpoint ("HTTP mode") evaluation harness added
to `evaluation/evaluator.py` (`run_case_http`) and `evaluation/run_live.py`
(`EVAL_BASE_URL`, `--delay-between-cases`, `--resume`/checkpointing).

Like the rest of the evaluation test suite (`tests/test_evaluation.py`),
these are offline and deterministic: every `/chat`/`/health` call is
answered by an `httpx.MockTransport` handler defined in this file --
nothing here makes a real network call to Render or Gemini.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest

import evaluation.run_live as run_live
from evaluation.cases import EvalCase, load_cases
from evaluation.evaluator import EvalResult, run_case_http
from evaluation.metrics import score_case

CASES_BY_ID = {c.case_id: c for c in load_cases()}


def _default_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "answer": "Here is your grounded answer.",
        "citations": [
            {"document_id": "POL-PTO-001", "document_title": "PTO Policy", "section": "Accrual",
             "snippet": "Employees accrue 10 hours per month.", "source_path": "policies/pol-pto-001.md"},
        ],
        "snippets": ["Employees accrue 10 hours per month."],
        "trace": [
            {"step": 1, "timestamp": "2026-01-01T00:00:00+00:00", "type": "decision", "tool": None,
             "arguments": None, "status": "success", "result_summary": "Model turn: requested 1 tool call(s).",
             "sources": [], "escalation": None, "duration_ms": 12.0},
            {"step": 2, "timestamp": "2026-01-01T00:00:01+00:00", "type": "tool_call",
             "tool": "search_policy_documents", "arguments": {"query": "pto accrual"}, "status": "success",
             "result_summary": "1 result", "sources": ["POL-PTO-001 §Accrual"], "escalation": None,
             "duration_ms": 8.0},
            {"step": 3, "timestamp": "2026-01-01T00:00:02+00:00", "type": "decision", "tool": None,
             "arguments": None, "status": "success", "result_summary": "Model turn: produced a final answer.",
             "sources": [], "escalation": None, "duration_ms": 10.0},
        ],
        "requires_clarification": False,
        "requires_confirmation": False,
        "escalated": False,
        "conversation_id": "whatever",
    }
    payload.update(overrides)
    return payload


class RecordingChatMock:
    """A minimal `httpx.MockTransport` handler standing in for a deployed
    HR Compass AI instance: answers `/health` with a fixed, provider-only
    payload (mirroring `app/models.py`'s `HealthResponse`) and `/chat` with
    either a fixed payload, a per-call sequence of payloads, an HTTP error
    status, or a raised transport exception -- whichever this instance was
    configured with. Records every `/chat` request body it receives, in
    order, so tests can assert on `conversation_id`/`message` sent per call.
    """

    def __init__(
        self,
        chat_payloads: list[dict[str, Any]] | None = None,
        chat_status: int = 200,
        malformed_json: bool = False,
        raise_exc: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._chat_payloads = chat_payloads
        self._chat_status = chat_status
        self._malformed_json = malformed_json
        self._raise_exc = raise_exc
        self._sleep_seconds = sleep_seconds

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "rag": {"ready": True},
                                              "mcp": {"ready": True, "tools_discovered": 8},
                                              "llm": {"configured": True, "provider": "gemini"}})
        assert request.url.path == "/chat"
        body = json.loads(request.content)
        self.requests.append(body)

        if self._sleep_seconds:
            import time
            time.sleep(self._sleep_seconds)

        if self._raise_exc is not None:
            raise self._raise_exc
        if self._malformed_json:
            return httpx.Response(200, content=b"{not valid json")
        if self._chat_status != 200:
            return httpx.Response(self._chat_status, json={"detail": "server error"})

        if self._chat_payloads is not None:
            idx = len(self.requests) - 1
            payload = self._chat_payloads[min(idx, len(self._chat_payloads) - 1)]
        else:
            payload = _default_payload()
        return httpx.Response(200, json=payload)


def _client_for(mock: RecordingChatMock) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(mock.handler))


def _simple_case(**overrides: Any) -> EvalCase:
    base = dict(case_id="EVAL-TEST", category="A_policy_simple", user_input="What is the PTO policy?",
                expected_behavior="test")
    base.update(overrides)
    return EvalCase.model_validate(base)


# ---------------------------------------------------------------------------
# run_case_http() unit tests (items 3-10, 17)
# ---------------------------------------------------------------------------
def test_successful_response_maps_into_eval_result() -> None:
    mock = RecordingChatMock()
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.error is None
    assert result.case_id == "EVAL-TEST"
    assert result.category == "A_policy_simple"
    assert result.user_input == case.user_input
    assert result.answer == "Here is your grounded answer."


def test_tools_called_derived_from_http_trace() -> None:
    mock = RecordingChatMock()
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.tools_called == ["search_policy_documents"]


def test_citations_survive_mapping_unchanged() -> None:
    mock = RecordingChatMock()
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.citations == _default_payload()["citations"]


def test_clarification_escalation_confirmation_flags_survive() -> None:
    mock = RecordingChatMock(chat_payloads=[_default_payload(
        requires_clarification=True, requires_confirmation=True, escalated=True,
    )])
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.requires_clarification is True
    assert result.requires_confirmation is True
    assert result.escalated is True


def test_http_wall_clock_latency_is_recorded() -> None:
    mock = RecordingChatMock(sleep_seconds=0.2)
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.latency_seconds >= 0.2


def test_non_2xx_response_becomes_error_without_crashing() -> None:
    mock = RecordingChatMock(chat_status=503)
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.error is not None
    assert result.answer == ""


def test_transport_timeout_becomes_error_without_crashing() -> None:
    mock = RecordingChatMock(raise_exc=httpx.ReadTimeout("simulated timeout"))
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.error is not None
    assert "timeout" in result.error.lower()


def test_malformed_json_becomes_error_without_crashing() -> None:
    mock = RecordingChatMock(malformed_json=True)
    case = _simple_case()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-1")

    result = anyio.run(body)
    assert result.error is not None


def test_scoring_is_identical_for_local_and_http_shaped_results() -> None:
    """metrics.score_case() must never branch on how an EvalResult was
    produced -- there is deliberately no separate scoring path for HTTP
    results (see evaluation/evaluator.py's module docstring)."""
    case = _simple_case(expected_tools=["search_policy_documents"], rubric_keywords=["grounded"])
    shared_fields = dict(
        case_id="EVAL-TEST", category="A_policy_simple", user_input=case.user_input,
        answer="Here is your grounded answer.", citations=[], trace=[],
        tools_called=["search_policy_documents"], requires_clarification=False,
        requires_confirmation=False, escalated=False, latency_seconds=1.23,
    )
    local_like = EvalResult(**shared_fields)
    http_like = EvalResult(**shared_fields)  # identical fields, standing in for an HTTP-mode result

    assert score_case(case, local_like) == score_case(case, http_like)


# ---------------------------------------------------------------------------
# EVAL-015 / EVAL-023 multi-turn replay (items 11, 12)
# ---------------------------------------------------------------------------
def test_multiturn_case_replays_prior_user_turn_under_same_conversation_id() -> None:
    case = CASES_BY_ID["EVAL-015"]
    assert case.conversation_context  # sanity: this case actually has prior turns
    mock = RecordingChatMock()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-015")

    result = anyio.run(body)
    assert result.error is None
    # One request per prior *user* turn, plus one for the evaluated turn.
    prior_user_turns = [t for t in case.conversation_context if t.role == "user"]
    assert len(mock.requests) == len(prior_user_turns) + 1
    assert all(r["conversation_id"] == "conv-015" for r in mock.requests)
    assert mock.requests[0]["message"] == prior_user_turns[0].content
    assert mock.requests[-1]["message"] == case.user_input


def test_action_confirmation_case_replays_prior_turn_same_conversation_id() -> None:
    case = CASES_BY_ID["EVAL-023"]
    assert case.conversation_context
    mock = RecordingChatMock()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-023")

    result = anyio.run(body)
    assert result.error is None
    assert len(mock.requests) == 2  # the scripted prior user turn + the evaluated "Yes, please proceed."
    assert mock.requests[0]["conversation_id"] == mock.requests[1]["conversation_id"] == "conv-023"
    assert mock.requests[1]["message"] == case.user_input


def test_eval_023_http_replay_uses_case_content_not_hidden_script_values() -> None:
    """G (2026-09 harness-correction audit): the replayed first turn must
    come from EVAL-023's own conversation_context -- not from any value
    evaluation/scripts.py happens to hard-code for local/offline mode.
    run_case_http (evaluator.py) never imports evaluation.scripts at all,
    so this holds structurally; this test additionally proves the replayed
    text itself carries the case's own employee ID and discrepancy detail,
    which is what makes a real deployed model's own confirmation-offer
    turn (not an injected scripted one) plausible in the first place."""
    import evaluation.evaluator as evaluator_module

    # evaluator.py (run_case_http included) never imports the scripted-mode
    # module at all -- structural proof the HTTP path has no access to it.
    assert not hasattr(evaluator_module, "get_script")
    assert not hasattr(evaluator_module, "SCRIPTS")

    case = CASES_BY_ID["EVAL-023"]
    prior_user_turn = next(t for t in case.conversation_context if t.role == "user")
    mock = RecordingChatMock()

    async def body() -> EvalResult:
        async with _client_for(mock) as client:
            return await run_case_http(case, "https://example.test", client, "conv-023-g")

    anyio.run(body)

    assert mock.requests[0]["message"] == prior_user_turn.content
    assert "EMP-1001" in mock.requests[0]["message"]
    assert "8 hours" in mock.requests[0]["message"]


# ---------------------------------------------------------------------------
# run_live._main() integration tests (items 1, 2, 11, 13, 14, 15, 16, 18)
# ---------------------------------------------------------------------------
def _run_main(monkeypatch, tmp_path, mock: RecordingChatMock, **kwargs) -> dict[str, Any]:
    """Drive evaluation.run_live._main() against a mocked deployed
    endpoint, isolated from the real evaluation/results/ directory."""
    monkeypatch.setenv("EVAL_BASE_URL", "https://example.test")
    monkeypatch.setattr(run_live, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_live, "_new_http_client", lambda: _client_for(mock))

    args = dict(case_ids=None, limit=None, delay_between_cases=0.0, resume_path=None, checkpoint_file=None)
    args.update(kwargs)

    exit_code = anyio.run(
        run_live._main, args["case_ids"], args["limit"], args["delay_between_cases"],
        args["resume_path"], args["checkpoint_file"],
    )
    assert exit_code == 0

    out_files = [p for p in tmp_path.glob("live_*.json") if "checkpoint" not in p.name]
    assert len(out_files) == 1
    return json.loads(out_files[0].read_text(encoding="utf-8"))


def test_eval_base_url_absent_preserves_existing_local_behavior(monkeypatch, capsys) -> None:
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    exit_code = anyio.run(run_live._main, ["EVAL-001"], None, 0.0, None, None)

    assert exit_code == 0  # unchanged: "not configured" exits cleanly, exit code 0
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY is not set" in captured.out  # unchanged local-mode guidance text


def test_eval_base_url_set_does_not_construct_local_mcp_or_llm(monkeypatch, tmp_path) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("local MCPClient/LLM provider must not be constructed in HTTP mode")

    monkeypatch.setattr(run_live, "MCPClient", _boom)
    monkeypatch.setattr(run_live, "get_provider", _boom)

    mock = RecordingChatMock()
    data = _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001"])
    assert data["metadata"]["execution_mode"] == "deployed_http"
    assert len(mock.requests) == 1


def test_each_case_gets_a_unique_conversation_id(monkeypatch, tmp_path) -> None:
    mock = RecordingChatMock()
    _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001", "EVAL-005"])
    conversation_ids = [r["conversation_id"] for r in mock.requests]
    assert len(conversation_ids) == 2
    assert len(set(conversation_ids)) == 2


def test_delay_between_cases_is_outside_measured_latency(monkeypatch, tmp_path) -> None:
    mock = RecordingChatMock()
    data = _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001", "EVAL-005"], delay_between_cases=0.3)

    for result in data["results"]:
        # The mock responds essentially instantly; if the 0.3s pacing delay
        # had leaked into the timed section, this would be >= 0.3s too.
        assert result["latency_seconds"] < 0.2


def test_checkpoint_is_written_after_each_case(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    seen_lengths: list[int] = []
    original_write = run_live._write_checkpoint

    def _spy_write(path, metadata, result_dicts):
        seen_lengths.append(len(result_dicts))
        original_write(path, metadata, result_dicts)

    monkeypatch.setattr(run_live, "_write_checkpoint", _spy_write)
    mock = RecordingChatMock()
    _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001", "EVAL-005", "EVAL-012"],
              checkpoint_file=checkpoint_path)

    assert seen_lengths == [1, 2, 3]  # one checkpoint write per completed case, growing each time
    assert checkpoint_path.exists()
    checkpoint_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert len(checkpoint_data["results"]) == 3


def test_resume_skips_already_completed_cases_and_preserves_their_results(monkeypatch, tmp_path) -> None:
    # Pre-seed a checkpoint as if EVAL-001 already completed (with an error,
    # to also prove an errored case is preserved as-is, never silently
    # upgraded to a pass -- see _load_checkpoint's docstring).
    prior_result = EvalResult(
        case_id="EVAL-001", category="A_policy_simple", user_input="What is Solara's paid holiday policy?",
        answer="", citations=[], trace=[], tools_called=[], requires_clarification=False,
        requires_confirmation=False, escalated=False, latency_seconds=5.0, error="simulated prior timeout",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps({"metadata": {}, "results": [prior_result.to_dict()]}), encoding="utf-8")

    mock = RecordingChatMock()
    data = _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001", "EVAL-005"], resume_path=checkpoint_path)

    # Only EVAL-005 should have been re-run over HTTP.
    assert len(mock.requests) == 1
    results_by_id = {r["case_id"]: r for r in data["results"]}
    assert results_by_id["EVAL-001"]["error"] == "simulated prior timeout"  # preserved, not silently fixed
    assert results_by_id["EVAL-005"]["error"] is None
    assert data["metadata"]["resumed_from"] == str(checkpoint_path)


def test_resumed_and_new_results_aggregate_with_unchanged_scoring(monkeypatch, tmp_path) -> None:
    prior_result = EvalResult(
        case_id="EVAL-001", category="A_policy_simple", user_input="What is Solara's paid holiday policy?",
        answer="Solara observes 10 fixed holidays.", citations=[], trace=[],
        tools_called=["search_policy_documents"], requires_clarification=False,
        requires_confirmation=False, escalated=False, latency_seconds=1.0, error=None,
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps({"metadata": {}, "results": [prior_result.to_dict()]}), encoding="utf-8")

    mock = RecordingChatMock()
    data = _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001", "EVAL-005"], resume_path=checkpoint_path)

    assert data["aggregate"]["total_cases"] == 2
    assert data["aggregate"]["errors"] == 0


def test_no_secret_is_written_into_result_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-super-secret-value-should-never-appear")
    mock = RecordingChatMock()
    data = _run_main(monkeypatch, tmp_path, mock, case_ids=["EVAL-001"])

    raw_text = json.dumps(data)
    assert "sk-super-secret-value-should-never-appear" not in raw_text
    assert "GEMINI_API_KEY" not in raw_text
