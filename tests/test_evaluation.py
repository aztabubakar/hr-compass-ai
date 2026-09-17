"""Tests for the evaluation harness itself (evaluation/). These are
offline and deterministic -- same constraint as the rest of the suite --
and prove the harness's plumbing (case loading, scripted runs through the
real orchestrator/MCP layer, and metric computation) is correct, not that
a real model performs well (see evaluation/README.md).
"""
from __future__ import annotations

import anyio

from evaluation.cases import EvalCase, load_cases
from evaluation.evaluator import EvalResult, run_case
from evaluation.metrics import aggregate, latency_stats, score_case
from evaluation.scripts import SCRIPTS, get_script
from llm.fake_provider import FakeLLMProvider
from mcp_client.client import MCPClient


def test_cases_json_has_25_well_formed_cases() -> None:
    cases = load_cases()
    assert len(cases) == 25
    assert len({c.case_id for c in cases}) == 25  # unique IDs
    for case in cases:
        assert case.case_id.startswith("EVAL-")
        assert case.user_input
        assert case.expected_behavior


def test_every_case_has_a_deterministic_script() -> None:
    cases = load_cases()
    for case in cases:
        assert case.case_id in SCRIPTS, f"{case.case_id} has no deterministic script"
        assert get_script(case.case_id)  # non-empty


def test_categories_cover_the_required_spread() -> None:
    categories = {c.category for c in load_cases()}
    # One letter-prefixed category per required behavior group (A through O
    # in the Phase 4 task description).
    prefixes = {c[0] for c in categories}
    assert prefixes == set("ABCDEFGHIJKLMNO")


def _run(mcp_server_env: dict[str, str], case: EvalCase) -> EvalResult:
    async def body():
        fake = FakeLLMProvider(get_script(case.case_id))
        async with MCPClient(env=mcp_server_env) as client:
            return await run_case(case, fake, client)

    return anyio.run(body)


def test_pto_workflow_case_runs_through_the_real_orchestrator_and_mcp_layer(mcp_server_env: dict[str, str]) -> None:
    case = next(c for c in load_cases() if c.case_id == "EVAL-006")
    result = _run(mcp_server_env, case)

    assert result.error is None
    assert result.answer
    assert set(case.expected_tools).issubset(set(result.tools_called))
    assert result.citations


def test_action_safety_case_ticket_downgraded_without_confirmation(mcp_server_env: dict[str, str]) -> None:
    case = next(c for c in load_cases() if c.case_id == "EVAL-022")
    result = _run(mcp_server_env, case)

    ticket_step = next(s for s in result.trace if s["tool"] == "create_mock_hr_ticket")
    assert ticket_step["arguments"]["confirmed"] is False  # downgraded by the code-level gate


def test_action_safety_case_ticket_allowed_after_confirmation(mcp_server_env: dict[str, str]) -> None:
    case = next(c for c in load_cases() if c.case_id == "EVAL-023")
    result = _run(mcp_server_env, case)

    ticket_step = next(s for s in result.trace if s["tool"] == "create_mock_hr_ticket")
    assert ticket_step["arguments"]["confirmed"] is True
    assert ticket_step["status"] == "success"


def test_conversation_continuity_case_resumes_after_clarification(mcp_server_env: dict[str, str]) -> None:
    case = next(c for c in load_cases() if c.case_id == "EVAL-015")
    assert case.conversation_context  # sanity: this case actually has prior turns
    result = _run(mcp_server_env, case)

    assert result.error is None
    tools_called = set(result.tools_called)
    assert set(case.expected_tools).issubset(tools_called)
    assert not any(step["type"] == "decision" and "trouble" in step["result_summary"].lower() for step in result.trace)


# ---------------------------------------------------------------------------
# metrics.py unit tests (synthetic results, no MCP/orchestrator needed)
# ---------------------------------------------------------------------------
def _case(**overrides) -> EvalCase:
    base = dict(
        case_id="EVAL-TEST", category="A_policy_simple", user_input="test",
        expected_behavior="test",
    )
    base.update(overrides)
    return EvalCase.model_validate(base)


def _result(**overrides) -> EvalResult:
    base = dict(
        case_id="EVAL-TEST", category="A_policy_simple", user_input="test",
        answer="", citations=[], trace=[], tools_called=[],
        requires_clarification=False, requires_confirmation=False, escalated=False,
        latency_seconds=0.1,
    )
    base.update(overrides)
    return EvalResult(**base)


def test_groundedness_scores_partial_keyword_coverage() -> None:
    case = _case(rubric_keywords=["10 hours", "vacation"])
    result = _result(answer="You accrue 10 hours per month.")
    score = score_case(case, result)
    assert score.groundedness_score == 0.5


def test_citation_recall_and_precision() -> None:
    case = _case(expected_policy_documents=["POL-A", "POL-B"])
    result = _result(citations=[{"document_id": "POL-A", "document_title": "A", "section": "s", "snippet": "x"},
                                 {"document_id": "POL-C", "document_title": "C", "section": "s", "snippet": "y"}])
    score = score_case(case, result)
    assert score.citation_recall == 0.5   # found POL-A out of {POL-A, POL-B}
    assert score.citation_precision == 0.5  # POL-A relevant out of {POL-A, POL-C} cited


def test_tool_selection_pass_requires_expected_and_forbids_forbidden() -> None:
    case = _case(expected_tools=["search_policy_documents"], forbidden_tools=["create_mock_hr_ticket"])
    good = score_case(case, _result(tools_called=["search_policy_documents"]))
    assert good.tool_selection_pass is True

    missing_expected = score_case(case, _result(tools_called=[]))
    assert missing_expected.tool_selection_pass is False

    used_forbidden = score_case(case, _result(tools_called=["search_policy_documents", "create_mock_hr_ticket"]))
    assert used_forbidden.tool_selection_pass is False
    assert used_forbidden.forbidden_tool_violated is True


def test_action_safety_ticket_downgrade_rule() -> None:
    case = _case(expected_action_safety="ticket_downgraded_without_prior_confirmation")
    downgraded = _result(trace=[{"type": "tool_call", "tool": "create_mock_hr_ticket",
                                  "arguments": {"confirmed": False}, "status": "success", "result_summary": ""}])
    assert score_case(case, downgraded).action_safety_pass is True

    not_downgraded = _result(trace=[{"type": "tool_call", "tool": "create_mock_hr_ticket",
                                      "arguments": {"confirmed": True}, "status": "success", "result_summary": ""}])
    assert score_case(case, not_downgraded).action_safety_pass is False


def test_latency_stats_nearest_rank() -> None:
    stats = latency_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert stats["min"] == 0.1
    assert stats["max"] == 0.5
    assert stats["p50"] == 0.3


def test_aggregate_computes_category_breakdown() -> None:
    cases = [_case(case_id="EVAL-A", category="A_policy_simple", rubric_keywords=["yes"]),
             _case(case_id="EVAL-B", category="B_policy_multidoc", rubric_keywords=["yes"])]
    results = [_result(case_id="EVAL-A", category="A_policy_simple", answer="yes indeed"),
               _result(case_id="EVAL-B", category="B_policy_multidoc", answer="no")]
    report = aggregate(cases, results, mode="offline")
    assert report.total_cases == 2
    assert report.by_category["A_policy_simple"]["groundedness_avg"] == 1.0
    assert report.by_category["B_policy_multidoc"]["groundedness_avg"] == 0.0
