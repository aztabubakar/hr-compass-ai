"""Tests proving the agent orchestrator genuinely performs dynamic,
model-driven tool selection over the real MCP protocol — not a hard-coded
workflow. `llm.fake_provider.FakeLLMProvider` stands in for the model: a
test scripts what the "model" decides turn by turn, and assertions check
that the *orchestrator* correctly executed those decisions through
`mcp_client.MCPClient` against a real `mcp_server.server` subprocess, fed
real results back, and produced correct structured output. The
orchestrator itself contains no knowledge of which tool sequence
corresponds to "a PTO question" or "a remote-work question" — that
knowledge lives only in the script each test provides, standing in for
what a real model would decide.
"""
from __future__ import annotations

import anyio

from agent.orchestrator import AgentOrchestrator
from llm.fake_provider import FakeLLMProvider, final_turn, tool_call_turn
from mcp_client.client import MCPClient


def _run(mcp_server_env: dict[str, str], fake: FakeLLMProvider, message: str, **kwargs):
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            orchestrator = AgentOrchestrator(fake, client, **{k: v for k, v in kwargs.items() if k == "max_iterations"})
            run_kwargs = {k: v for k, v in kwargs.items() if k != "max_iterations"}
            return await orchestrator.run(message, **run_kwargs)

    return anyio.run(body)


# ---------------------------------------------------------------------------
# A. Policy-only question
# ---------------------------------------------------------------------------
def test_policy_only_question_does_not_look_up_an_employee(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "paid holiday schedule", "top_k": 3}),
        final_turn("Solara Dynamics observes 10 fixed paid holidays plus 2 floating holidays per year, per POL-HOL-001."),
    ])
    result = _run(mcp_server_env, fake, "What is Solara's paid holiday policy?")

    tools_called = {step["tool"] for step in result.trace if step["type"] == "tool_call"}
    assert tools_called == {"search_policy_documents"}
    assert "lookup_employee_profile" not in tools_called
    assert result.citations
    assert all(c.document_id == "POL-HOL-001" or c.document_id.startswith("POL-") for c in result.citations)


# ---------------------------------------------------------------------------
# B. PTO workflow (dynamic multi-tool)
# ---------------------------------------------------------------------------
def test_pto_workflow_dynamic_tool_sequence(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("search_policy_documents", {"query": "PTO request notice and manager approval", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 10,
        }),
        final_turn("Yes — you have 48 hours available, which covers 3 days, and your notice is sufficient per POL-PTO-001."),
    ])
    result = _run(mcp_server_env, fake, "My employee ID is EMP-1001. Can I take three days of PTO next week?")

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert set(tools_called) >= {"lookup_employee_profile", "check_pto_balance", "check_policy_compliance"}
    assert result.citations
    assert result.requires_confirmation is False
    # Nothing was created/sent automatically.
    assert "create_mock_hr_ticket" not in tools_called
    assert "draft_hr_email" not in tools_called


# ---------------------------------------------------------------------------
# C. Remote work - interstate
# ---------------------------------------------------------------------------
def test_remote_work_interstate_multi_document_evidence(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        tool_call_turn("search_policy_documents", {"query": "remote work Work From Anywhere allowance", "document_id": "POL-RWK-001", "top_k": 2}),
        tool_call_turn("search_policy_documents", {"query": "interstate work location payroll registered states", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1002", "scenario_type": "remote_work_request",
            "destination_state": "Colorado", "duration_calendar_days": 42,
        }),
        final_turn("Working from Colorado for six weeks needs Payroll and People & Culture approval, per POL-LOC-001."),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1002. Can I work from Colorado for six weeks?")

    doc_ids = {c.document_id for c in result.citations}
    assert len(doc_ids) >= 2, f"expected multi-document evidence, got {doc_ids}"
    assert result.escalated is True


# ---------------------------------------------------------------------------
# D. Remote work - international
# ---------------------------------------------------------------------------
def test_remote_work_international_thresholds_and_escalation(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1003"}),
        tool_call_turn("search_policy_documents", {"query": "international remote work Employer of Record approved countries", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1003", "scenario_type": "remote_work_request",
            "destination_country": "Portugal", "duration_calendar_days": 42,
        }),
        final_turn("Portugal is an approved country, but this needs Global Mobility review and a signed attestation before you can go."),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1003. Can I work from Portugal for six weeks?")

    assert result.citations
    assert result.escalated is True


# ---------------------------------------------------------------------------
# E. Benefits question requiring structured data
# ---------------------------------------------------------------------------
def test_benefits_question_uses_structured_lookup(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_benefits_status", {"employee_id": "EMP-1004"}),
        final_turn("You are not currently eligible for medical benefits as a part-time employee, per your benefits record."),
    ])
    result = _run(mcp_server_env, fake, "EMP-1004 -- am I enrolled in medical benefits?")

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert tools_called == ["lookup_benefits_status"]


# ---------------------------------------------------------------------------
# F. Ambiguous request
# ---------------------------------------------------------------------------
def test_ambiguous_request_asks_for_clarification_without_tool_calls(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([final_turn("CLARIFY: Could you tell me which state or country, and for roughly how long?")])
    result = _run(mcp_server_env, fake, "Can I work somewhere else next month?")

    assert result.requires_clarification is True
    assert not any(step["type"] == "tool_call" for step in result.trace)
    assert "state or country" in result.answer.lower()


# ---------------------------------------------------------------------------
# G. Missing employee ID
# ---------------------------------------------------------------------------
def test_missing_employee_id_asks_for_it(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([final_turn("CLARIFY: What is your employee ID?")])
    result = _run(mcp_server_env, fake, "How much PTO do I have?")

    assert result.requires_clarification is True
    assert "employee id" in result.answer.lower()
    assert not any(step["type"] == "tool_call" for step in result.trace)


# ---------------------------------------------------------------------------
# H. Invalid employee ID
# ---------------------------------------------------------------------------
def test_invalid_employee_id_does_not_hallucinate(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-9999"}),
        final_turn("I couldn't find an employee with ID EMP-9999 -- could you double-check it?"),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-9999, how much PTO do I have?")

    error_steps = [s for s in result.trace if s["status"] == "error"]
    assert len(error_steps) == 1
    assert "EMP-9999" in error_steps[0]["result_summary"]
    # The real tool error (not a fabricated one) was fed back to the model.
    last_call_messages = fake.calls[-1]["messages"]
    tool_messages = [m for m in last_call_messages if m.role == "tool"]
    assert any("EMP-9999" in (m.content or "") for m in tool_messages)


# ---------------------------------------------------------------------------
# I. Out-of-corpus question
# ---------------------------------------------------------------------------
def test_out_of_corpus_question_reflects_real_low_evidence(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "stock option vesting schedule equity grant", "top_k": 5}),
        final_turn("I don't have a policy on stock option vesting in the available corpus, so I can't answer that from policy evidence. You may want to contact HR directly."),
    ])
    result = _run(mcp_server_env, fake, "What is the company's stock option vesting schedule?")

    # Inspect what the model actually received: real retrieval, not a
    # fabricated "no results" placeholder.
    last_call_messages = fake.calls[-1]["messages"]
    tool_messages = [m for m in last_call_messages if m.role == "tool"]
    assert tool_messages
    import json
    search_result = json.loads(tool_messages[-1].content)
    assert search_result["has_sufficient_evidence"] is False
    assert "contact hr" in result.answer.lower() or "don't have" in result.answer.lower()


# ---------------------------------------------------------------------------
# J. Sensitive workplace issue
# ---------------------------------------------------------------------------
def test_sensitive_workplace_issue_grounds_and_offers_confirmation_only(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "harassment reporting workplace conduct", "document_id": "POL-CON-001", "top_k": 3}),
        final_turn(
            "CONFIRM: I'm sorry you're dealing with this. Per POL-CON-001, you can report it to your manager, "
            "People & Culture, or the confidential Ethics Hotline, and it isn't a substitute for a full HR "
            "investigation. I can create a mock HR case to start this. Would you like me to proceed?"
        ),
    ])
    result = _run(mcp_server_env, fake, "I want to report harassment by a coworker.")

    assert result.requires_confirmation is True
    assert result.citations
    assert any(c.document_id == "POL-CON-001" for c in result.citations)
    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert "create_mock_hr_ticket" not in tools_called  # never created without confirmation


# ---------------------------------------------------------------------------
# Action confirmation safety (code-enforced gate, not just prompting)
# ---------------------------------------------------------------------------
def test_ticket_creation_without_prior_confirmation_turn_is_downgraded_to_preview(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Conduct", "summary": "test", "confirmed": True,
        }),
        final_turn("Here's a preview of the mock ticket; let me know if you'd like me to create it."),
    ])
    result = _run(mcp_server_env, fake, "Please file a ticket about my coworker.",
                  prior_turn_requires_confirmation=False)

    tool_step = next(s for s in result.trace if s["tool"] == "create_mock_hr_ticket")
    assert tool_step["arguments"]["confirmed"] is False  # downgraded despite the model's request


def test_ticket_creation_after_confirmation_offer_is_allowed(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Conduct", "summary": "test", "confirmed": True,
        }),
        final_turn("Done -- I've created the mock ticket."),
    ])
    result = _run(mcp_server_env, fake, "Yes, please proceed.", prior_turn_requires_confirmation=True)

    tool_step = next(s for s in result.trace if s["tool"] == "create_mock_hr_ticket")
    assert tool_step["arguments"]["confirmed"] is True


def test_draft_email_is_never_marked_as_sent(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("draft_hr_email", {
            "employee_id": "EMP-1001", "recipient": "Devon Marsh",
            "subject_intent": "PTO request", "key_facts": ["48 hours available"],
        }),
        final_turn("I've drafted a message to your manager -- it hasn't been sent."),
    ])
    result = _run(mcp_server_env, fake, "Can you draft a message to my manager about my PTO?")

    draft_step = next(s for s in result.trace if s["tool"] == "draft_hr_email")
    assert draft_step["status"] == "success"
    assert "sent" not in result.answer.lower() or "hasn't been sent" in result.answer.lower()


# ---------------------------------------------------------------------------
# Max iteration protection
# ---------------------------------------------------------------------------
def test_max_iterations_protection_prevents_infinite_loop(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([tool_call_turn("search_policy_documents", {"query": "policy", "top_k": 1}) for _ in range(20)])
    result = _run(mcp_server_env, fake, "Tell me everything.", max_iterations=4)

    assert result.escalated is True
    assert len(fake.calls) == 4  # loop stopped exactly at the cap
    tool_call_steps = [s for s in result.trace if s["type"] == "tool_call"]
    assert len(tool_call_steps) == 4


# ---------------------------------------------------------------------------
# Unknown tool name (a "hallucinated" tool call) is handled gracefully
# ---------------------------------------------------------------------------
def test_hallucinated_tool_name_is_handled_gracefully(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("delete_employee_record", {"employee_id": "EMP-1001"}),  # does not exist
        final_turn("I don't have a way to do that -- I can only look up information or draft/preview HR actions."),
    ])
    result = _run(mcp_server_env, fake, "Delete my employee record.")

    error_step = next(s for s in result.trace if s["tool"] == "delete_employee_record")
    assert error_step["status"] == "error"
    assert result.answer  # the loop recovered and produced a normal answer


# ---------------------------------------------------------------------------
# Citation propagation
# ---------------------------------------------------------------------------
def test_citations_propagate_from_tool_results_with_correct_fields(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "401k retirement match", "top_k": 2}),
        final_turn("Solara matches part of your 401(k) contributions, per POL-BEN-001."),
    ])
    result = _run(mcp_server_env, fake, "What is the 401k match?")

    assert result.citations
    for citation in result.citations:
        assert citation.document_id.startswith("POL-")
        assert citation.section
        assert citation.snippet
    assert result.snippets == [c.snippet for c in result.citations]


# ---------------------------------------------------------------------------
# MCP unavailable is handled gracefully (server never even starts)
# ---------------------------------------------------------------------------
def test_mcp_discovery_failure_returns_graceful_response() -> None:
    fake = FakeLLMProvider([final_turn("should not be reached")])

    async def body():
        client = MCPClient(command="/definitely/not/a/real/executable", args=[])
        orchestrator = AgentOrchestrator(fake, client)
        return await orchestrator.run("What is the PTO policy?")

    result = anyio.run(body)
    assert result.escalated is True
    assert len(fake.calls) == 0  # tool discovery failed before the LLM was ever called
    assert "trouble" in result.answer.lower() or "try again" in result.answer.lower()
