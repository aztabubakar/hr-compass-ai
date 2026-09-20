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
# B1. PTO with a relative date ("next week") must still reach a compliance
# check with a resolved notice figure, not stop at the balance check and
# hedge the notice question back to the user. Regression test for the
# production bug where the agent said "if you can meet the 10-business-day
# notice requirement" instead of resolving it.
# ---------------------------------------------------------------------------
def test_pto_relative_date_request_calls_compliance_check_with_resolved_notice(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("search_policy_documents", {"query": "PTO request notice and manager approval", "document_id": "POL-PTO-001", "top_k": 3}),
        # A well-behaved live model resolves "next week" against the
        # current-date system message (agent/orchestrator.py's
        # _current_date_context) and passes a concrete notice_business_days
        # -- scripted here as what that resolution would produce, since
        # FakeLLMProvider cannot itself do date arithmetic; see
        # agent/prompts.py rule 4a.
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 3,
        }),
        final_turn(
            "You have 48 hours (6 days) of PTO available, which covers 3 days. Treating "
            "\"next week\" as starting next Monday, that's about 3 business days of notice, "
            "which is short of POL-PTO-001's 10-business-day requirement for a request this "
            "length, so this would need your manager's discretion as a short-notice exception "
            "rather than standard approval."
        ),
    ])
    result = _run(mcp_server_env, fake, "My employee ID is EMP-1001. Can I take three days of PTO next week?")

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert "check_policy_compliance" in tools_called, (
        "the compliance check must not be skipped once balance and a resolvable timeframe are known"
    )
    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert compliance_step["arguments"]["requested_days"] == 3
    assert compliance_step["arguments"]["notice_business_days"] is not None

    # The model actually had the grounding to do this resolution: a
    # deterministic current-date system message must have been sent.
    sent_messages = fake.calls[0]["messages"]
    assert any(m.role == "system" and m.content and "Current date:" in m.content for m in sent_messages), (
        "the model must be given today's date to resolve a relative reference like 'next week'"
    )

    # Never claim/imply approval -- compliance status is not manager sign-off.
    assert "approved" not in result.answer.lower()
    assert "granted" not in result.answer.lower()


# ---------------------------------------------------------------------------
# B2. Insufficient notice -> requires_approval (short-notice exception), not
# silently denied and not described as approved.
# ---------------------------------------------------------------------------
def test_pto_insufficient_notice_is_flagged_for_manager_discretion(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 2,
        }),
        final_turn(
            "You have enough PTO balance for 3 days, but only 2 business days of notice was "
            "given, short of POL-PTO-001's 10-business-day requirement, so this needs your "
            "manager's discretion as a short-notice exception rather than standard approval."
        ),
    ])
    result = _run(mcp_server_env, fake, "EMP-1001, can I take 3 days off starting in 2 business days?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=requires_approval" in compliance_step["result_summary"]
    assert compliance_step["escalation"]
    assert result.escalated is True
    assert "approved" not in result.answer.lower()


# ---------------------------------------------------------------------------
# B3. Sufficient notice -> compliant, still described as meeting policy
# requirements rather than as final approval.
# ---------------------------------------------------------------------------
def test_pto_sufficient_notice_is_compliant_but_not_worded_as_approval(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 12,
        }),
        final_turn(
            "You have enough PTO balance for 3 days, and 12 business days of notice meets "
            "POL-PTO-001's 10-business-day requirement, so this meets the policy's requirements. "
            "Your manager still needs to approve the request in Compass HR."
        ),
    ])
    result = _run(mcp_server_env, fake, "EMP-1001, can I take 3 days off starting in three weeks?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=compliant" in compliance_step["result_summary"]
    assert "approved" not in result.answer.lower()
    assert "granted" not in result.answer.lower()


# ---------------------------------------------------------------------------
# B4. Insufficient balance -> non_compliant, distinct from a notice problem.
# ---------------------------------------------------------------------------
def test_pto_insufficient_balance_is_reported_as_non_compliant(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1004"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1004"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1004", "scenario_type": "pto_request",
            "requested_days": 5, "notice_business_days": 15,
        }),
        final_turn(
            "You have 20 hours of PTO available, which is not enough for 5 days (40 hours "
            "needed), so this request does not have sufficient balance regardless of notice."
        ),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1004. Can I take 5 days off starting in three weeks?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=non_compliant" in compliance_step["result_summary"]
    assert "not enough" in result.answer.lower()
    assert "approved" not in result.answer.lower()


# ---------------------------------------------------------------------------
# B5. Recovery from an invented/invalid policy section name: the model must
# use the real heading list from the tool error, not repeat the guess or
# fabricate content, per agent/prompts.py rule 3a.
# ---------------------------------------------------------------------------
def test_recovery_from_invented_policy_section_name(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("get_policy_section", {"document_id": "POL-PTO-001", "section": "Request Procedure"}),
        tool_call_turn("get_policy_section", {"document_id": "POL-PTO-001", "section": "Requesting and Approving PTO"}),
        final_turn("To request PTO, submit it through Compass HR for manager approval, per POL-PTO-001."),
    ])
    result = _run(mcp_server_env, fake, "How do I request PTO?")

    error_steps = [s for s in result.trace if s["status"] == "error"]
    assert len(error_steps) == 1
    assert "Request Procedure" in error_steps[0]["result_summary"]
    # The real heading was listed in the tool's own error, and the model
    # used it on retry rather than guessing again or giving up.
    assert "Requesting and Approving PTO" in error_steps[0]["result_summary"]

    success_steps = [s for s in result.trace if s["tool"] == "get_policy_section" and s["status"] == "success"]
    assert len(success_steps) == 1
    assert result.answer


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

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    # Genuine multi-step orchestration: employee context, policy evidence
    # from more than one document, and the rule check -- not just a single
    # check_policy_compliance call (agent/prompts.py rule 4c).
    assert tools_called.count("lookup_employee_profile") >= 1
    assert tools_called.count("search_policy_documents") >= 2
    assert "check_policy_compliance" in tools_called
    doc_ids = {c.document_id for c in result.citations}
    assert len(doc_ids) >= 2, f"expected multi-document evidence, got {doc_ids}"
    assert result.escalated is True


# ---------------------------------------------------------------------------
# C1. Interstate remote work within the no-approval threshold -> compliant,
# not escalated -- still gathered through employee + policy evidence, not
# just a bare rule check. A different employee/state than the flagship
# EMP-1002/Colorado case, to prove this isn't special-cased to one example.
# ---------------------------------------------------------------------------
def test_remote_work_interstate_within_no_approval_threshold_is_compliant(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1006"}),
        tool_call_turn("get_policy_section", {"document_id": "POL-LOC-001", "section": "Temporary Interstate Work"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1006", "scenario_type": "remote_work_request",
            "destination_state": "New York", "duration_calendar_days": 14,
        }),
        final_turn(
            "Yes -- 14 calendar days in New York is within the 30-calendar-day threshold, so "
            "manager notice in Compass HR is sufficient and no additional approval is required, "
            "per POL-LOC-001."
        ),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1006. Can I work from New York for two weeks?")

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert "lookup_employee_profile" in tools_called
    assert "check_policy_compliance" in tools_called
    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=compliant" in compliance_step["result_summary"]
    assert result.escalated is False
    assert result.citations


# ---------------------------------------------------------------------------
# C2. Interstate work exceeding the 30-calendar-day threshold ->
# requires_approval, escalated. (The EMP-1002/Colorado/six-week case above
# already covers this with full multi-document evidence; this covers it
# with a different employee/state/duration to confirm the behavior is
# generic, not tied to one hard-coded example.)
# ---------------------------------------------------------------------------
def test_remote_work_interstate_exceeding_threshold_requires_approval(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1007"}),
        tool_call_turn("search_policy_documents", {"query": "interstate work location payroll registered states approval", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1007", "scenario_type": "remote_work_request",
            "destination_state": "Washington", "duration_calendar_days": 35,
        }),
        final_turn(
            "35 calendar days in Washington exceeds the 30-calendar-day no-approval threshold, "
            "so this needs Payroll and People & Culture Team approval, per POL-LOC-001."
        ),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1007. Can I work from Washington for five weeks?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=requires_approval" in compliance_step["result_summary"]
    assert result.escalated is True
    assert "approved" not in result.answer.lower()


# ---------------------------------------------------------------------------
# C3. Unsupported/non-payroll-registered state -> requires_approval,
# treated as a permanent-relocation-style request regardless of duration.
# ---------------------------------------------------------------------------
def test_remote_work_unsupported_state_requires_approval(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        tool_call_turn("search_policy_documents", {"query": "interstate work location payroll registered states", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1002", "scenario_type": "remote_work_request",
            "destination_state": "Ohio", "duration_calendar_days": 20,
        }),
        final_turn(
            "Ohio isn't currently a state Solara Dynamics is payroll-registered in, so this is "
            "treated as a permanent relocation request needing VP and Payroll approval, per "
            "POL-LOC-001, regardless of how long you'd be there."
        ),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1002. Can I work from Ohio for a few weeks?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=requires_approval" in compliance_step["result_summary"]
    assert result.escalated is True
    assert result.citations


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

    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert "lookup_employee_profile" in tools_called
    assert "search_policy_documents" in tools_called
    assert result.citations
    assert result.escalated is True


# ---------------------------------------------------------------------------
# D1. International remote work to a country NOT on the approved EOR list
# -> non_compliant, escalated, no citations fabricated.
# ---------------------------------------------------------------------------
def test_remote_work_international_unapproved_country_is_non_compliant(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1003"}),
        tool_call_turn("get_policy_section", {"document_id": "POL-LOC-001", "section": "International Remote Work"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1003", "scenario_type": "remote_work_request",
            "destination_country": "Germany", "duration_calendar_days": 20,
        }),
        final_turn(
            "Germany isn't on Solara Dynamics' approved Employer-of-Record country list, so "
            "international remote work there isn't currently supported, per POL-LOC-001."
        ),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1003. Can I work from Germany for three weeks?")

    compliance_step = next(s for s in result.trace if s["tool"] == "check_policy_compliance")
    assert "status=non_compliant" in compliance_step["result_summary"]
    assert result.escalated is True
    assert result.citations
    assert "approved" not in result.answer.lower() or "isn't" in result.answer.lower()


# ---------------------------------------------------------------------------
# E. Remote work with missing information (no destination and/or no
# duration) -> clarify rather than guess, per rule 6/7. Distinct from full
# ambiguity: here an employee ID is known but a required piece is missing.
# ---------------------------------------------------------------------------
def test_remote_work_missing_destination_asks_for_clarification(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        final_turn("CLARIFY: Which state or country would you like to work from, and for how long?"),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1002. Can I work remotely for a while?")

    assert result.requires_clarification is True
    assert not any(step["type"] == "tool_call" for step in result.trace)
    assert "state or country" in result.answer.lower()


def test_remote_work_missing_duration_asks_for_clarification(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        final_turn("CLARIFY: How long would you like to work from Colorado?"),
    ])
    result = _run(mcp_server_env, fake, "I'm EMP-1002. Can I work from Colorado?")

    assert result.requires_clarification is True
    tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
    assert "check_policy_compliance" not in tools_called, (
        "must not guess a duration and call the compliance tool with fabricated data"
    )


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
# Redundant tool-call reduction: an exact-repeat call is served from a
# per-run cache instead of hitting the MCP server again. A narrow,
# mechanical safety net -- the primary lever against a model re-searching
# the same topic with reworded queries is prompt guidance (rule 3b), which
# this offline harness cannot itself validate (see design-and-evaluation.md).
# ---------------------------------------------------------------------------
def test_exact_duplicate_tool_call_is_served_from_cache(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "PTO request notice", "document_id": "POL-PTO-001", "top_k": 3}),
        # A model that (incorrectly) repeats the identical call.
        tool_call_turn("search_policy_documents", {"query": "PTO request notice", "document_id": "POL-PTO-001", "top_k": 3}),
        final_turn("Per POL-PTO-001, PTO requests need manager approval and advance notice."),
    ])
    result = _run(mcp_server_env, fake, "What is the PTO request process?")

    search_steps = [s for s in result.trace if s["tool"] == "search_policy_documents"]
    assert len(search_steps) == 2  # the model's second call is still recorded in the trace...
    assert search_steps[0]["duration_ms"] > 0  # ...but only the first one did real work...
    assert search_steps[1]["duration_ms"] == 0.0  # ...the second was served from cache
    assert "cached result reused" in search_steps[1]["result_summary"]
    # Citations aren't duplicated by the repeat call.
    pto_citations = [c for c in result.citations if c.document_id == "POL-PTO-001"]
    assert len(pto_citations) == len({(c.section, c.snippet[:80]) for c in pto_citations})


def test_different_arguments_are_not_treated_as_duplicates(mcp_server_env: dict[str, str]) -> None:
    # Sanity check the cache key is specific: two calls to the same tool
    # with genuinely different arguments must both actually execute.
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        final_turn("Looked up both employees."),
    ])
    result = _run(mcp_server_env, fake, "Compare EMP-1001 and EMP-1002.")

    lookup_steps = [s for s in result.trace if s["tool"] == "lookup_employee_profile"]
    assert len(lookup_steps) == 2
    assert all(s["duration_ms"] > 0 for s in lookup_steps)  # neither was a cache hit


def test_action_tools_are_never_deduplicated(mcp_server_env: dict[str, str]) -> None:
    # create_mock_hr_ticket must never be silently answered from a cached
    # prior call, even with identical arguments -- each call is a real (if
    # synthetic) decision point gated by the confirmation rule, not a pure
    # read safe to memoize.
    fake = FakeLLMProvider([
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Payroll", "summary": "test", "confirmed": True,
        }),
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Payroll", "summary": "test", "confirmed": True,
        }),
        final_turn("Done."),
    ])
    result = _run(mcp_server_env, fake, "Yes, please proceed.", prior_turn_requires_confirmation=True)

    ticket_steps = [s for s in result.trace if s["tool"] == "create_mock_hr_ticket"]
    assert len(ticket_steps) == 2
    assert all("cached result reused" not in s["result_summary"] for s in ticket_steps)


# ---------------------------------------------------------------------------
# Efficient PTO orchestration: a regression guard on tool-call count, per
# the production trace showing ~8 MCP calls before a PTO answer. A
# well-behaved (scripted, here) sequence needs at most one call per
# distinct piece of evidence -- employee, balance, one on-topic policy
# search, and the compliance check -- not repeated searches.
# ---------------------------------------------------------------------------
def test_pto_workflow_completes_in_a_bounded_number_of_efficient_tool_calls(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("search_policy_documents", {"query": "PTO request notice and manager approval", "document_id": "POL-PTO-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 3,
        }),
        final_turn("You have 48 hours (6 days) of PTO available, which covers 3 days of notice."),
    ])
    result = _run(mcp_server_env, fake, "My employee ID is EMP-1001. Can I take three days of PTO next week?")

    tool_call_steps = [s for s in result.trace if s["type"] == "tool_call"]
    # One call per distinct piece of evidence -- not the ~8 seen in the
    # flagged production trace (which included repeated, reworded
    # search_policy_documents calls on the same topic).
    assert len(tool_call_steps) <= 5
    assert [s["tool"] for s in tool_call_steps] == [
        "lookup_employee_profile", "check_pto_balance", "search_policy_documents", "check_policy_compliance",
    ]


# ---------------------------------------------------------------------------
# Latency instrumentation: every LLM turn and every MCP tool call records
# how long it took.
# ---------------------------------------------------------------------------
def test_every_llm_turn_and_tool_call_records_duration(mcp_server_env: dict[str, str]) -> None:
    fake = FakeLLMProvider([
        tool_call_turn("search_policy_documents", {"query": "holiday schedule", "document_id": "POL-HOL-001", "top_k": 2}),
        final_turn("Solara observes 10 fixed holidays, per POL-HOL-001."),
    ])
    result = _run(mcp_server_env, fake, "What is Solara's holiday policy?")

    decision_steps = [s for s in result.trace if s["type"] == "decision"]
    tool_call_steps = [s for s in result.trace if s["type"] == "tool_call"]
    assert len(decision_steps) == 2  # one per LLM turn: tool-call-requesting, then final-answer
    assert len(tool_call_steps) == 1
    assert all(isinstance(s["duration_ms"], (int, float)) and s["duration_ms"] >= 0 for s in decision_steps)
    assert all(isinstance(s["duration_ms"], (int, float)) and s["duration_ms"] >= 0 for s in tool_call_steps)


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
