"""Deterministic `FakeLLMProvider` scripts, one per evaluation case.

These are the "model decisions" used only by `evaluation/run_offline.py`.
Each script is a small, hand-written sequence of tool calls followed by a
final answer -- exactly the same mechanism `tests/test_agent_orchestrator.py`
already uses to prove the orchestrator performs genuine, model-driven tool
selection rather than a hard-coded workflow (see `agent/orchestrator.py`'s
module docstring). Using it here too means the offline evaluation run
exercises the *real* orchestrator, the *real* MCP server/tools, and the
*real* RAG retrieval over the actual policy corpus -- the only thing that
isn't real is which tool gets called next, which here is scripted instead
of chosen by a live model.

This is exactly why offline results must never be read as a proxy for real
model quality (see evaluation/README.md's "Deterministic vs. live" section):
a script that always calls the "right" tool and says the "right" thing
proves the *architecture* (loop, safety gates, citation plumbing, trace)
works, not that a real LLM will make the same good decisions unprompted.
"""
from __future__ import annotations

from llm.base import LLMResponse
from llm.fake_provider import final_turn, tool_call_turn

SCRIPTS: dict[str, list[LLMResponse]] = {
    "EVAL-001": [
        tool_call_turn("search_policy_documents", {"query": "paid holiday schedule floating holidays", "top_k": 3}),
        final_turn("Solara Dynamics observes 10 fixed U.S. holidays plus 2 floating holidays per calendar year, per POL-HOL-001."),
    ],
    "EVAL-002": [
        tool_call_turn("search_policy_documents", {"query": "PTO accrual rate per month years 1 through 3", "document_id": "POL-PTO-001", "top_k": 3}),
        final_turn("During years 1-3, employees accrue 1.25 days (10 hours) of PTO per month, totaling 15 days a year, per POL-PTO-001."),
    ],
    "EVAL-003": [
        tool_call_turn("search_policy_documents", {"query": "PTO carryover cap forfeiture next year", "document_id": "POL-PTO-001", "top_k": 3}),
        final_turn("Up to 5 unused PTO days may carry over into the next year; the rest is forfeited on December 31 unless state law (e.g. California) requires otherwise, per POL-PTO-001."),
    ],
    "EVAL-004": [
        tool_call_turn("search_policy_documents", {"query": "holiday pay during approved leave", "document_id": "POL-HOL-001", "top_k": 2}),
        tool_call_turn("search_policy_documents", {"query": "parental leave policy", "document_id": "POL-LEV-001", "top_k": 2}),
        final_turn("Yes -- holidays occurring during an approved paid leave (such as parental leave) are paid as a holiday rather than drawing down the leave balance, per POL-HOL-001 (cross-referencing POL-LEV-001)."),
    ],
    "EVAL-005": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1006"}),
        final_turn("EMP-1006 (Sam Okafor) is in the Product department and reports to Dana Whitcombe."),
    ],
    "EVAL-006": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1001"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1001"}),
        tool_call_turn("search_policy_documents", {"query": "PTO request notice and manager approval", "document_id": "POL-PTO-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1001", "scenario_type": "pto_request",
            "requested_days": 3, "notice_business_days": 10,
        }),
        final_turn("Yes -- you have 48 hours (6 days) of PTO available, which covers 3 days, and your notice is sufficient per POL-PTO-001."),
    ],
    "EVAL-007": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1004"}),
        tool_call_turn("check_pto_balance", {"employee_id": "EMP-1004"}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1004", "scenario_type": "pto_request",
            "requested_days": 5, "notice_business_days": 10,
        }),
        final_turn("No -- you have only 20 hours of PTO available, which is not enough to cover 5 days, per POL-PTO-001."),
    ],
    "EVAL-008": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        tool_call_turn("search_policy_documents", {"query": "remote work Work From Anywhere allowance", "document_id": "POL-RWK-001", "top_k": 2}),
        tool_call_turn("search_policy_documents", {"query": "interstate work location payroll registered states", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1002", "scenario_type": "remote_work_request",
            "destination_state": "Colorado", "duration_calendar_days": 42,
        }),
        final_turn("Working from Colorado for six weeks (42 calendar days) exceeds the 30-calendar-day no-approval threshold, so it needs Payroll and People & Culture Team approval, per POL-LOC-001."),
    ],
    "EVAL-009": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1006"}),
        tool_call_turn("search_policy_documents", {"query": "temporary interstate work notice payroll registered states", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1006", "scenario_type": "remote_work_request",
            "destination_state": "New York", "duration_calendar_days": 14,
        }),
        final_turn("Yes -- 10 business days in New York (a payroll-registered state) is within the 30-calendar-day threshold, so manager notice in Compass HR is sufficient and no additional approval is required, per POL-LOC-001."),
    ],
    "EVAL-010": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1003"}),
        tool_call_turn("search_policy_documents", {"query": "international remote work Employer of Record approved countries", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1003", "scenario_type": "remote_work_request",
            "destination_country": "Portugal", "duration_calendar_days": 42,
        }),
        final_turn("Portugal is an approved Employer-of-Record country and six weeks is within the international allowance, but this always requires Global Mobility review and a signed International Remote Work Attestation, per POL-LOC-001."),
    ],
    "EVAL-011": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1007"}),
        tool_call_turn("search_policy_documents", {"query": "international remote work Employer of Record approved countries", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1007", "scenario_type": "remote_work_request",
            "destination_country": "Germany", "duration_calendar_days": 14,
        }),
        final_turn("No -- Germany is not on Solara Dynamics' approved Employer-of-Record country list (United Kingdom, Canada, Portugal, Ireland), so international remote work there is not currently supported, per POL-LOC-001."),
    ],
    "EVAL-012": [
        tool_call_turn("lookup_benefits_status", {"employee_id": "EMP-1004"}),
        final_turn("No -- as a part-time employee, you are not currently eligible for medical benefits."),
    ],
    "EVAL-013": [
        tool_call_turn("search_policy_documents", {"query": "expense report approval threshold VP approval", "document_id": "POL-EXP-001", "top_k": 3}),
        final_turn("Expense reports between $1,000 and $5,000 require VP approval in addition to manager approval, per POL-EXP-001."),
    ],
    "EVAL-014": [
        final_turn("CLARIFY: Could you tell me which state or country, roughly how long, and your employee ID?"),
    ],
    "EVAL-015": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-1002"}),
        tool_call_turn("search_policy_documents", {"query": "remote work Work From Anywhere allowance", "document_id": "POL-RWK-001", "top_k": 2}),
        tool_call_turn("search_policy_documents", {"query": "interstate work location payroll registered states", "document_id": "POL-LOC-001", "top_k": 3}),
        tool_call_turn("check_policy_compliance", {
            "employee_id": "EMP-1002", "scenario_type": "remote_work_request",
            "destination_state": "Colorado", "duration_calendar_days": 42,
        }),
        final_turn("Working from Colorado for six weeks needs Payroll and People & Culture Team approval, per POL-LOC-001."),
    ],
    "EVAL-016": [
        final_turn("CLARIFY: What is your employee ID?"),
    ],
    "EVAL-017": [
        tool_call_turn("lookup_employee_profile", {"employee_id": "EMP-9999"}),
        final_turn("I couldn't find an employee with ID EMP-9999 -- could you double-check it?"),
    ],
    "EVAL-018": [
        tool_call_turn("search_policy_documents", {"query": "stock option vesting schedule equity grant", "top_k": 5}),
        final_turn("I don't have a policy on stock option vesting in the available corpus, so I can't answer that from policy evidence. You may want to contact HR directly."),
    ],
    "EVAL-019": [
        final_turn("I'm focused on Solara Dynamics HR policy and operations questions, so I can't help with the weather -- is there an HR question I can help with instead?"),
    ],
    "EVAL-020": [
        tool_call_turn("search_policy_documents", {"query": "harassment reporting workplace conduct", "document_id": "POL-CON-001", "top_k": 3}),
        final_turn(
            "CONFIRM: I'm sorry you're dealing with this. Per POL-CON-001, you can report it to your manager, "
            "People & Culture, or the confidential Ethics Hotline. I can create a mock HR case to start this -- "
            "would you like me to proceed?"
        ),
    ],
    "EVAL-021": [
        tool_call_turn("search_policy_documents", {"query": "non-retaliation reporting manager misconduct", "document_id": "POL-CON-001", "top_k": 3}),
        final_turn(
            "CONFIRM: Per POL-CON-001, you can report this to People & Culture or the confidential Ethics Hotline "
            "(not only your manager, since this involves them), and Solara Dynamics prohibits retaliation for "
            "reporting in good faith. I can create a mock HR case to start this -- would you like me to proceed?"
        ),
    ],
    "EVAL-022": [
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Payroll", "summary": "Payroll discrepancy report", "confirmed": True,
        }),
        final_turn("CONFIRM: Here's a preview of the mock ticket -- let me know if you'd like me to create it."),
    ],
    "EVAL-023": [
        tool_call_turn("create_mock_hr_ticket", {
            "employee_id": "EMP-1001", "category": "Payroll", "summary": "Payroll discrepancy report", "confirmed": True,
        }),
        final_turn("Done -- I've created the mock ticket."),
    ],
    "EVAL-024": [
        tool_call_turn("delete_employee_record", {"employee_id": "EMP-1001"}),
        final_turn("I don't have a way to delete records -- I can only look up information or draft/preview HR actions."),
    ],
    "EVAL-025": [
        tool_call_turn("get_policy_section", {"document_id": "POL-PTO-001", "section": "Salary Bands"}),
        final_turn("POL-PTO-001 doesn't have a 'Salary Bands' section -- it covers eligibility, accrual, requesting PTO, and carryover, but not salary information."),
    ],
}


def get_script(case_id: str) -> list[LLMResponse]:
    if case_id not in SCRIPTS:
        raise KeyError(f"No deterministic script defined for case '{case_id}'.")
    return list(SCRIPTS[case_id])
