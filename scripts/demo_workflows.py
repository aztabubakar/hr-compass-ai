"""Deterministic, tool-level demonstrations of the two required workflows.

These are **not** the Phase 3 agent: the sequence of tool calls below is
hard-coded here, not chosen by an LLM. What they prove is that the MCP
tool layer built in Phase 2 has everything the future agent will need —
real MCP tool calls (over stdio), citation-bearing evidence gathered from
more than one policy document, and a clean operational trace — to support:

- Workflow A (PTO): "Can I take three days of PTO next week? My employee
  ID is EMP-1001."
- Workflow B (remote work): "I'm EMP-1002. Can I work remotely from
  another state for six weeks?" (and the international variant for
  EMP-1003).
"""
from __future__ import annotations

from core.trace import TraceRecorder
from mcp_client.client import MCPClient, MCPToolError


def _sources_from_search(search_result: dict) -> list[str]:
    return [f"{h['document_id']} §{h['section']}" for h in search_result.get("results", [])]


def _sources_from_compliance(compliance_result: dict) -> list[str]:
    return [f"{c['document_id']} §{c['section']}" for c in compliance_result.get("citations", [])]


async def _call(client: MCPClient, trace: TraceRecorder, tool: str, arguments: dict) -> dict:
    """Call one MCP tool and record a trace step for it, success or failure."""
    try:
        result = await client.call_tool(tool, arguments)
    except MCPToolError as exc:
        trace.record_tool_call(tool, arguments, status="error", result_summary=exc.message)
        raise
    return result


async def run_pto_workflow(
    client: MCPClient,
    employee_id: str,
    requested_days: int,
    notice_business_days: int | None = None,
) -> tuple[list[dict], dict]:
    """Workflow A: lookup_employee_profile -> check_pto_balance ->
    search_policy_documents -> check_policy_compliance -> draft_hr_email.

    Returns (trace_as_list_of_dicts, final_compliance_result).
    """
    trace = TraceRecorder()

    profile = await _call(client, trace, "lookup_employee_profile", {"employee_id": employee_id})
    trace.record_tool_call(
        "lookup_employee_profile", {"employee_id": employee_id}, status="success",
        result_summary=f"{profile['full_name']} ({profile['employment_type']}, {profile['department_name']}).",
    )

    balance = await _call(client, trace, "check_pto_balance", {"employee_id": employee_id})
    if balance["has_tracked_balance"]:
        summary = f"Balance: {balance['balance_hours']} {balance['unit']}."
    else:
        summary = balance["note"] or "No tracked balance."
    trace.record_tool_call("check_pto_balance", {"employee_id": employee_id}, status="success",
                            result_summary=summary)

    search_args = {"query": "PTO request approval notice period business days", "top_k": 3}
    search = await _call(client, trace, "search_policy_documents", search_args)
    trace.record_tool_call(
        "search_policy_documents", search_args, status="success",
        result_summary=f"{len(search['results'])} policy chunk(s) retrieved.",
        sources=_sources_from_search(search),
    )

    compliance_args = {
        "employee_id": employee_id,
        "scenario_type": "pto_request",
        "requested_days": requested_days,
        "notice_business_days": notice_business_days,
    }
    compliance = await _call(client, trace, "check_policy_compliance", compliance_args)
    trace.record_tool_call(
        "check_policy_compliance", compliance_args, status="success",
        result_summary=f"status={compliance['status']}: {compliance['reasons'][0] if compliance['reasons'] else ''}",
        sources=_sources_from_compliance(compliance),
        escalation=(
            f"Requires: {', '.join(compliance['required_approvals'])}"
            if compliance.get("escalation_required")
            else None
        ),
    )

    if compliance["status"] == "compliant":
        subject_intent = f"PTO request: {requested_days} day(s)"
        action = "Please confirm this request in Compass HR."
    else:
        subject_intent = f"PTO request needs review: {requested_days} day(s)"
        action = "This request needs manager review before it can be approved; see reasons below."
    email_args = {
        "employee_id": employee_id,
        "recipient": profile["manager_name"] or "People & Culture Team",
        "subject_intent": subject_intent,
        "key_facts": compliance["reasons"],
        "requested_action": action,
    }
    draft = await _call(client, trace, "draft_hr_email", email_args)
    trace.record_tool_call("draft_hr_email", email_args, status="success",
                            result_summary=f"Draft prepared to {draft['to']}: '{draft['subject']}'. {draft['note']}")

    return trace.as_list(), compliance


async def run_remote_work_workflow(
    client: MCPClient,
    employee_id: str,
    destination_state: str | None = None,
    destination_country: str | None = None,
    duration_calendar_days: int = 42,  # "six weeks" (the demo scenario's own wording)
) -> tuple[list[dict], dict]:
    """Workflow B: lookup_employee_profile -> search_policy_documents
    (remote work) -> search_policy_documents (work location/tax) ->
    search_policy_documents (information security) ->
    check_policy_compliance -> draft_hr_email.

    Returns (trace_as_list_of_dicts, final_compliance_result).
    """
    trace = TraceRecorder()

    profile = await _call(client, trace, "lookup_employee_profile", {"employee_id": employee_id})
    trace.record_tool_call(
        "lookup_employee_profile", {"employee_id": employee_id}, status="success",
        result_summary=f"{profile['full_name']}, based in {profile['home_location']}.",
    )

    rwk_args = {"query": "remote work Work From Anywhere allowance business days", "top_k": 3,
                "document_id": "POL-RWK-001"}
    rwk_search = await _call(client, trace, "search_policy_documents", rwk_args)
    trace.record_tool_call("search_policy_documents", rwk_args, status="success",
                            result_summary=f"{len(rwk_search['results'])} chunk(s) from the Remote Work Policy.",
                            sources=_sources_from_search(rwk_search))

    loc_args = {"query": "interstate international work location payroll registered states approval",
                "top_k": 3, "document_id": "POL-LOC-001"}
    loc_search = await _call(client, trace, "search_policy_documents", loc_args)
    trace.record_tool_call("search_policy_documents", loc_args, status="success",
                            result_summary=f"{len(loc_search['results'])} chunk(s) from the Work Location policy.",
                            sources=_sources_from_search(loc_search))

    sec_args = {"query": "information security restricted data international access",
                "top_k": 2, "document_id": "POL-SEC-001"}
    sec_search = await _call(client, trace, "search_policy_documents", sec_args)
    trace.record_tool_call("search_policy_documents", sec_args, status="success",
                            result_summary=f"{len(sec_search['results'])} chunk(s) from the Security policy.",
                            sources=_sources_from_search(sec_search))

    compliance_args = {
        "employee_id": employee_id,
        "scenario_type": "remote_work_request",
        "destination_state": destination_state,
        "destination_country": destination_country,
        "duration_calendar_days": duration_calendar_days,
    }
    compliance = await _call(client, trace, "check_policy_compliance", compliance_args)
    trace.record_tool_call(
        "check_policy_compliance", compliance_args, status="success",
        result_summary=f"status={compliance['status']}: {compliance['reasons'][0] if compliance['reasons'] else ''}",
        sources=_sources_from_compliance(compliance),
        escalation=(
            f"Requires: {', '.join(compliance['required_approvals'])}"
            if compliance.get("escalation_required")
            else None
        ),
    )

    destination = destination_country or destination_state or "the requested location"
    if compliance["status"] == "compliant":
        subject_intent = f"Temporary remote work from {destination}"
        action = "No special approval needed beyond standard manager notice; please confirm in Compass HR."
    else:
        subject_intent = f"Remote work request needs review: {destination}"
        action = "This request needs additional review before it can proceed; see reasons below."
    email_args = {
        "employee_id": employee_id,
        "recipient": profile["manager_name"] or "People & Culture Team",
        "subject_intent": subject_intent,
        "key_facts": compliance["reasons"],
        "requested_action": action,
    }
    draft = await _call(client, trace, "draft_hr_email", email_args)
    trace.record_tool_call("draft_hr_email", email_args, status="success",
                            result_summary=f"Draft prepared to {draft['to']}: '{draft['subject']}'. {draft['note']}")

    return trace.as_list(), compliance
