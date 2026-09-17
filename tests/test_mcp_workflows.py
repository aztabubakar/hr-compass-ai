"""Integration tests for the two deterministic, tool-level demo workflows.

These exercise `scripts.demo_workflows` end-to-end over real MCP calls,
proving the tool layer supports both required workflows and that the
remote-work workflow gathers evidence from more than one policy document.
"""
from __future__ import annotations

import anyio

from mcp_client.client import MCPClient
from scripts.demo_workflows import run_pto_workflow, run_remote_work_workflow


def test_pto_workflow_happy_path_produces_citation_bearing_compliant_result(
    mcp_server_env: dict[str, str],
) -> None:
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            return await run_pto_workflow(client, "EMP-1001", requested_days=3, notice_business_days=10)

    trace, compliance = anyio.run(body)

    tool_sequence = [step["tool"] for step in trace if step["type"] == "tool_call"]
    assert tool_sequence == [
        "lookup_employee_profile",
        "check_pto_balance",
        "search_policy_documents",
        "check_policy_compliance",
        "draft_hr_email",
    ]
    assert all(step["status"] == "success" for step in trace)
    assert compliance["status"] == "compliant"
    assert compliance["citations"]
    assert compliance["escalation_required"] is False


def test_pto_workflow_edge_case_insufficient_balance_escalates(mcp_server_env: dict[str, str]) -> None:
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            # EMP-1009 has only a 4-hour balance.
            return await run_pto_workflow(client, "EMP-1009", requested_days=3, notice_business_days=10)

    trace, compliance = anyio.run(body)
    assert compliance["status"] == "non_compliant"
    # The trace still completes all steps (drafting an escalation email),
    # it just reflects the non-compliant outcome rather than failing.
    assert [s["tool"] for s in trace if s["type"] == "tool_call"][-1] == "draft_hr_email"


def test_remote_work_workflow_interstate_gathers_multi_document_evidence(
    mcp_server_env: dict[str, str],
) -> None:
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            return await run_remote_work_workflow(
                client, "EMP-1002", destination_state="Colorado", duration_calendar_days=42
            )

    trace, compliance = anyio.run(body)

    tool_sequence = [step["tool"] for step in trace if step["type"] == "tool_call"]
    assert tool_sequence == [
        "lookup_employee_profile",
        "search_policy_documents",
        "search_policy_documents",
        "search_policy_documents",
        "check_policy_compliance",
        "draft_hr_email",
    ]
    assert compliance["status"] == "requires_approval"
    assert compliance["escalation_required"] is True

    all_sources: set[str] = set()
    for step in trace:
        for source in step["sources"]:
            all_sources.add(source.split(" §")[0])
    assert len(all_sources) >= 2, f"expected multi-policy evidence across the workflow, got {all_sources}"


def test_remote_work_workflow_international_gathers_multi_document_evidence(
    mcp_server_env: dict[str, str],
) -> None:
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            return await run_remote_work_workflow(
                client, "EMP-1003", destination_country="Portugal", duration_calendar_days=42
            )

    trace, compliance = anyio.run(body)

    assert compliance["status"] == "requires_approval"
    assert "Global Mobility review" in compliance["required_approvals"]

    citation_doc_ids = {c["document_id"] for c in compliance["citations"]}
    # The international scenario should surface evidence spanning remote
    # work, work-location, and security policies.
    assert {"POL-LOC-001"} <= citation_doc_ids
    assert len(citation_doc_ids) >= 2


def test_remote_work_workflow_unsupported_country_is_non_compliant(mcp_server_env: dict[str, str]) -> None:
    async def body():
        async with MCPClient(env=mcp_server_env) as client:
            return await run_remote_work_workflow(
                client, "EMP-1003", destination_country="Thailand", duration_calendar_days=42
            )

    _trace, compliance = anyio.run(body)
    assert compliance["status"] == "non_compliant"
