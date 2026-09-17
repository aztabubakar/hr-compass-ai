"""Per-tool tests, all made through the real MCP protocol (never by
importing `mcp_server.tools` functions directly).
"""
from __future__ import annotations

import json

import anyio
import pytest

from mcp_client.client import MCPClient, MCPToolError


async def _call(client: MCPClient, tool: str, arguments: dict) -> dict:
    return await client.call_tool(tool, arguments)


# ---------------------------------------------------------------------------
# search_policy_documents
# ---------------------------------------------------------------------------
def test_search_policy_documents_returns_citable_evidence(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "search_policy_documents",
                                  {"query": "PTO carryover accrual balance", "top_k": 3})
            assert result["results"], "expected at least one retrieved chunk"
            hit = result["results"][0]
            assert {"document_id", "document_title", "section", "source_path", "snippet", "score"} <= hit.keys()

    anyio.run(body)


def test_search_policy_documents_multi_document_case(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(
                client, "search_policy_documents",
                {"query": "remote work interstate international payroll registered states", "top_k": 6},
            )
            doc_ids = {hit["document_id"] for hit in result["results"]}
            assert len(doc_ids) >= 2, f"expected evidence from multiple documents, got {doc_ids}"

    anyio.run(body)


def test_search_policy_documents_filters_by_document_id(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(
                client, "search_policy_documents",
                {"query": "policy", "top_k": 5, "document_id": "POL-BEN-001"},
            )
            assert all(hit["document_id"] == "POL-BEN-001" for hit in result["results"])

    anyio.run(body)


# ---------------------------------------------------------------------------
# get_policy_section
# ---------------------------------------------------------------------------
def test_get_policy_section_returns_content_and_citation(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "get_policy_section",
                                  {"document_id": "POL-PTO-001", "section": "Carryover and Payout Rules"})
            assert result["document_id"] == "POL-PTO-001"
            assert "carryover" in result["content"].lower() or "carry over" in result["content"].lower() \
                or "forfeit" in result["content"].lower()
            assert result["source_path"]

    anyio.run(body)


def test_get_policy_section_unknown_document_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError, match="POL-ZZZ-999"):
                await _call(client, "get_policy_section", {"document_id": "POL-ZZZ-999", "section": "Anything"})

    anyio.run(body)


def test_get_policy_section_unknown_section_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError, match="Available sections"):
                await _call(client, "get_policy_section",
                             {"document_id": "POL-PTO-001", "section": "Not A Real Section"})

    anyio.run(body)


# ---------------------------------------------------------------------------
# lookup_employee_profile
# ---------------------------------------------------------------------------
def test_lookup_employee_profile_returns_expected_fields(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "lookup_employee_profile", {"employee_id": "EMP-1001"})
            assert result["employee_id"] == "EMP-1001"
            assert result["full_name"] == "Jordan Vance"
            assert result["employment_type"] == "full_time"
            # Deliberately minimized: no email/work-authorization fields.
            assert "email" not in result
            assert "work_authorization" not in result

    anyio.run(body)


def test_lookup_employee_profile_invalid_id_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError, match="EMP-9999"):
                await _call(client, "lookup_employee_profile", {"employee_id": "EMP-9999"})

    anyio.run(body)


# ---------------------------------------------------------------------------
# check_pto_balance
# ---------------------------------------------------------------------------
def test_check_pto_balance_happy_path(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "check_pto_balance", {"employee_id": "EMP-1001"})
            assert result["has_tracked_balance"] is True
            assert result["balance_hours"] >= 24.0
            assert result["unit"] == "hours"

    anyio.run(body)


def test_check_pto_balance_flexible_plan_employee_has_no_balance(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            # EMP-1007 is a Director (Flexible PTO plan) - a known employee
            # with a legitimately absent balance, not an error.
            result = await _call(client, "check_pto_balance", {"employee_id": "EMP-1007"})
            assert result["has_tracked_balance"] is False
            assert result["plan_type"] == "flexible"
            assert result["note"]

    anyio.run(body)


def test_check_pto_balance_invalid_employee_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError, match="EMP-0000"):
                await _call(client, "check_pto_balance", {"employee_id": "EMP-0000"})

    anyio.run(body)


# ---------------------------------------------------------------------------
# lookup_benefits_status
# ---------------------------------------------------------------------------
def test_lookup_benefits_status_happy_path(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "lookup_benefits_status", {"employee_id": "EMP-1001"})
            assert result["has_record"] is True
            assert result["benefits_eligible"] is True
            assert result["medical_plan"]

    anyio.run(body)


def test_lookup_benefits_status_no_record_employee(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            # EMP-1005 is a contractor: known employee, no benefits record.
            result = await _call(client, "lookup_benefits_status", {"employee_id": "EMP-1005"})
            assert result["has_record"] is False
            assert result["benefits_eligible"] is False
            assert result["note"]

    anyio.run(body)


def test_lookup_benefits_status_invalid_employee_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError):
                await _call(client, "lookup_benefits_status", {"employee_id": "EMP-8888"})

    anyio.run(body)


# ---------------------------------------------------------------------------
# check_policy_compliance
# ---------------------------------------------------------------------------
def test_compliance_pto_happy_path_is_compliant(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "check_policy_compliance", {
                "employee_id": "EMP-1001", "scenario_type": "pto_request",
                "requested_days": 3, "notice_business_days": 10,
            })
            assert result["status"] == "compliant"
            assert result["citations"]
            assert result["rule_basis"] == "deterministic_rule"

    anyio.run(body)


def test_compliance_pto_insufficient_balance_is_non_compliant(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            # EMP-1009 has a 4-hour balance; 3 days = 24 hours requested.
            result = await _call(client, "check_policy_compliance", {
                "employee_id": "EMP-1009", "scenario_type": "pto_request",
                "requested_days": 3, "notice_business_days": 10,
            })
            assert result["status"] == "non_compliant"

    anyio.run(body)


def test_compliance_remote_work_interstate_requires_approval(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "check_policy_compliance", {
                "employee_id": "EMP-1002", "scenario_type": "remote_work_request",
                "destination_state": "Colorado", "duration_calendar_days": 42,
            })
            assert result["status"] == "requires_approval"
            assert result["escalation_required"] is True
            doc_ids = {c["document_id"] for c in result["citations"]}
            assert len(doc_ids) >= 2  # multi-document evidence

    anyio.run(body)


def test_compliance_remote_work_international_requires_approval(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "check_policy_compliance", {
                "employee_id": "EMP-1003", "scenario_type": "remote_work_request",
                "destination_country": "Portugal", "duration_calendar_days": 42,
            })
            assert result["status"] == "requires_approval"
            assert "Global Mobility review" in result["required_approvals"]

    anyio.run(body)


def test_compliance_remote_work_unsupported_country_is_non_compliant(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "check_policy_compliance", {
                "employee_id": "EMP-1003", "scenario_type": "remote_work_request",
                "destination_country": "Thailand", "duration_calendar_days": 42,
            })
            assert result["status"] == "non_compliant"

    anyio.run(body)


def test_compliance_invalid_scenario_type_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError, match="scenario_type"):
                await _call(client, "check_policy_compliance", {
                    "employee_id": "EMP-1001", "scenario_type": "not_a_real_scenario",
                })

    anyio.run(body)


# ---------------------------------------------------------------------------
# draft_hr_email
# ---------------------------------------------------------------------------
def test_draft_hr_email_is_never_sent(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "draft_hr_email", {
                "employee_id": "EMP-1001", "recipient": "Devon Marsh",
                "subject_intent": "Test draft", "key_facts": ["fact one"],
                "requested_action": "please review",
            })
            assert result["is_draft"] is True
            assert "has been sent" in result["note"].lower()
            assert result["subject"] and result["body"]

    anyio.run(body)


# ---------------------------------------------------------------------------
# create_mock_hr_ticket (safety/confirmation behavior)
# ---------------------------------------------------------------------------
def test_create_mock_ticket_preview_does_not_persist(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            result = await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1001", "category": "PTO", "summary": "Test preview ticket",
            })
            assert result["status"] == "preview"
            assert result["ticket_id"] is None

    anyio.run(body)


def test_create_mock_ticket_requires_explicit_confirmation(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            preview = await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1001", "category": "PTO", "summary": "Should stay a preview",
                "confirmed": False,
            })
            assert preview["status"] == "preview"
            assert preview["ticket_id"] is None

    anyio.run(body)


def test_create_mock_ticket_confirmed_creates_and_returns_id(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            created = await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1001", "category": "PTO", "summary": "Confirmed test ticket",
                "confirmed": True,
            })
            assert created["status"] == "created"
            assert created["ticket_id"] and created["ticket_id"].startswith("TCK-")

    anyio.run(body)


def test_create_mock_ticket_preview_never_writes_runtime_file(mcp_server_env: dict[str, str]) -> None:
    from pathlib import Path

    runtime_path = Path(mcp_server_env["HR_COMPASS_RUNTIME_TICKETS_PATH"])

    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1001", "category": "PTO", "summary": "Should not persist",
                "confirmed": False,
            })

    anyio.run(body)
    assert not runtime_path.exists(), "a preview-only ticket call must not write any runtime state"


def test_create_mock_ticket_confirmed_appends_exactly_one_record(mcp_server_env: dict[str, str]) -> None:
    from pathlib import Path

    runtime_path = Path(mcp_server_env["HR_COMPASS_RUNTIME_TICKETS_PATH"])

    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1001", "category": "PTO", "summary": "First confirmed ticket",
                "confirmed": True,
            })
            await _call(client, "create_mock_hr_ticket", {
                "employee_id": "EMP-1002", "category": "Benefits", "summary": "Second confirmed ticket",
                "confirmed": True,
            })

    anyio.run(body)
    records = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["ticket_id"] != records[1]["ticket_id"]


def test_create_mock_ticket_invalid_employee_is_clean_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError):
                await _call(client, "create_mock_hr_ticket", {
                    "employee_id": "EMP-7777", "category": "PTO", "summary": "x", "confirmed": True,
                })

    anyio.run(body)


def test_malformed_arguments_return_clean_validation_error(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            # Missing the required 'employee_id' argument entirely.
            with pytest.raises(MCPToolError):
                await _call(client, "check_pto_balance", {})

    anyio.run(body)
