"""Turns a raw MCP tool result (already a plain dict — see
`mcp_client.MCPClient.call_tool`) into the three things the agent loop
needs for its trace and structured output: a short human-readable
summary, a list of citable evidence, and whether the result carries an
escalation.

This module knows the *shape* of each of the 8 tools' results (since
that's needed to render a readable trace line and to reliably extract
citations without ever trusting the model's own prose), but contains no
policy logic itself and never decides *which* tool to call — that
decision is the LLM's alone, made in `agent/orchestrator.py`.
"""
from __future__ import annotations

from agent.schemas import Citation


def summarize_tool_result(tool_name: str, result: dict) -> str:
    """A short, human-readable one-liner for the operational trace."""
    if tool_name == "search_policy_documents":
        n = len(result.get("results", []))
        docs = sorted({r["document_id"] for r in result.get("results", [])})
        return f"{n} chunk(s) retrieved" + (f" from {', '.join(docs)}." if docs else ".")
    if tool_name == "get_policy_section":
        return f"Retrieved {result.get('document_id')} §{result.get('section')}."
    if tool_name == "lookup_employee_profile":
        return f"{result.get('full_name')} ({result.get('employment_type')}, {result.get('department_name')})."
    if tool_name == "check_pto_balance":
        if result.get("has_tracked_balance"):
            return f"Balance: {result.get('balance_hours')} {result.get('unit', 'hours')} available."
        return result.get("note") or "No tracked PTO balance."
    if tool_name == "lookup_benefits_status":
        if result.get("has_record"):
            plan = result.get("medical_plan") or "no medical plan selected"
            return f"Benefits eligible: {result.get('benefits_eligible')}; medical plan: {plan}."
        return result.get("note") or "No benefits record on file."
    if tool_name == "check_policy_compliance":
        reasons = result.get("reasons") or []
        return f"status={result.get('status')}" + (f": {reasons[0]}" if reasons else "")
    if tool_name == "draft_hr_email":
        return f"Draft prepared to {result.get('to')}: '{result.get('subject')}'. {result.get('note', '')}"
    if tool_name == "create_mock_hr_ticket":
        return result.get("message", f"status={result.get('status')}")
    # Unknown/future tool: fall back to a generic, bounded summary rather
    # than assuming a shape.
    return str(result)[:200]


def citations_from_tool_result(tool_name: str, result: dict) -> list[Citation]:
    """Extract citable evidence, if any, from a tool result. Only ever
    reads fields the corresponding MCP tool actually returns (see
    `mcp_server/schemas.py`) — never invents a citation."""
    if tool_name == "search_policy_documents":
        return [
            Citation(
                document_id=hit["document_id"],
                document_title=hit["document_title"],
                section=hit["section"],
                snippet=hit["snippet"],
                source_path=hit.get("source_path"),
            )
            for hit in result.get("results", [])
        ]
    if tool_name == "get_policy_section":
        content = result.get("content", "")
        snippet = content if len(content) <= 220 else content[:220].rsplit(" ", 1)[0] + "..."
        return [
            Citation(
                document_id=result["document_id"],
                document_title=result["document_title"],
                section=result["section"],
                snippet=snippet,
                source_path=result.get("source_path"),
            )
        ]
    if tool_name == "check_policy_compliance":
        return [
            Citation(
                document_id=c["document_id"],
                document_title=c["document_title"],
                section=c["section"],
                snippet=c["snippet"],
            )
            for c in result.get("citations", [])
        ]
    return []


def escalation_from_tool_result(tool_name: str, result: dict) -> str | None:
    """A short escalation note for the trace, if this result carries one."""
    if tool_name == "check_policy_compliance" and result.get("escalation_required"):
        approvals = result.get("required_approvals") or []
        return f"Requires: {', '.join(approvals)}" if approvals else "Escalation required."
    return None


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Citation] = []
    for citation in citations:
        key = (citation.document_id, citation.section, citation.snippet[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped
