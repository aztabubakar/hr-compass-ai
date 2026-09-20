"""Turns a raw MCP tool result (already a plain dict — see
`mcp_client.MCPClient.call_tool`) into the things the agent loop needs for
its trace, structured output, and the conversation it sends the LLM: a
short human-readable summary, a list of citable evidence (filtered for
relevance — see `_filter_relevant_hits`), a trimmed copy safe to feed back
to the model, and whether the result carries an escalation.

This module knows the *shape* of each of the 8 tools' results (since
that's needed to render a readable trace line and to reliably extract
citations without ever trusting the model's own prose), but contains no
policy logic itself and never decides *which* tool to call — that
decision is the LLM's alone, made in `agent/orchestrator.py`.
"""
from __future__ import annotations

from agent.schemas import Citation

# How much lower than the single best-scoring hit in one search_policy_documents
# call a hit's score may be and still count as genuinely relevant evidence,
# not just whatever filled out the remaining top-k slots. Relative (not an
# absolute score) so it adapts to whatever range the configured embedding
# backend happens to produce; the single best hit always survives (its own
# ratio to itself is 1.0), so this can never drop every result down to zero.
#
# This is deliberately a general "how much noise is in one ranked list"
# heuristic, not a PTO-specific or topic-specific rule -- the exact same
# threshold applies to every search_policy_documents call regardless of
# what it searched for, which is what keeps this a generic filter rather
# than hard-coded topic branching. Genuine multi-document evidence is not
# lost by this: this project's demo workflows (and the tests that prove
# them) already retrieve multi-document evidence via *separate*
# document_id-scoped search_policy_documents/get_policy_section calls, and
# a separate call's own top hit always clears its own bar.
CITATION_RELATIVE_SCORE_THRESHOLD = 0.75


def _filter_relevant_hits(hits: list[dict], relative_threshold: float = CITATION_RELATIVE_SCORE_THRESHOLD) -> list[dict]:
    """Keep only the hits from one ranked `search_policy_documents` result
    that are genuinely comparable in relevance to the single best match,
    not just whatever else happened to fill out the remaining top-k slots
    (e.g. an unrelated equipment or holiday-schedule chunk turning up
    alongside a PTO-relevant one). Hits are assumed already ranked
    best-first, as the retriever returns them."""
    if not hits:
        return []
    top_score = hits[0].get("score", 0.0)
    if top_score <= 0:
        return hits[:1]
    return [hit for hit in hits if hit.get("score", 0.0) >= top_score * relative_threshold]


def trim_tool_result_for_llm(tool_name: str, result: dict) -> dict:
    """Return a copy of `result` with fields the model doesn't need for
    reasoning/synthesis removed, before it's added to the conversation
    history that gets resent, in full, on every subsequent turn. This is
    what keeps a request with several tool calls from accumulating an
    unnecessarily large context: `source_path` is a filesystem path the
    model never needs to read or cite (citation source paths are already
    captured server-side via `citations_from_tool_result`, called with the
    *original*, untrimmed result), and a `search_policy_documents` hit
    list is narrowed to the same relevance-filtered subset
    `citations_from_tool_result` uses, so the model reasons over (and the
    conversation carries the token cost of) only the evidence that's
    actually likely to matter.

    The full, untrimmed result is still what `summarize_tool_result`,
    `citations_from_tool_result`, and the operational trace use — this
    function only affects what accumulates in the LLM's own context.
    """
    if tool_name == "search_policy_documents":
        relevant_hits = _filter_relevant_hits(result.get("results", []))
        trimmed_hits = [{k: v for k, v in hit.items() if k != "source_path"} for hit in relevant_hits]
        return {**result, "results": trimmed_hits}
    if tool_name == "get_policy_section":
        return {k: v for k, v in result.items() if k != "source_path"}
    return result


def summarize_tool_result(tool_name: str, result: dict) -> str:
    """A short, human-readable one-liner for the operational trace."""
    if tool_name == "search_policy_documents":
        all_hits = result.get("results", [])
        relevant_hits = _filter_relevant_hits(all_hits)
        docs = sorted({hit["document_id"] for hit in relevant_hits})
        if len(relevant_hits) == len(all_hits):
            return f"{len(all_hits)} chunk(s) retrieved" + (f" from {', '.join(docs)}." if docs else ".")
        return (
            f"{len(all_hits)} chunk(s) retrieved, {len(relevant_hits)} judged relevant"
            + (f" from {', '.join(docs)}." if docs else ".")
        )
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
    `mcp_server/schemas.py`) — never invents a citation.

    For `search_policy_documents`, only relevance-filtered hits become
    citations (`_filter_relevant_hits`) — the UI's citation panel should
    reflect evidence that plausibly supports the answer, not every raw
    top-k chunk the retriever happened to return alongside it (e.g. an
    equipment or holiday-schedule chunk that scored far below the actual
    match for an unrelated query). `get_policy_section` and
    `check_policy_compliance` are unaffected: a section fetch already
    names one exact section, and `check_policy_compliance`'s citations
    come from the tool's own small set of targeted, scenario-specific
    retrieval queries (`mcp_server.compliance.gather_citations`), not a
    single broad top-k dump.
    """
    if tool_name == "search_policy_documents":
        return [
            Citation(
                document_id=hit["document_id"],
                document_title=hit["document_title"],
                section=hit["section"],
                snippet=hit["snippet"],
                source_path=hit.get("source_path"),
            )
            for hit in _filter_relevant_hits(result.get("results", []))
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
