"""Typed structured output for one agent turn."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """One piece of policy evidence backing the answer. Always built by the
    agent from an actual MCP tool result (`search_policy_documents`,
    `get_policy_section`, or `check_policy_compliance`'s citations) —
    never parsed or inferred from the model's own prose, so a citation
    appearing here is always traceable to a real retrieved chunk."""

    document_id: str
    document_title: str
    section: str
    snippet: str
    source_path: str | None = None
    # Which MCP tool produced this citation ("search_policy_documents",
    # "get_policy_section", or "check_policy_compliance") -- never read by
    # the UI or by scoring, only by agent/tool_results.py's
    # attribute_citations for its exact-section-authority rule (a citation
    # from get_policy_section names a section the model explicitly asked
    # for by name, which is stronger evidence of intentional use than a
    # section that merely turned up in a broader search_policy_documents
    # top-k). Optional/defaulted so existing callers that construct a
    # Citation without it (tests, any future tool) remain valid.
    source_tool: str | None = None


class AgentResponse(BaseModel):
    """The complete result of one agent turn."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    requires_clarification: bool = False
    requires_confirmation: bool = False
    escalated: bool = False
