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


class AgentResponse(BaseModel):
    """The complete result of one agent turn."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    requires_clarification: bool = False
    requires_confirmation: bool = False
    escalated: bool = False
