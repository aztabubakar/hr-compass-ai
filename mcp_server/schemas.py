"""Typed JSON-compatible input/output schemas for every MCP tool.

Every tool returns a `pydantic.BaseModel` (never a bare `dict`) because the
MCP SDK only auto-generates a structured `output_schema` and populates
`CallToolResult.structured_content` for typed return values — a bare
`dict` return produces text-only output with no machine-readable schema
(verified empirically against `mcp` 2.2.0 before writing this module). A
typed model also documents, in one place, exactly what each tool promises
to return.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# search_policy_documents
# ---------------------------------------------------------------------------
class PolicySearchHit(BaseModel):
    document_id: str
    document_title: str
    section: str
    source_path: str
    snippet: str
    score: float


class PolicySearchResult(BaseModel):
    query: str
    results: list[PolicySearchHit]
    has_sufficient_evidence: bool


# ---------------------------------------------------------------------------
# get_policy_section
# ---------------------------------------------------------------------------
class PolicySectionResult(BaseModel):
    document_id: str
    document_title: str
    effective_date: str
    section: str
    content: str
    source_path: str


# ---------------------------------------------------------------------------
# lookup_employee_profile
# ---------------------------------------------------------------------------
class EmployeeProfile(BaseModel):
    employee_id: str
    full_name: str
    title: str
    department_id: str
    department_name: str
    manager_id: str | None
    manager_name: str | None
    employment_type: str
    work_arrangement: str
    home_location_id: str
    home_location: str
    employment_country: str
    hire_date: str
    status: str


# ---------------------------------------------------------------------------
# check_pto_balance
# ---------------------------------------------------------------------------
class PTOBalanceResult(BaseModel):
    employee_id: str
    has_tracked_balance: bool
    plan_type: str  # "accrual_based" | "flexible" | "not_covered"
    balance_hours: float | None
    unit: str = "hours"
    accrued_ytd_hours: float | None = None
    used_ytd_hours: float | None = None
    pending_hours: float | None = None
    carryover_hours_from_prior_year: float | None = None
    as_of_date: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# lookup_benefits_status
# ---------------------------------------------------------------------------
class BenefitsStatusResult(BaseModel):
    employee_id: str
    has_record: bool
    benefits_eligible: bool | None
    medical_plan: str | None = None
    dental_enrolled: bool | None = None
    vision_enrolled: bool | None = None
    retirement_401k_enrolled: bool | None = None
    retirement_401k_contribution_pct: float | None = None
    fsa_healthcare_enrolled: bool | None = None
    fsa_dependent_care_enrolled: bool | None = None
    wellness_stipend_eligible: bool | None = None
    waiting_period_end_date: str | None = None
    coverage_effective_date: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# check_policy_compliance
# ---------------------------------------------------------------------------
ComplianceStatus = Literal["compliant", "non_compliant", "requires_approval", "insufficient_evidence"]
RuleBasis = Literal["deterministic_rule", "insufficient_evidence"]


class ComplianceCitation(BaseModel):
    document_id: str
    document_title: str
    section: str
    snippet: str
    score: float


class ComplianceResult(BaseModel):
    status: ComplianceStatus
    reasons: list[str]
    citations: list[ComplianceCitation]
    required_approvals: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    facts_considered: list[str] = Field(default_factory=list)
    rule_basis: RuleBasis


# ---------------------------------------------------------------------------
# draft_hr_email
# ---------------------------------------------------------------------------
class DraftEmailResult(BaseModel):
    to: str
    subject: str
    body: str
    is_draft: bool = True
    note: str = "This is a draft only. No email has been sent."


# ---------------------------------------------------------------------------
# create_mock_hr_ticket
# ---------------------------------------------------------------------------
TicketActionStatus = Literal["preview", "created"]


class MockTicketResult(BaseModel):
    status: TicketActionStatus
    ticket_id: str | None
    employee_id: str
    category: str
    summary: str
    details: str
    priority: str
    message: str
