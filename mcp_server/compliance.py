"""Deterministic policy-compliance rule evaluation.

**Design principle: this module must never hallucinate a policy rule.**
`check_policy_compliance` produces its `status`/`reasons`/`required_approvals`
from one of exactly two sources, and the result's `rule_basis` field says
which:

1. `"deterministic_rule"` — a fixed, hand-coded rule table below, derived
   directly from the specific numeric thresholds written into the Solara
   Dynamics policy corpus (`policies/content.py`): PTO accrual/balance
   arithmetic, the "Work From Anywhere" 20-business-day allowance, the
   interstate 30/90-day approval tiers, and the international 30-business-
   day EOR allowance. These thresholds are *citable facts from the
   policies*, encoded once here so every compliance check applies them
   consistently instead of re-deriving them ad hoc (and, before this tool
   exists, instead of an LLM re-reading policy prose and possibly getting
   the arithmetic wrong).
2. `"insufficient_evidence"` — returned instead of a rule result whenever
   this module cannot find supporting retrieved policy evidence for the
   scenario (RAG returns no chunk above `RagConfig.min_relevance_score`),
   or the caller didn't supply enough structured facts to select a rule
   branch. The deterministic rule table is *never* applied without at
   least one corroborating citation attached to the result.

What this module is explicitly **not**: a natural-language interpreter of
arbitrary HR questions. It accepts a small, fixed set of typed scenario
inputs (see `mcp_server.tools.check_policy_compliance`'s signature) and
returns a rule-table lookup. Interpreting a free-form employee question
into these structured inputs, and turning a `ComplianceResult` into
prose, is explicitly deferred to the future LLM agent (Phase 3) — see
`design-and-evaluation.md`'s MCP Architecture section, "Facts vs. rules vs.
future LLM interpretation."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mock_data.models import Employee, PTOBalance
from rag.retrieval import PolicyRetriever, get_default_retriever
from mcp_server.schemas import ComplianceCitation, ComplianceStatus

# --- Rule constants, each traceable to a specific policy section ----------
# POL-LOC-001 Section 3 "Payroll-Registered States".
PAYROLL_REGISTERED_STATES = {
    "texas", "colorado", "north carolina", "california", "new york",
    "washington", "illinois", "georgia", "florida", "massachusetts",
    "virginia", "arizona", "oregon", "tennessee",
}
# POL-LOC-001 Section 6 "International Remote Work".
APPROVED_EOR_COUNTRIES = {"united kingdom", "uk", "canada", "portugal", "ireland"}
# POL-RWK-001 Section 6 "Temporary Remote Work ('Work From Anywhere')".
WORK_FROM_ANYWHERE_LIMIT_BUSINESS_DAYS = 20
# POL-LOC-001 Section 6 "International Remote Work" (6 consecutive weeks).
INTERNATIONAL_ALLOWANCE_BUSINESS_DAYS = 30
# Approximation of POL-LOC-001 Section 4's 90-*calendar*-day interstate
# approval-tier boundary, expressed in business days for consistency with
# the other duration inputs this tool accepts (~5 business days/week).
# Documented simplification: the source policy's interstate thresholds are
# in calendar days while this tool's `duration_business_days` input is not;
# treating a full LLM-grade calendar/business-day reconciliation is left to
# Phase 3, per this module's docstring.
INTERSTATE_RELOCATION_TIER_BUSINESS_DAYS = 65


@dataclass
class RuleOutcome:
    status: ComplianceStatus
    reasons: list[str]
    required_approvals: list[str] = field(default_factory=list)
    escalation_required: bool = False


def evaluate_pto_request(
    employee: Employee,
    balance: PTOBalance | None,
    requested_days: int | None,
    notice_business_days: int | None,
) -> RuleOutcome:
    if employee.status != "active":
        return RuleOutcome(
            status="non_compliant",
            reasons=[f"Employee status is '{employee.status}', not active; PTO cannot be scheduled."],
        )

    if balance is None:
        if employee.employment_type in ("contractor", "intern", "eor_international"):
            return RuleOutcome(
                status="non_compliant",
                reasons=[
                    f"Employment type '{employee.employment_type}' is not covered by the PTO policy "
                    "(POL-PTO-001 Section 9); no company PTO balance applies."
                ],
            )
        return RuleOutcome(
            status="requires_approval",
            reasons=[
                "This employee has no accrued PTO balance because Director-level and above "
                "employees participate in the Flexible PTO Plan (POL-PTO-001 Section 4), which is "
                "taken as needed subject to manager approval rather than checked against a balance."
            ],
            required_approvals=["Manager approval"],
            escalation_required=True,
        )

    if requested_days is None:
        return RuleOutcome(
            status="insufficient_evidence",
            reasons=["requested_days was not provided; cannot evaluate balance sufficiency."],
        )

    requested_hours = requested_days * 8
    reasons: list[str] = []
    required_approvals: list[str] = []
    escalation_required = False

    if balance.balance_hours < requested_hours:
        status: ComplianceStatus = "non_compliant"
        reasons.append(
            f"Requested {requested_days} day(s) = {requested_hours:.1f} hours, but the current "
            f"balance is only {balance.balance_hours:.1f} hours (POL-PTO-001 Section 3)."
        )
    else:
        status = "compliant"
        reasons.append(
            f"Requested {requested_days} day(s) = {requested_hours:.1f} hours is within the "
            f"current balance of {balance.balance_hours:.1f} hours (POL-PTO-001 Section 3)."
        )
        required_notice = 5 if requested_days <= 2 else 10 if requested_days <= 5 else 30
        if notice_business_days is None:
            reasons.append(
                f"POL-PTO-001 Section 5 requires at least {required_notice} business day(s) of "
                "advance notice for a request of this length; notice_business_days was not "
                "provided, so notice compliance was not evaluated."
            )
        elif notice_business_days < required_notice:
            status = "requires_approval"
            required_approvals.append("Manager approval (short-notice exception)")
            escalation_required = True
            reasons.append(
                f"Only {notice_business_days} business day(s) of notice was given; POL-PTO-001 "
                f"Section 5 calls for {required_notice}, so this needs manager discretion as "
                "short-notice/emergency PTO rather than standard approval."
            )
        else:
            reasons.append(
                f"{notice_business_days} business day(s) of notice satisfies the "
                f"{required_notice}-business-day requirement in POL-PTO-001 Section 5."
            )

    if employee.department_id == "DEPT-FIN":
        reasons.append(
            "Employee is in Finance: POL-PTO-001 Section 6 prohibits PTO during the last 5 "
            "business days of a fiscal quarter (quarter-close blackout). Verify the requested "
            "dates against the current blackout window before final approval."
        )

    return RuleOutcome(status=status, reasons=reasons, required_approvals=required_approvals,
                        escalation_required=escalation_required)


def evaluate_remote_work_request(
    destination_state: str | None,
    destination_country: str | None,
    duration_business_days: int | None,
) -> RuleOutcome:
    is_international = bool(destination_country) and destination_country.strip().lower() not in (
        "", "us", "usa", "united states",
    )

    if is_international:
        country_norm = destination_country.strip().lower()
        if country_norm not in APPROVED_EOR_COUNTRIES:
            return RuleOutcome(
                status="non_compliant",
                reasons=[
                    f"'{destination_country}' is not on Solara Dynamics' approved Employer-of-Record "
                    "country list (United Kingdom, Canada, Portugal, Ireland) per POL-LOC-001 "
                    "Section 6; international remote work there is not supported."
                ],
                escalation_required=True,
            )
        if duration_business_days is not None and duration_business_days > INTERNATIONAL_ALLOWANCE_BUSINESS_DAYS:
            return RuleOutcome(
                status="non_compliant",
                reasons=[
                    f"Requested duration ({duration_business_days} business days) exceeds the "
                    f"{INTERNATIONAL_ALLOWANCE_BUSINESS_DAYS}-business-day (6-week) annual "
                    f"international remote work allowance for {destination_country} per "
                    "POL-LOC-001 Section 6."
                ],
                escalation_required=True,
            )
        return RuleOutcome(
            status="requires_approval",
            reasons=[
                f"International remote work to {destination_country} (an approved "
                "Employer-of-Record country) is within the 6-week/30-business-day allowance but "
                "always requires Global Mobility review, at least 20 business days' notice, and a "
                "signed International Remote Work Attestation per POL-LOC-001 Section 6."
            ],
            required_approvals=["Global Mobility review", "International Remote Work Attestation"],
            escalation_required=True,
        )

    if destination_state:
        state_norm = destination_state.strip().lower()
        if state_norm not in PAYROLL_REGISTERED_STATES:
            return RuleOutcome(
                status="requires_approval",
                reasons=[
                    f"'{destination_state}' is not currently a Solara Dynamics payroll-registered "
                    "state; per POL-LOC-001 Section 4/5 this is treated as a permanent relocation "
                    "request requiring VP and Payroll approval and possibly new state registration."
                ],
                required_approvals=["VP approval", "Payroll registration review"],
                escalation_required=True,
            )
        if duration_business_days is None:
            return RuleOutcome(
                status="insufficient_evidence",
                reasons=["duration_business_days was not provided; cannot determine which "
                         "approval tier applies under POL-LOC-001 Section 4."],
            )
        if duration_business_days <= WORK_FROM_ANYWHERE_LIMIT_BUSINESS_DAYS:
            return RuleOutcome(
                status="compliant",
                reasons=[
                    f"{duration_business_days} business days in {destination_state} (a "
                    f"payroll-registered state) is within the "
                    f"{WORK_FROM_ANYWHERE_LIMIT_BUSINESS_DAYS}-business-day 'Work From Anywhere' "
                    "allowance per POL-RWK-001 Section 6; manager notice in Compass HR is "
                    "sufficient."
                ],
            )
        if duration_business_days <= INTERSTATE_RELOCATION_TIER_BUSINESS_DAYS:
            return RuleOutcome(
                status="requires_approval",
                reasons=[
                    f"{duration_business_days} business days in {destination_state} exceeds the "
                    f"{WORK_FROM_ANYWHERE_LIMIT_BUSINESS_DAYS}-business-day no-approval allowance; "
                    "Payroll and People & Culture Team approval is required at least 15 business "
                    "days in advance per POL-LOC-001 Section 4."
                ],
                required_approvals=["Payroll approval", "People & Culture Team approval"],
                escalation_required=True,
            )
        return RuleOutcome(
            status="requires_approval",
            reasons=[
                f"{duration_business_days} business days in {destination_state} is long enough to "
                "be treated as a permanent relocation request per POL-LOC-001 Section 4/5, "
                "requiring VP and People & Culture Team approval."
            ],
            required_approvals=["VP approval", "People & Culture Team approval"],
            escalation_required=True,
        )

    return RuleOutcome(
        status="insufficient_evidence",
        reasons=["Neither destination_state nor destination_country was provided."],
    )


def gather_citations(
    queries: list[str],
    retriever: PolicyRetriever | None = None,
    top_k: int = 2,
) -> tuple[list[ComplianceCitation], bool]:
    """Run each query through real RAG retrieval and collect deduplicated
    citations. Returns (citations, any_query_had_sufficient_evidence).
    """
    retriever = retriever or get_default_retriever()
    citations: list[ComplianceCitation] = []
    seen_chunk_ids: set[str] = set()
    any_sufficient = False

    for query in queries:
        response = retriever.query(query, top_k=top_k)
        if response.has_sufficient_evidence:
            any_sufficient = True
        for result in response.results:
            if result.chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk.chunk_id)
            citations.append(
                ComplianceCitation(
                    document_id=result.doc_id,
                    document_title=result.title,
                    section=result.section,
                    snippet=result.snippet,
                    score=result.score,
                )
            )

    return citations, any_sufficient
