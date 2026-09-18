"""The 8 MCP tools, registered on the shared `mcp_server.app.server`.

Every tool here is a thin wrapper: it validates/looks up inputs, calls
straight into the existing Phase 1 modules (`rag.retrieval`, `rag.parsers`,
`mock_data.loader`) or the Phase 2 helper modules in this package
(`compliance.py`, `tickets.py`), and shapes the result into one of the
typed models in `schemas.py`. No retrieval, parsing, or mock-data logic is
re-implemented here.

Expected/anticipated failures (unknown employee ID, unknown document ID,
unknown section, the RAG index not having been built yet) are raised as
`mcp.server.mcpserver.exceptions.ToolError` with a specific, safe message;
the MCP SDK turns that into a clean `CallToolResult(is_error=True, ...)`
for the client with no Python traceback attached (verified empirically —
see design-and-evaluation.md). A record that is simply *absent* for a
legitimate business reason (a Director has no tracked PTO balance; a
contractor has no benefits record) is **not** an error — those tools
return a normal, successful result whose fields explain why.
"""
from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import compliance, tickets
from mcp_server.app import server
from mcp_server.dataset import get_dataset, get_parsed_docs
from mcp_server.schemas import (
    BenefitsStatusResult,
    ComplianceResult,
    DraftEmailResult,
    EmployeeProfile,
    MockTicketResult,
    PolicySearchHit,
    PolicySearchResult,
    PolicySectionResult,
    PTOBalanceResult,
)


# ---------------------------------------------------------------------------
# 1. search_policy_documents
# ---------------------------------------------------------------------------
@server.tool()
def search_policy_documents(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> PolicySearchResult:
    """Search the Solara Dynamics policy corpus and return ranked, citable
    evidence (document ID/title, section, snippet, source path, score).

    Uses the Phase 1 RAG retrieval pipeline (`rag.retrieval`) directly; this
    tool does not re-implement embedding or vector search. If `document_id`
    is given, results are filtered to that document only (over-fetching
    first so filtering doesn't starve the result set).
    """
    from rag.retrieval import get_default_retriever
    from rag.vector_store import VectorStoreError

    try:
        retriever = get_default_retriever()
    except VectorStoreError as exc:
        raise ToolError(
            "The policy search index has not been built yet. Run `python -m rag.ingest` "
            f"(details: {exc})"
        ) from exc

    fetch_k = top_k * 4 if document_id else top_k
    response = retriever.query(query, top_k=fetch_k)
    results = response.results
    if document_id:
        results = [r for r in results if r.doc_id == document_id][:top_k]

    hits = [
        PolicySearchHit(
            document_id=r.doc_id,
            document_title=r.title,
            section=r.section,
            source_path=r.chunk.source_path,
            snippet=r.snippet,
            score=r.score,
        )
        for r in results
    ]
    has_sufficient_evidence = bool(hits) and hits[0].score >= retriever.config.min_relevance_score
    return PolicySearchResult(query=query, results=hits, has_sufficient_evidence=has_sufficient_evidence)


# ---------------------------------------------------------------------------
# 2. get_policy_section
# ---------------------------------------------------------------------------
@server.tool()
def get_policy_section(document_id: str, section: str) -> PolicySectionResult:
    """Fetch one named section's full text from a specific policy document,
    with stable citation metadata. Handles an unknown document or an
    unknown section with a specific, actionable error rather than a
    generic failure.
    """
    docs = get_parsed_docs()
    doc = docs.get(document_id)
    if doc is None:
        known = ", ".join(sorted(docs))
        raise ToolError(f"No policy document found with ID '{document_id}'. Known document IDs: {known}.")

    normalized = section.strip().lower()
    match = next((s for s in doc.sections if s.heading.lower() == normalized), None)
    if match is None:
        match = next((s for s in doc.sections if normalized in s.heading.lower()), None)
    if match is None:
        available = "; ".join(s.heading for s in doc.sections)
        raise ToolError(
            f"No section matching '{section}' was found in {document_id} ({doc.title}). "
            f"Available sections: {available}."
        )

    return PolicySectionResult(
        document_id=doc.doc_id,
        document_title=doc.title,
        effective_date=doc.effective_date,
        section=match.heading,
        content=match.text,
        source_path=str(doc.source_path),
    )


# ---------------------------------------------------------------------------
# 3. lookup_employee_profile
# ---------------------------------------------------------------------------
@server.tool()
def lookup_employee_profile(employee_id: str) -> EmployeeProfile:
    """Look up a synthetic employee's HR-relevant profile fields. Excludes
    fields not needed for HR workflows (e.g. email address, detailed work
    authorization status) to avoid exposing more than necessary.
    """
    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    department = next((d for d in ds.departments if d.department_id == employee.department_id), None)
    location = next((l for l in ds.locations if l.location_id == employee.home_location_id), None)
    manager = ds.employee_by_id(employee.manager_id) if employee.manager_id else None

    if location is not None:
        parts = [location.city] + ([location.region] if location.region else []) + [location.country]
        home_location = ", ".join(parts)
    else:
        home_location = employee.home_location_id

    return EmployeeProfile(
        employee_id=employee.employee_id,
        full_name=f"{employee.first_name} {employee.last_name}",
        title=employee.title,
        department_id=employee.department_id,
        department_name=department.name if department else employee.department_id,
        manager_id=employee.manager_id,
        manager_name=f"{manager.first_name} {manager.last_name}" if manager else None,
        employment_type=employee.employment_type,
        work_arrangement=employee.work_arrangement,
        home_location_id=employee.home_location_id,
        home_location=home_location,
        employment_country=employee.employment_country,
        hire_date=employee.hire_date.isoformat(),
        status=employee.status,
    )


# ---------------------------------------------------------------------------
# 4. check_pto_balance
# ---------------------------------------------------------------------------
@server.tool()
def check_pto_balance(employee_id: str) -> PTOBalanceResult:
    """Return an employee's current PTO balance. An unknown employee ID is
    an error; a *known* employee with no tracked balance (Director+ on the
    Flexible PTO plan, or an employment type not covered by PTO at all) is
    a normal, successful result explaining why, per POL-PTO-001.
    """
    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    balance = ds.pto_balance_for(employee_id)
    if balance is None:
        if employee.employment_type in ("contractor", "intern", "eor_international"):
            return PTOBalanceResult(
                employee_id=employee_id,
                has_tracked_balance=False,
                plan_type="not_covered",
                balance_hours=None,
                note=(
                    f"Employment type '{employee.employment_type}' is not covered by the PTO "
                    "policy (POL-PTO-001 Section 9); no balance is tracked."
                ),
            )
        return PTOBalanceResult(
            employee_id=employee_id,
            has_tracked_balance=False,
            plan_type="flexible",
            balance_hours=None,
            note=(
                "This employee participates in the Flexible PTO plan (Director-level and above); "
                "no fixed balance is tracked per POL-PTO-001 Section 4."
            ),
        )

    return PTOBalanceResult(
        employee_id=employee_id,
        has_tracked_balance=True,
        plan_type=balance.accrual_type,
        balance_hours=balance.balance_hours,
        accrued_ytd_hours=balance.accrued_ytd_hours,
        used_ytd_hours=balance.used_ytd_hours,
        pending_hours=balance.pending_hours,
        carryover_hours_from_prior_year=balance.carryover_hours_from_prior_year,
        as_of_date=balance.as_of_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# 5. lookup_benefits_status
# ---------------------------------------------------------------------------
@server.tool()
def lookup_benefits_status(employee_id: str) -> BenefitsStatusResult:
    """Return an employee's benefits enrollment/eligibility status. An
    unknown employee ID is an error; a known employee with no benefits
    record (contractor/intern/EOR international) is a normal result
    explaining why, per POL-BEN-001.
    """
    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    record = ds.benefits_for(employee_id)
    if record is None:
        return BenefitsStatusResult(
            employee_id=employee_id,
            has_record=False,
            benefits_eligible=False,
            note=(
                f"No benefits record on file; employment type '{employee.employment_type}' is not "
                "eligible for company benefits per POL-BEN-001."
            ),
        )

    return BenefitsStatusResult(
        employee_id=employee_id,
        has_record=True,
        benefits_eligible=record.benefits_eligible,
        medical_plan=record.medical_plan,
        dental_enrolled=record.dental_enrolled,
        vision_enrolled=record.vision_enrolled,
        retirement_401k_enrolled=record.retirement_401k_enrolled,
        retirement_401k_contribution_pct=record.retirement_401k_contribution_pct,
        fsa_healthcare_enrolled=record.fsa_healthcare_enrolled,
        fsa_dependent_care_enrolled=record.fsa_dependent_care_enrolled,
        wellness_stipend_eligible=record.wellness_stipend_eligible,
        waiting_period_end_date=record.waiting_period_end_date.isoformat() if record.waiting_period_end_date else None,
        coverage_effective_date=record.coverage_effective_date.isoformat() if record.coverage_effective_date else None,
        note=record.notes,
    )


# ---------------------------------------------------------------------------
# 6. check_policy_compliance
# ---------------------------------------------------------------------------
@server.tool()
def check_policy_compliance(
    employee_id: str,
    scenario_type: str,
    requested_days: int | None = None,
    notice_business_days: int | None = None,
    destination_state: str | None = None,
    destination_country: str | None = None,
    duration_calendar_days: int | None = None,
) -> ComplianceResult:
    """Evaluate one of two scenario types against Solara Dynamics policy:

    - "pto_request": uses `requested_days` and optionally
      `notice_business_days`.
    - "remote_work_request": uses `destination_state` and/or
      `destination_country`, and `duration_calendar_days` (the real-world
      duration, e.g. "six weeks" = 42 — see `mcp_server.compliance` for why
      calendar days, not business days, is the authoritative input: the
      interstate 30/90-day tiers in POL-LOC-001 are themselves defined in
      calendar days).

    The verdict always comes from a fixed, documented rule table
    (`mcp_server.compliance`) applied to structured facts (the employee's
    record plus these arguments) — never from free-text interpretation —
    and is only returned as a rule result if real RAG retrieval also found
    at least one supporting policy citation; otherwise the status is
    "insufficient_evidence". See `mcp_server/compliance.py`'s module
    docstring for the full facts/rules/future-LLM distinction.
    """
    if scenario_type not in ("pto_request", "remote_work_request"):
        raise ToolError(
            f"Unknown scenario_type '{scenario_type}'. Expected 'pto_request' or "
            "'remote_work_request'."
        )

    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    facts_considered = [
        f"employment_type={employee.employment_type}",
        f"status={employee.status}",
        f"department_id={employee.department_id}",
    ]

    if scenario_type == "pto_request":
        balance = ds.pto_balance_for(employee_id)
        facts_considered.append(
            f"pto_balance_hours={balance.balance_hours}" if balance else "pto_balance_hours=none_tracked"
        )
        outcome = compliance.evaluate_pto_request(employee, balance, requested_days, notice_business_days)
        queries = ["PTO accrual balance carryover request notice", "Flexible PTO plan director approval"]
    else:
        facts_considered.append(f"home_location_id={employee.home_location_id}")
        outcome = compliance.evaluate_remote_work_request(destination_state, destination_country, duration_calendar_days)
        queries = [
            "remote work Work From Anywhere allowance business days",
            "payroll registered states interstate work location approval",
        ]
        if destination_country:
            queries.append("international remote work Employer of Record approved countries")

    citations, any_sufficient = compliance.gather_citations(queries)

    if not any_sufficient:
        return ComplianceResult(
            status="insufficient_evidence",
            reasons=[
                "No sufficiently relevant policy evidence was retrieved for this scenario, so no "
                "compliance determination is returned (see rag.retrieval's relevance guardrail)."
            ]
            + outcome.reasons,
            citations=citations,
            facts_considered=facts_considered,
            rule_basis="insufficient_evidence",
        )

    return ComplianceResult(
        status=outcome.status,
        reasons=outcome.reasons,
        citations=citations,
        required_approvals=outcome.required_approvals,
        escalation_required=outcome.escalation_required,
        facts_considered=facts_considered,
        rule_basis="deterministic_rule",
    )


# ---------------------------------------------------------------------------
# 7. draft_hr_email
# ---------------------------------------------------------------------------
@server.tool()
def draft_hr_email(
    employee_id: str,
    recipient: str,
    subject_intent: str,
    key_facts: list[str] | None = None,
    requested_action: str = "",
) -> DraftEmailResult:
    """Draft (never send) an HR-related email on behalf of an employee.
    This is a template-based draft in Phase 2 (no LLM agent yet); Phase 3
    may generate richer prose but the safety contract — draft only, never
    sent — does not change.
    """
    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    facts = key_facts or []
    facts_block = "\n".join(f"- {fact}" for fact in facts) if facts else "(no additional facts provided)"
    subject = f"{subject_intent} - {employee.first_name} {employee.last_name} ({employee.employee_id})"
    body = (
        f"Hi {recipient},\n\n"
        f"{subject_intent}\n\n"
        f"Employee: {employee.first_name} {employee.last_name} ({employee.employee_id})\n"
        f"Department: {employee.department_id}\n\n"
        f"Relevant details:\n{facts_block}\n\n"
        f"{requested_action}\n\n"
        f"Thanks,\n{employee.first_name}\n"
    )
    return DraftEmailResult(to=recipient, subject=subject, body=body)


# ---------------------------------------------------------------------------
# 8. create_mock_hr_ticket
# ---------------------------------------------------------------------------
@server.tool()
def create_mock_hr_ticket(
    employee_id: str,
    category: str,
    summary: str,
    details: str = "",
    priority: str = "normal",
    confirmed: bool = False,
) -> MockTicketResult:
    """Create a MOCK HR ticket. No real external ticketing system is ever
    contacted. Without `confirmed=true` this is a preview only — nothing
    is persisted, and no ticket ID is generated. Call again with
    `confirmed=true` to actually append the (still synthetic) ticket
    record.
    """
    ds = get_dataset()
    employee = ds.employee_by_id(employee_id)
    if employee is None:
        raise ToolError(f"No employee found with ID '{employee_id}'.")

    if not confirmed:
        return MockTicketResult(
            status="preview",
            ticket_id=None,
            employee_id=employee_id,
            category=category,
            summary=summary,
            details=details,
            priority=priority,
            message=(
                "Preview only; no mock ticket has been created and no state was changed. "
                "Call again with confirmed=true to create it."
            ),
        )

    ticket_id = tickets.create_ticket(employee_id, category, summary, details, priority)
    return MockTicketResult(
        status="created",
        ticket_id=ticket_id,
        employee_id=employee_id,
        category=category,
        summary=summary,
        details=details,
        priority=priority,
        message=f"Mock ticket {ticket_id} created (synthetic; no external ticketing system was contacted).",
    )
