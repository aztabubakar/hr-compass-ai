"""Unit-level regression tests for the deterministic compliance rule
engine (`mcp_server/compliance.py`), focused on the exact calendar-day
thresholds POL-LOC-001 defines for interstate work, and the business-day
thresholds POL-RWK-001/POL-LOC-001 define for the "Work From Anywhere" and
international allowances. These call the rule functions directly (no MCP
protocol, no RAG) so boundary values can be checked exhaustively and fast.
"""
from __future__ import annotations

import pytest

from mcp_server.compliance import calendar_to_business_days, evaluate_remote_work_request


def test_calendar_to_business_days_matches_policy_stated_equivalence():
    # The policy itself states "6 consecutive weeks (30 business days)",
    # i.e. exactly 5 business days per 7 calendar days.
    assert calendar_to_business_days(42) == 30
    assert calendar_to_business_days(7) == 5
    assert calendar_to_business_days(0) == 0


# ---------------------------------------------------------------------------
# Interstate: exact calendar-day tiers (POL-LOC-001 Section 4).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("days", [1, 15, 30])
def test_interstate_at_or_under_30_calendar_days_is_compliant(days: int) -> None:
    outcome = evaluate_remote_work_request("Colorado", None, days)
    assert outcome.status == "compliant"
    assert outcome.escalation_required is False


@pytest.mark.parametrize("days", [31, 45, 90])
def test_interstate_31_to_90_calendar_days_requires_payroll_approval(days: int) -> None:
    outcome = evaluate_remote_work_request("Colorado", None, days)
    assert outcome.status == "requires_approval"
    assert "Payroll approval" in outcome.required_approvals
    assert "People & Culture Team approval" in outcome.required_approvals
    assert "VP approval" not in outcome.required_approvals


@pytest.mark.parametrize("days", [91, 120, 365])
def test_interstate_over_90_calendar_days_is_relocation_tier(days: int) -> None:
    outcome = evaluate_remote_work_request("Colorado", None, days)
    assert outcome.status == "requires_approval"
    assert "VP approval" in outcome.required_approvals
    assert "Payroll approval" not in outcome.required_approvals


def test_interstate_boundary_30_vs_31_calendar_days_changes_tier() -> None:
    at_boundary = evaluate_remote_work_request("Colorado", None, 30)
    just_over = evaluate_remote_work_request("Colorado", None, 31)
    assert at_boundary.status == "compliant"
    assert just_over.status == "requires_approval"


def test_interstate_boundary_90_vs_91_calendar_days_changes_tier() -> None:
    payroll_tier = evaluate_remote_work_request("Colorado", None, 90)
    relocation_tier = evaluate_remote_work_request("Colorado", None, 91)
    assert "Payroll approval" in payroll_tier.required_approvals
    assert "VP approval" in relocation_tier.required_approvals


def test_interstate_six_weeks_matches_policy_worked_example() -> None:
    """The corpus's own POL-LOC-001 'Worked Examples' section states that
    six consecutive weeks to Colorado (a registered state) falls in the
    31-90 calendar-day tier and is 'approvable with review', not
    automatically compliant and not denied."""
    outcome = evaluate_remote_work_request("Colorado", None, 42)
    assert outcome.status == "requires_approval"
    assert outcome.escalation_required is True


def test_interstate_unregistered_state_requires_approval_regardless_of_duration() -> None:
    short = evaluate_remote_work_request("Wyoming", None, 5)
    long = evaluate_remote_work_request("Wyoming", None, 200)
    assert short.status == "requires_approval"
    assert long.status == "requires_approval"
    assert "Payroll registration review" in short.required_approvals


def test_interstate_missing_duration_is_insufficient_evidence() -> None:
    outcome = evaluate_remote_work_request("Colorado", None, None)
    assert outcome.status == "insufficient_evidence"


def test_interstate_short_stay_exceeding_work_from_anywhere_business_days_still_compliant() -> None:
    """A stay of e.g. 30 calendar days is ~21.4 business days, over the
    POL-RWK-001 20-business-day 'Work From Anywhere' allowance, but
    POL-LOC-001 Section 4's own <=30-calendar-day tier still requires no
    additional approval — the request is simply routed as a formal Work
    Location Change Request instead of informal manager notice. Status
    must still be 'compliant', with the business-day note surfaced in
    reasons for transparency, not a stricter status."""
    outcome = evaluate_remote_work_request("Colorado", None, 30)
    assert outcome.status == "compliant"
    assert any("business days" in reason for reason in outcome.reasons)


# ---------------------------------------------------------------------------
# International: exact business-day allowance (POL-LOC-001 Section 6),
# derived from the calendar-day input.
# ---------------------------------------------------------------------------
def test_international_at_six_weeks_requires_approval_not_denied() -> None:
    outcome = evaluate_remote_work_request(None, "Portugal", 42)
    assert outcome.status == "requires_approval"
    assert "Global Mobility review" in outcome.required_approvals


def test_international_boundary_30_vs_31_business_days() -> None:
    # 42 calendar days -> 30 business days (within allowance).
    within = evaluate_remote_work_request(None, "Canada", 42)
    # 43 calendar days -> round(43*5/7) = 31 business days (exceeds allowance).
    assert calendar_to_business_days(43) == 31
    over = evaluate_remote_work_request(None, "Canada", 43)
    assert within.status == "requires_approval"
    assert over.status == "non_compliant"


def test_international_unapproved_country_is_non_compliant_regardless_of_duration() -> None:
    short = evaluate_remote_work_request(None, "Thailand", 5)
    long_stay = evaluate_remote_work_request(None, "Thailand", 42)
    assert short.status == "non_compliant"
    assert long_stay.status == "non_compliant"


def test_international_missing_duration_is_insufficient_evidence() -> None:
    outcome = evaluate_remote_work_request(None, "Portugal", None)
    assert outcome.status == "insufficient_evidence"


def test_neither_state_nor_country_is_insufficient_evidence() -> None:
    outcome = evaluate_remote_work_request(None, None, 42)
    assert outcome.status == "insufficient_evidence"
