"""Tests for the synthetic HR structured data (referential integrity and
the deterministic demo employees the two Phase 2 workflows depend on).
"""
from __future__ import annotations

from mock_data.loader import load_all


def test_dataset_loads_and_validates() -> None:
    dataset = load_all()
    assert len(dataset.employees) >= 15
    assert len(dataset.departments) >= 5
    assert len(dataset.locations) >= 5


def test_pto_demo_employee_has_sufficient_balance() -> None:
    """Demo workflow A: 'Can I take three days of PTO next week?'"""
    dataset = load_all()
    employee = dataset.employee_by_id("EMP-1001")
    assert employee is not None
    assert employee.status == "active"
    assert employee.employment_type == "full_time"

    balance = dataset.pto_balance_for("EMP-1001")
    assert balance is not None
    three_days_in_hours = 24.0
    assert balance.balance_hours >= three_days_in_hours


def test_pto_demo_edge_case_employee_has_insufficient_balance() -> None:
    """A contrasting edge case for the PTO workflow: not enough balance."""
    dataset = load_all()
    balance = dataset.pto_balance_for("EMP-1009")
    assert balance is not None
    assert balance.balance_hours < 24.0


def test_remote_work_demo_employees_exist_for_interstate_and_international() -> None:
    """Demo workflow B: 'Can I work from another state/country for six weeks?'"""
    dataset = load_all()

    interstate_employee = dataset.employee_by_id("EMP-1002")
    assert interstate_employee is not None
    assert interstate_employee.work_arrangement == "remote"
    assert interstate_employee.home_location_id == "LOC-TX-AUS"

    international_employee = dataset.employee_by_id("EMP-1003")
    assert international_employee is not None
    assert international_employee.work_arrangement == "remote"


def test_manager_hierarchy_has_no_dangling_or_circular_top_level() -> None:
    dataset = load_all()
    ceo = [e for e in dataset.employees if e.manager_id is None]
    assert len(ceo) == 1  # exactly one root of the org chart


def test_edge_case_employees_are_present() -> None:
    dataset = load_all()
    by_type = {e.employment_type for e in dataset.employees}
    assert {"full_time", "part_time", "contractor", "intern", "eor_international"} <= by_type

    by_status = {e.status for e in dataset.employees}
    assert {"active", "on_leave", "terminated"} <= by_status

    # Director-level employees use the Flexible PTO plan and therefore
    # should have no tracked PTO balance record at all.
    director = next(e for e in dataset.employees if e.employee_id == "EMP-1007")
    assert dataset.pto_balance_for(director.employee_id) is None


def test_part_time_employee_not_eligible_for_medical_benefits() -> None:
    dataset = load_all()
    benefits = dataset.benefits_for("EMP-1004")
    assert benefits is not None
    assert benefits.benefits_eligible is False
    assert benefits.medical_plan is None


def test_tickets_reference_real_employees() -> None:
    dataset = load_all()
    for ticket in dataset.hr_tickets:
        assert dataset.employee_by_id(ticket.employee_id) is not None
