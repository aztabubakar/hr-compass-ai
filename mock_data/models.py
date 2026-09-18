"""Typed models for the synthetic HR structured data.

These mirror the JSON files in this directory field-for-field. Using
pydantic here (rather than treating the JSON as untyped dicts) catches
schema drift and referential-integrity mistakes (e.g. an employee
pointing at a manager_id that doesn't exist) at load time instead of
inside a future MCP tool call.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

EmploymentType = Literal["full_time", "part_time", "contractor", "intern", "eor_international"]
WorkArrangement = Literal["onsite", "hybrid", "remote"]
EmploymentStatus = Literal["active", "on_leave", "terminated"]
LocationType = Literal["hq_office", "office", "remote", "eor_international"]
AccrualType = Literal["accrual_based", "flexible"]
TicketStatus = Literal["open", "in_progress", "resolved", "closed"]


class Department(BaseModel):
    model_config = ConfigDict(extra="forbid")

    department_id: str
    name: str
    description: str


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location_id: str
    city: str
    region: Optional[str] = None
    country: str
    country_code: str
    type: LocationType
    payroll_registered: bool


class Employee(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str
    first_name: str
    last_name: str
    email: str
    title: str
    department_id: str
    manager_id: Optional[str] = None
    employment_type: EmploymentType
    work_arrangement: WorkArrangement
    home_location_id: str
    work_authorization: str
    employment_country: str
    hire_date: date
    status: EmploymentStatus
    notes: Optional[str] = None


class PTOBalance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str
    plan_year: int
    accrual_type: AccrualType
    carryover_hours_from_prior_year: float
    accrued_ytd_hours: float
    used_ytd_hours: float
    pending_hours: float
    balance_hours: float
    as_of_date: date


class BenefitsRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str
    benefits_eligible: bool
    medical_plan: Optional[str] = None
    dental_enrolled: bool
    vision_enrolled: bool
    retirement_401k_enrolled: bool
    retirement_401k_contribution_pct: float
    fsa_healthcare_enrolled: bool
    fsa_dependent_care_enrolled: bool
    wellness_stipend_eligible: bool
    waiting_period_end_date: Optional[date] = None
    coverage_effective_date: Optional[date] = None
    notes: Optional[str] = None


class HRTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    employee_id: str
    category: str
    subject: str
    status: TicketStatus
    created_date: date
    resolved_date: Optional[date] = None
    summary: str


class HRDataset(BaseModel):
    """The full, cross-validated synthetic HR dataset."""

    model_config = ConfigDict(extra="forbid")

    departments: list[Department]
    locations: list[Location]
    employees: list[Employee]
    pto_balances: list[PTOBalance]
    benefits: list[BenefitsRecord]
    hr_tickets: list[HRTicket]

    def employee_by_id(self, employee_id: str) -> Optional[Employee]:
        return next((e for e in self.employees if e.employee_id == employee_id), None)

    def pto_balance_for(self, employee_id: str) -> Optional[PTOBalance]:
        return next((b for b in self.pto_balances if b.employee_id == employee_id), None)

    def benefits_for(self, employee_id: str) -> Optional[BenefitsRecord]:
        return next((b for b in self.benefits if b.employee_id == employee_id), None)

    def tickets_for(self, employee_id: str) -> list[HRTicket]:
        return [t for t in self.hr_tickets if t.employee_id == employee_id]

    def direct_reports(self, manager_id: str) -> list[Employee]:
        return [e for e in self.employees if e.manager_id == manager_id]
