"""Loads and cross-validates the synthetic HR JSON data into `HRDataset`.

Referential integrity (an employee's department/location/manager exist, a
PTO balance or benefits record points at a real employee, etc.) is checked
here at load time so a broken fixture fails loudly and immediately rather
than surfacing as a confusing bug much later in an MCP tool.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.paths import MOCK_DATA_DIR
from mock_data.models import (
    BenefitsRecord,
    Department,
    Employee,
    HRDataset,
    HRTicket,
    Location,
    PTOBalance,
)


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Expected mock data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_departments(data_dir: Path = MOCK_DATA_DIR) -> list[Department]:
    return [Department.model_validate(d) for d in _load_json(data_dir / "departments.json")]


def load_locations(data_dir: Path = MOCK_DATA_DIR) -> list[Location]:
    return [Location.model_validate(d) for d in _load_json(data_dir / "locations.json")]


def load_employees(data_dir: Path = MOCK_DATA_DIR) -> list[Employee]:
    return [Employee.model_validate(d) for d in _load_json(data_dir / "employees.json")]


def load_pto_balances(data_dir: Path = MOCK_DATA_DIR) -> list[PTOBalance]:
    return [PTOBalance.model_validate(d) for d in _load_json(data_dir / "pto_balances.json")]


def load_benefits(data_dir: Path = MOCK_DATA_DIR) -> list[BenefitsRecord]:
    return [BenefitsRecord.model_validate(d) for d in _load_json(data_dir / "benefits.json")]


def load_hr_tickets(data_dir: Path = MOCK_DATA_DIR) -> list[HRTicket]:
    return [HRTicket.model_validate(d) for d in _load_json(data_dir / "hr_tickets.json")]


def load_all(data_dir: Path = MOCK_DATA_DIR) -> HRDataset:
    """Load every synthetic HR data file and validate referential integrity.

    Raises ValueError with a descriptive message on any dangling
    reference (unknown department/location/manager/employee id).
    """
    departments = load_departments(data_dir)
    locations = load_locations(data_dir)
    employees = load_employees(data_dir)
    pto_balances = load_pto_balances(data_dir)
    benefits = load_benefits(data_dir)
    hr_tickets = load_hr_tickets(data_dir)

    department_ids = {d.department_id for d in departments}
    location_ids = {l.location_id for l in locations}
    employee_ids = {e.employee_id for e in employees}

    errors: list[str] = []

    for e in employees:
        if e.department_id not in department_ids:
            errors.append(f"{e.employee_id}: unknown department_id {e.department_id!r}")
        if e.home_location_id not in location_ids:
            errors.append(f"{e.employee_id}: unknown home_location_id {e.home_location_id!r}")
        if e.manager_id is not None and e.manager_id not in employee_ids:
            errors.append(f"{e.employee_id}: unknown manager_id {e.manager_id!r}")

    for b in pto_balances:
        if b.employee_id not in employee_ids:
            errors.append(f"pto_balances: unknown employee_id {b.employee_id!r}")

    for b in benefits:
        if b.employee_id not in employee_ids:
            errors.append(f"benefits: unknown employee_id {b.employee_id!r}")

    for t in hr_tickets:
        if t.employee_id not in employee_ids:
            errors.append(f"hr_tickets: unknown employee_id {t.employee_id!r}")

    if errors:
        raise ValueError(
            "Synthetic HR data failed referential integrity checks:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return HRDataset(
        departments=departments,
        locations=locations,
        employees=employees,
        pto_balances=pto_balances,
        benefits=benefits,
        hr_tickets=hr_tickets,
    )
