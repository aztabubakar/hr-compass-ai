# Synthetic HR Data

**Status: implemented (Phase 1).** 20 employees, 9 departments, 9 locations,
12 PTO balance records, 16 benefits records, and 12 mock HR tickets — all
clearly synthetic (see `design-and-evaluation.md` Section 5 for the full
roster and the deterministic demo employees for the two Phase 2 workflows).

- `departments.json`, `locations.json`, `employees.json`, `pto_balances.json`, `benefits.json`, `hr_tickets.json` — the data.
- `models.py` — pydantic schema.
- `loader.py` — `load_all()` loads and cross-validates referential integrity (raises on any dangling department/location/manager/employee reference).

No real PII; all data is invented for this project.
