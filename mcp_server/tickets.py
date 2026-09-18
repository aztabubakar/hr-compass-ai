"""Mock HR ticket persistence.

`create_mock_hr_ticket` is the one tool in this project that mutates
state, even if only synthetic local state, so it gets the strictest
safety treatment:

- A call without `confirmed=true` is a **preview only**: nothing is
  written anywhere, and the tool result makes that explicit.
- A call with `confirmed=true` appends one record to a runtime-only JSON
  file (`core.paths.RUNTIME_TICKETS_PATH`), separate from the read-only
  committed fixture `mock_data/hr_tickets.json`, so running the server (or
  its tests) never mutates committed synthetic data.
- No real external ticketing system is ever contacted; "creation" means
  "appended to this local JSON file."

Ticket IDs continue the numbering already used in the committed fixture
data (`TCK-2xxxx`) so generated IDs look like a natural continuation
rather than an obviously different namespace.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from core.paths import RUNTIME_TICKETS_PATH
from mcp_server.dataset import get_dataset

_TICKET_NUMBER_RE = re.compile(r"(\d+)$")


def _load_runtime_tickets(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _next_ticket_id(path: Path) -> str:
    numbers = []
    for ticket in get_dataset().hr_tickets:
        match = _TICKET_NUMBER_RE.search(ticket.ticket_id)
        if match:
            numbers.append(int(match.group(1)))
    for record in _load_runtime_tickets(path):
        match = _TICKET_NUMBER_RE.search(record["ticket_id"])
        if match:
            numbers.append(int(match.group(1)))
    next_number = max(numbers, default=20000) + 1
    return f"TCK-{next_number}"


def create_ticket(
    employee_id: str,
    category: str,
    summary: str,
    details: str,
    priority: str,
    *,
    runtime_tickets_path: Path = RUNTIME_TICKETS_PATH,
) -> str:
    """Persist one confirmed mock ticket and return its generated ID.

    Callers (the `create_mock_hr_ticket` tool) are responsible for only
    calling this once `confirmed=true` has been verified; this function
    itself has no preview mode, by design, so there is exactly one place
    in the codebase where a mock ticket is actually written.
    """
    ticket_id = _next_ticket_id(runtime_tickets_path)
    record = {
        "ticket_id": ticket_id,
        "employee_id": employee_id,
        "category": category,
        "subject": summary,
        "status": "open",
        "created_date": date.today().isoformat(),
        "resolved_date": None,
        "summary": details or summary,
        "priority": priority,
    }
    tickets = _load_runtime_tickets(runtime_tickets_path)
    tickets.append(record)
    runtime_tickets_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tickets_path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    return ticket_id
