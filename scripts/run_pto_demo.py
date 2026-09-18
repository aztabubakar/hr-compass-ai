"""Run the deterministic PTO workflow (Workflow A) over real MCP calls.

    python -m scripts.run_pto_demo                     # EMP-1001, 3 days, happy path
    python -m scripts.run_pto_demo EMP-1009 3 2         # edge case: insufficient balance
"""
from __future__ import annotations

import json
import sys

import anyio

from mcp_client.client import MCPClient
from scripts.demo_workflows import run_pto_workflow


async def main() -> None:
    employee_id = sys.argv[1] if len(sys.argv) > 1 else "EMP-1001"
    requested_days = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    notice_business_days = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    async with MCPClient() as client:
        trace, compliance = await run_pto_workflow(client, employee_id, requested_days, notice_business_days)

    print(f"=== PTO workflow for {employee_id}: {requested_days} day(s), {notice_business_days} business days notice ===\n")
    print("Operational trace:")
    print(json.dumps(trace, indent=2))
    print("\nFinal compliance result:")
    print(json.dumps(compliance, indent=2))


if __name__ == "__main__":
    anyio.run(main)
