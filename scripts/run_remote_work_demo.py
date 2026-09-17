"""Run the deterministic remote-work workflow (Workflow B) over real MCP
calls, for both the interstate (EMP-1002) and international (EMP-1003)
scenarios.

    python -m scripts.run_remote_work_demo
"""
from __future__ import annotations

import json

import anyio

from mcp_client.client import MCPClient
from scripts.demo_workflows import run_remote_work_workflow


async def main() -> None:
    async with MCPClient() as client:
        print("=== Remote-work workflow: EMP-1002, Colorado, 30 business days (interstate) ===\n")
        trace_a, compliance_a = await run_remote_work_workflow(
            client, "EMP-1002", destination_state="Colorado", duration_business_days=30
        )
        print(json.dumps(trace_a, indent=2))
        print("\nFinal compliance result:")
        print(json.dumps(compliance_a, indent=2))

        print("\n\n=== Remote-work workflow: EMP-1003, Portugal, 30 business days (international) ===\n")
        trace_b, compliance_b = await run_remote_work_workflow(
            client, "EMP-1003", destination_country="Portugal", duration_business_days=30
        )
        print(json.dumps(trace_b, indent=2))
        print("\nFinal compliance result:")
        print(json.dumps(compliance_b, indent=2))


if __name__ == "__main__":
    anyio.run(main)
