"""Standalone MCP discovery + call smoke test.

Run with:

    python -m scripts.mcp_smoke_test

Connects to the MCP server over stdio, lists the discovered tools and
their schemas, and makes one real tool call, printing everything to
stdout. Useful as a quick manual check that the server/client wiring
works, independent of pytest.
"""
from __future__ import annotations

import json

import anyio

from mcp_client.client import MCPClient


async def main() -> None:
    async with MCPClient() as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools:\n")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print(f"    input_schema: {json.dumps(tool.input_schema)}")
        print()

        print("Calling lookup_employee_profile(employee_id='EMP-1001')...")
        result = await client.call_tool("lookup_employee_profile", {"employee_id": "EMP-1001"})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    anyio.run(main)
