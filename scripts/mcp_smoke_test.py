"""Automated MCP discovery + tool-call smoke test.

    python -m scripts.mcp_smoke_test

Builds a small, hermetic RAG index (hashing embedding backend, so this
needs no network access or model download), launches the real MCP server
subprocess, discovers its tools over the real Model Context Protocol, and
calls two of them (one RAG-backed, one structured-data-backed) to prove
both required MCP tool categories work end to end. Exits non-zero on any
failure so CI can gate on it.

This is deliberately a thin, standalone script (not just "run pytest") so
it can be one explicit, individually-named CI step -- PROJECT_SPEC and
docs/acceptance-checklist.md both call out "MCP discovery/tool-call test"
as its own required CI step, separate from the general test suite (which
already covers this much more thoroughly via tests/test_mcp_*.py).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import anyio

from core.paths import POLICIES_DIR
from mcp_client.client import MCPClient, MCPClientError, MCPToolError
from rag.config import RagConfig
from rag.ingest import run_ingest

EXPECTED_TOOL_NAMES = {
    "search_policy_documents", "get_policy_section", "lookup_employee_profile",
    "check_pto_balance", "lookup_benefits_status", "check_policy_compliance",
    "draft_hr_email", "create_mock_hr_ticket",
}


async def _main() -> int:
    with tempfile.TemporaryDirectory(prefix="hr_compass_mcp_smoke_") as tmp_dir:
        index_dir = Path(tmp_dir)
        print(f"Building a hermetic RAG index (hashing backend) at {index_dir} ...")
        run_ingest(RagConfig(policies_dir=POLICIES_DIR, index_dir=index_dir, embedding_backend="hashing"))

        env = {**os.environ, "EMBEDDING_BACKEND": "hashing", "RAG_INDEX_DIR": str(index_dir)}

        try:
            async with MCPClient(env=env) as client:
                print("Connected to MCP server. Discovering tools ...")
                tools = await client.list_tools()
                names = {t.name for t in tools}
                missing = EXPECTED_TOOL_NAMES - names
                if missing:
                    print(f"FAIL: expected tools missing from discovery: {sorted(missing)}")
                    return 1
                print(f"OK: discovered {len(names)} tools, including all {len(EXPECTED_TOOL_NAMES)} required ones.")

                print("Calling a RAG-backed tool: search_policy_documents ...")
                search_result = await client.call_tool("search_policy_documents", {"query": "paid holidays", "top_k": 3})
                if not search_result.get("results"):
                    print("FAIL: search_policy_documents returned no results.")
                    return 1
                print(f"OK: {len(search_result['results'])} result(s) returned.")

                print("Calling a structured-data-backed tool: lookup_employee_profile ...")
                profile = await client.call_tool("lookup_employee_profile", {"employee_id": "EMP-1001"})
                if profile.get("employee_id") != "EMP-1001":
                    print(f"FAIL: unexpected profile result: {profile}")
                    return 1
                print(f"OK: looked up {profile.get('full_name')}.")

                print("Confirming a clean tool-level error for an unknown employee ID ...")
                try:
                    await client.call_tool("lookup_employee_profile", {"employee_id": "EMP-9999"})
                    print("FAIL: expected an MCPToolError for an unknown employee ID.")
                    return 1
                except MCPToolError:
                    print("OK: unknown employee ID raised a clean MCPToolError, not a crash.")
        except MCPClientError as exc:
            print(f"FAIL: could not connect to or use the MCP server: {exc}")
            return 1

    print("\nMCP discovery/tool-call smoke test PASSED.")
    return 0


def main() -> int:
    return anyio.run(_main)


if __name__ == "__main__":
    sys.exit(main())
