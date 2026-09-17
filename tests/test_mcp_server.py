"""Tests proving the MCP server actually starts and speaks the protocol:
discovery, tool schemas, unknown-tool handling, and clean shutdown.

These spin up a real `mcp_server.server` subprocess over stdio for every
test (via `mcp_client.MCPClient`) — this is deliberate: at least one test
must prove real MCP tool discovery/calling through the protocol, not an
in-process stand-in. Each test uses `anyio.run(...)` to run its async body,
so no extra pytest-asyncio-style plugin is required.
"""
from __future__ import annotations

import anyio
import pytest

from mcp_client.client import MCPClient, MCPClientError, MCPToolError

REQUIRED_TOOL_NAMES = {
    "search_policy_documents",
    "get_policy_section",
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "check_policy_compliance",
    "draft_hr_email",
    "create_mock_hr_ticket",
}


def test_server_starts_and_client_connects(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        client = MCPClient(env=mcp_server_env)
        await client.connect()
        try:
            tools = await client.list_tools()
            assert len(tools) >= 8
        finally:
            await client.close()

    anyio.run(body)


def test_tool_discovery_exposes_all_required_tools(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            tools = await client.list_tools()
            names = {t.name for t in tools}
            assert REQUIRED_TOOL_NAMES <= names

    anyio.run(body)


def test_tool_schemas_are_valid_json_schema_objects(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            tools = await client.list_tools()
            by_name = {t.name: t for t in tools}

            for name in REQUIRED_TOOL_NAMES:
                schema = by_name[name].input_schema
                assert schema.get("type") == "object"
                assert "properties" in schema

            # Spot-check a couple of schemas carry the arguments we rely on.
            assert "employee_id" in by_name["check_pto_balance"].input_schema["properties"]
            assert "query" in by_name["search_policy_documents"].input_schema["properties"]
            compliance_props = by_name["check_policy_compliance"].input_schema["properties"]
            assert {"employee_id", "scenario_type"} <= compliance_props.keys()

    anyio.run(body)


def test_unknown_tool_name_returns_clean_error_not_a_crash(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        async with MCPClient(env=mcp_server_env) as client:
            with pytest.raises(MCPToolError):
                await client.call_tool("this_tool_does_not_exist", {})
            # The session should still be usable after a tool-level error.
            tools = await client.list_tools()
            assert len(tools) >= 8

    anyio.run(body)


def test_client_and_server_shutdown_cleanly(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        client = MCPClient(env=mcp_server_env)
        await client.connect()
        await client.list_tools()
        await client.close()
        # Idempotent: closing an already-closed client must not raise.
        await client.close()
        # Using a closed client raises a clean, typed error, not an
        # arbitrary low-level exception.
        with pytest.raises(MCPClientError):
            await client.list_tools()

    anyio.run(body)


def test_server_startup_failure_is_handled_gracefully() -> None:
    async def body() -> None:
        client = MCPClient(command="/definitely/not/a/real/executable", args=[])
        with pytest.raises(MCPClientError):
            await client.connect()
        # Must still be safe to close after a failed connect.
        await client.close()

    anyio.run(body)
