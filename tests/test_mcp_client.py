"""Client-specific behavior not already covered in test_mcp_server.py:
using the client before connecting, manual connect/close (no context
manager), and idempotent connect().
"""
from __future__ import annotations

import anyio
import pytest

from mcp_client.client import MCPClient, MCPClientError


def test_call_tool_before_connect_raises_clean_error() -> None:
    async def body() -> None:
        client = MCPClient()
        with pytest.raises(MCPClientError, match="Not connected"):
            await client.call_tool("check_pto_balance", {"employee_id": "EMP-1001"})

    anyio.run(body)


def test_list_tools_before_connect_raises_clean_error() -> None:
    async def body() -> None:
        client = MCPClient()
        with pytest.raises(MCPClientError, match="Not connected"):
            await client.list_tools()

    anyio.run(body)


def test_manual_connect_and_close_without_context_manager(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        client = MCPClient(env=mcp_server_env)
        await client.connect()
        try:
            tools = await client.list_tools()
            assert len(tools) >= 8
        finally:
            await client.close()

    anyio.run(body)


def test_connect_is_idempotent(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        client = MCPClient(env=mcp_server_env)
        await client.connect()
        await client.connect()  # must not raise or open a second subprocess
        try:
            tools = await client.list_tools()
            assert len(tools) >= 8
        finally:
            await client.close()

    anyio.run(body)


def test_async_context_manager_closes_on_exception(mcp_server_env: dict[str, str]) -> None:
    async def body() -> None:
        with pytest.raises(ValueError):
            async with MCPClient(env=mcp_server_env) as client:
                await client.list_tools()
                raise ValueError("boom")
        # If we get here without hanging, the exit stack cleaned up correctly.

    anyio.run(body)
