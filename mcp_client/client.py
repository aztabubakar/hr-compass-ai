"""Async MCP client: connects to `mcp_server` over stdio, discovers tools,
and invokes them through the real Model Context Protocol.

Usage:

    async with MCPClient() as client:
        tools = await client.list_tools()
        result = await client.call_tool("check_pto_balance", {"employee_id": "EMP-1001"})

or, without the context manager:

    client = MCPClient()
    await client.connect()
    try:
        ...
    finally:
        await client.close()

This module intentionally never imports anything from `mcp_server.tools`.
Every `call_tool` here is a genuine MCP `tools/call` request sent over the
stdio pipe to a separate server subprocess and answered over that same
pipe — there is no in-process shortcut.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool

DEFAULT_SERVER_COMMAND = sys.executable
DEFAULT_SERVER_ARGS = ["-m", "mcp_server.server"]


class MCPClientError(RuntimeError):
    """Raised for connection/transport-level failures (server wouldn't
    start, connection dropped, client used before connect()/after close())."""


class MCPToolError(RuntimeError):
    """Raised when a tool call completes but the server reports
    `is_error=True` — i.e. a clean, structured tool-level failure (unknown
    tool, unknown employee ID, missing required argument, etc.), as
    opposed to a transport failure."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"MCP tool '{tool_name}' failed: {message}")


def _extract_text(content: list[Any]) -> str:
    parts = [block.text for block in content if isinstance(block, TextContent)]
    return "\n".join(parts)


class MCPClient:
    """Reusable async wrapper around `mcp.ClientSession` for a single
    stdio-connected server process."""

    def __init__(
        self,
        command: str = DEFAULT_SERVER_COMMAND,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args if args is not None else list(DEFAULT_SERVER_ARGS)
        # The MCP SDK's own default (env=None passed straight through) is a
        # *minimal, security-filtered* environment (PATH/HOME/etc. only) —
        # sensible for launching an arbitrary third-party MCP server, but
        # wrong here: `mcp_server` is our own trusted, first-party code that
        # needs this project's configuration (EMBEDDING_BACKEND,
        # RAG_INDEX_DIR, and in a later phase LLM_API_KEY) to behave the
        # same way the parent process was configured. So unless the caller
        # passes an explicit `env`, we inherit the full parent environment.
        self._env = env if env is not None else dict(os.environ)
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """Launch the MCP server subprocess and complete the MCP
        `initialize` handshake. Raises `MCPClientError` (never a raw
        low-level exception) if the server can't be started or the
        handshake fails."""
        if self._session is not None:
            return
        try:
            params = StdioServerParameters(command=self._command, args=self._args, env=self._env)
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
        except BaseException as exc:
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
            raise MCPClientError(f"Failed to start or connect to the MCP server: {exc}") from exc

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPClientError(
                "Not connected to an MCP server. Call connect() first, or use "
                "'async with MCPClient() as client:'."
            )
        return self._session

    async def list_tools(self) -> list[Tool]:
        """Return the tools discovered from the server via MCP `tools/list`."""
        session = self._require_session()
        try:
            result = await session.list_tools()
        except BaseException as exc:
            raise MCPClientError(f"Failed to list tools: {exc}") from exc
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke `name` via MCP `tools/call` and return its structured
        result as a plain dict.

        Raises `MCPToolError` for a clean tool-level failure (unknown tool
        name, unknown employee ID, a validation error on the arguments,
        etc. — anything the server reported as `is_error=True`), and
        `MCPClientError` for a transport-level failure.
        """
        session = self._require_session()
        try:
            result = await session.call_tool(name, arguments or {})
        except BaseException as exc:
            raise MCPClientError(f"Transport error calling tool '{name}': {exc}") from exc

        if result.is_error:
            raise MCPToolError(name, _extract_text(result.content) or f"Tool '{name}' failed.")

        if result.structured_content is not None:
            return result.structured_content

        # Defensive fallback: every tool in mcp_server.tools returns a typed
        # pydantic model, which always populates structured_content (verified
        # empirically against mcp 2.2.0), so this path should not normally be
        # hit. It exists so a text-only tool response is still usable rather
        # than raising.
        text = _extract_text(result.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    async def close(self) -> None:
        """Idempotent: safe to call multiple times, and safe to call even
        if connect() was never called or already failed."""
        await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()
        self._session = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
