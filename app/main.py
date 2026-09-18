"""HR Compass AI FastAPI application.

Run locally with:

    uvicorn app.main:app --reload

The MCP client is connected once at startup (a single, shared, long-lived
subprocess) and closed at shutdown, rather than spawned per-request — see
`lifespan()` below. Both MCP and LLM unavailability are handled
gracefully: the app still starts and `/health` still responds accurately
even if the RAG index hasn't been built or `GROQ_API_KEY` isn't set;
`/chat` returns a clean error in that case instead of crashing.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.orchestrator import AgentOrchestrator
from app.models import ChatRequest, ChatResponse, HealthResponse, LlmHealth, McpHealth, RagHealth, ToolInfo, ToolListResponse
from app.state import ConversationStore
from core.paths import DEFAULT_RAG_INDEX_DIR
from llm.base import LLMConfigError, LLMProvider, LLMProviderError
from llm.config import LLMConfig, get_provider, is_llm_configured
from mcp_client.client import MCPClient, MCPClientError

logger = logging.getLogger("hr_compass_ai.app")

# Without this, the root logger has no handler at all, so anything below
# WARNING (e.g. this module's own `logger.info("Connected to MCP
# server.")`) is silently dropped by Python's logging "handler of last
# resort" -- and even WARNING/ERROR only reach stderr by accident of that
# fallback, with no timestamp/logger-name context. `basicConfig` is a
# no-op if a handler is already attached to the root logger (e.g. because
# something else configured logging first), so this is safe to call
# unconditionally. Overridable via LOG_LEVEL for local debugging
# (`LOG_LEVEL=DEBUG uvicorn app.main:app`); defaults to INFO in production.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MCPClient()
    try:
        await client.connect()
        app.state.mcp_client = client
        logger.info("Connected to MCP server.")
    except MCPClientError as exc:
        logger.warning("MCP server unavailable at startup: %s", exc)
        app.state.mcp_client = None

    app.state.conversation_store = ConversationStore()

    yield

    if app.state.mcp_client is not None:
        await app.state.mcp_client.close()


app = FastAPI(title="HR Compass AI", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last-resort safety net: agent/MCP/LLM failures are already caught and
    # turned into clean responses well before this point (see
    # agent.orchestrator.AgentOrchestrator.run), so reaching here means a
    # genuinely unexpected error. Never leak the exception or a traceback.
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred. Please try again."})


def get_llm_provider() -> LLMProvider:
    """FastAPI dependency constructing the configured real LLM provider.

    Kept as an overridable dependency (rather than called directly inside
    the route) specifically so tests can swap in
    `llm.fake_provider.FakeLLMProvider` via
    `app.dependency_overrides[get_llm_provider] = ...` without touching
    real configuration or making a network call — see
    tests/test_api.py.
    """
    try:
        return get_provider()
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=f"LLM is not configured: {exc}") from exc


def _rag_status() -> str:
    # Reads RAG_INDEX_DIR directly from the environment at call time rather
    # than via `rag.config.RagConfig()`'s default: that dataclass field
    # default is computed once, at module-import time, so going through it
    # here would report whatever RAG_INDEX_DIR happened to be set to the
    # *first* time `rag.config` was imported in this process, not its
    # current value. Not an issue in normal deployment (env vars are set
    # before the process starts), but this avoids the sharp edge entirely.
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_RAG_INDEX_DIR)))
    return "ok" if (index_dir / "manifest.json").exists() else "not_built"


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Useful for Render's health check and for grading: reports readiness
    of each dependency without ever exposing a key or secret. `status` is
    "degraded" (not an error response -- the app is still up) when an
    important dependency isn't ready, so a monitoring/grading tool can see
    the difference between "process is up" and "fully functional"."""
    client = request.app.state.mcp_client
    tools_discovered: int | None = None
    mcp_ready = client is not None
    if client is not None:
        try:
            tools_discovered = len(await client.list_tools())
        except MCPClientError:
            mcp_ready = False

    rag_ready = _rag_status() == "ok"
    llm_configured = is_llm_configured()
    status = "ok" if (rag_ready and mcp_ready) else "degraded"

    return HealthResponse(
        status=status,
        rag=RagHealth(ready=rag_ready),
        mcp=McpHealth(ready=mcp_ready, tools_discovered=tools_discovered),
        llm=LlmHealth(configured=llm_configured, provider=LLMConfig().provider),
    )


@app.get("/api/tools", response_model=ToolListResponse)
async def list_tools(request: Request) -> ToolListResponse:
    client: MCPClient | None = request.app.state.mcp_client
    if client is None:
        raise HTTPException(status_code=503, detail="MCP server is not available.")
    try:
        tools = await client.list_tools()
    except MCPClientError as exc:
        raise HTTPException(status_code=503, detail=f"Failed to list MCP tools: {exc}") from exc
    return ToolListResponse(
        tools=[ToolInfo(name=t.name, description=t.description or "", input_schema=t.input_schema) for t in tools]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    llm: LLMProvider = Depends(get_llm_provider),
) -> ChatResponse:
    client: MCPClient | None = request.app.state.mcp_client
    if client is None:
        raise HTTPException(status_code=503, detail="MCP server is not available. Please try again shortly.")

    store: ConversationStore = request.app.state.conversation_store
    conversation_id = payload.conversation_id or str(uuid4())
    state = store.get_state(conversation_id)

    orchestrator = AgentOrchestrator(llm, client)
    try:
        result = await orchestrator.run(
            payload.message,
            history=state.history,
            prior_turn_requires_confirmation=state.last_requires_confirmation,
        )
    except LLMProviderError as exc:
        # Defense in depth: AgentOrchestrator.run() already catches
        # LLMProviderError internally and returns a graceful AgentResponse,
        # so this should not normally trigger; kept as a safety net.
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    store.append_turn(conversation_id, payload.message, result.answer, result.requires_confirmation)
    return ChatResponse(**result.model_dump(), conversation_id=conversation_id)


# Serve the chat UI. Registered last so the API routes above take
# precedence over the catch-all static mount at "/".
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
