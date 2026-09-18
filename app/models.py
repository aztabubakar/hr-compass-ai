"""Request/response models for the FastAPI API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from agent.schemas import AgentResponse


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(AgentResponse):
    conversation_id: str


class RagHealth(BaseModel):
    ready: bool


class McpHealth(BaseModel):
    ready: bool
    tools_discovered: int | None = None


class LlmHealth(BaseModel):
    configured: bool
    provider: str


class HealthResponse(BaseModel):
    """Never include API keys, secrets, or other sensitive environment data
    here -- only readiness booleans and non-sensitive configuration (e.g.
    which provider *name* is selected, never a key)."""

    status: str  # "ok" if every important dependency is ready, else "degraded"
    rag: RagHealth
    mcp: McpHealth
    llm: LlmHealth


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
