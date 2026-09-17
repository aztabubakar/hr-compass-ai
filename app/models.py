"""Request/response models for the FastAPI API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from agent.schemas import AgentResponse


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(AgentResponse):
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    mcp: str
    rag: str
    llm_configured: bool


class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
