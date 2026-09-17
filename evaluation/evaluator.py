"""Shared evaluation harness: runs one `EvalCase` through the real
`AgentOrchestrator` (real MCP client/server, real RAG retrieval) and
records everything needed for scoring -- independent of whether the model
behind it is `FakeLLMProvider` (offline mode) or a real `GroqProvider`
(live mode). `run_offline.py` and `run_live.py` both build on this module;
neither re-implements the run/record logic itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent.orchestrator import AgentOrchestrator
from evaluation.cases import EvalCase
from llm.base import ChatMessage, LLMProvider
from mcp_client.client import MCPClient


@dataclass
class EvalResult:
    """Everything one case run produced, in a plain, JSON-serializable
    shape -- this is what gets written to `evaluation/results/*.json` and
    what `evaluation/metrics.py` scores."""

    case_id: str
    category: str
    user_input: str
    answer: str
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    tools_called: list[str]
    requires_clarification: bool
    requires_confirmation: bool
    escalated: bool
    latency_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "category": self.category, "user_input": self.user_input,
            "answer": self.answer, "citations": self.citations, "trace": self.trace,
            "tools_called": self.tools_called, "requires_clarification": self.requires_clarification,
            "requires_confirmation": self.requires_confirmation, "escalated": self.escalated,
            "latency_seconds": self.latency_seconds, "error": self.error,
        }


def _history_from_context(case: EvalCase) -> list[ChatMessage] | None:
    if not case.conversation_context:
        return None
    return [ChatMessage(role=turn.role, content=turn.content) for turn in case.conversation_context]


async def run_case(case: EvalCase, llm: LLMProvider, mcp_client: MCPClient) -> EvalResult:
    """Run a single case through the real orchestrator and record the
    result. Never raises: any exception (including an `LLMProviderError`
    the orchestrator itself didn't catch -- it normally does, but a live
    provider's timeout could still surface here) is captured into
    `EvalResult.error` so one bad case doesn't abort the whole run."""
    orchestrator = AgentOrchestrator(llm, mcp_client)
    start = time.perf_counter()
    try:
        result = await orchestrator.run(
            case.user_input,
            history=_history_from_context(case),
            prior_turn_requires_confirmation=case.prior_turn_requires_confirmation,
        )
        elapsed = time.perf_counter() - start
        tools_called = [step["tool"] for step in result.trace if step["type"] == "tool_call"]
        return EvalResult(
            case_id=case.case_id, category=case.category, user_input=case.user_input,
            answer=result.answer, citations=[c.model_dump() for c in result.citations],
            trace=result.trace, tools_called=tools_called,
            requires_clarification=result.requires_clarification,
            requires_confirmation=result.requires_confirmation, escalated=result.escalated,
            latency_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; see docstring
        elapsed = time.perf_counter() - start
        return EvalResult(
            case_id=case.case_id, category=case.category, user_input=case.user_input,
            answer="", citations=[], trace=[], tools_called=[],
            requires_clarification=False, requires_confirmation=False, escalated=False,
            latency_seconds=elapsed, error=str(exc),
        )


async def run_cases(
    cases: list[EvalCase],
    llm_factory,
    mcp_client: MCPClient,
) -> list[EvalResult]:
    """Run every case in sequence (not concurrently -- Groq free-tier rate
    limits and the single shared MCP server subprocess both make
    sequential execution the safer default; see evaluation/README.md).

    `llm_factory` is a zero-argument callable returning an `LLMProvider`
    for one case. In offline mode this constructs a fresh
    `FakeLLMProvider` per case (each case has its own script); in live
    mode it returns the same real provider instance every time.
    """
    results: list[EvalResult] = []
    for case in cases:
        llm = llm_factory(case)
        results.append(await run_case(case, llm, mcp_client))
    return results
