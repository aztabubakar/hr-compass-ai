"""Shared evaluation harness: runs one `EvalCase` through the real
`AgentOrchestrator` (real MCP client/server, real RAG retrieval) and
records everything needed for scoring -- independent of whether the model
behind it is `FakeLLMProvider` (offline mode) or a real `GroqProvider`
(live mode). `run_offline.py` and `run_live.py` both build on this module;
neither re-implements the run/record logic itself.

`run_case_http()` is a second, independent way to produce the exact same
`EvalResult` shape: instead of constructing a local `AgentOrchestrator` +
`MCPClient`, it exercises a *deployed* HR Compass AI instance's real
`/chat` endpoint over HTTP (see `run_live.py`'s `EVAL_BASE_URL` support and
`evaluation/README.md`'s "Deployed mode" section). It never touches
`AgentOrchestrator`, `MCPClient`, or an LLM provider directly, and it never
raises -- any HTTP-level failure is captured into `EvalResult.error`,
mirroring `run_case()`'s own broad catch. Both functions feed the same
`metrics.score_case()`/`metrics.aggregate()` -- there is deliberately no
separate scoring path for HTTP-mode results.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.orchestrator import AgentOrchestrator
from evaluation.cases import EvalCase
from llm.base import ChatMessage, LLMProvider
from mcp_client.client import MCPClient

# Render's free tier can take 50+ seconds to cold-start an idle instance --
# generous on purpose for evaluation mode only. This is never applied to
# the production /chat endpoint itself (see app/main.py's
# AGENT_RUN_TIMEOUT_SECONDS, which is unrelated and unchanged).
DEFAULT_HTTP_TIMEOUT_SECONDS = 120.0


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


def _tools_called_from_trace(trace: list[dict[str, Any]]) -> list[str]:
    """The one place `tools_called` is derived from a raw trace -- used
    identically by `run_case()` (a local `AgentResponse.trace`) and
    `run_case_http()` (a deployed `/chat` JSON response's `trace`), so the
    two execution modes can never silently diverge on this logic."""
    return [step["tool"] for step in trace if step["type"] == "tool_call"]


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
        tools_called = _tools_called_from_trace(result.trace)
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


async def run_case_http(
    case: EvalCase,
    base_url: str,
    http_client: httpx.AsyncClient,
    conversation_id: str,
) -> EvalResult:
    """Run one case against a deployed HR Compass AI instance's real
    `/chat` endpoint, instead of a local `AgentOrchestrator` + `MCPClient`.
    Produces the exact same `EvalResult` shape as `run_case()`, from the
    exact same `ChatResponse`/`AgentResponse` fields `/chat` already
    returns (see `app/models.py`) -- no new data is required from the
    application, and nothing about `/chat` itself is touched here.

    `conversation_id` must be unique to this one case (never reused across
    cases or across separate evaluation runs) so conversation state on the
    deployed app's `ConversationStore` can never leak between cases -- see
    `run_live.py`, which generates it.

    For a case with `conversation_context` (only EVAL-015 and EVAL-023 as
    of this writing), each prior turn's *user* message is replayed through
    `/chat` first, under this same `conversation_id`, before the case's
    actual `user_input` is sent. The deployed app generates its own real
    intermediate assistant response from that replay; this deliberately
    never tries to force it to reproduce the scripted assistant turn
    `conversation_context` stores for local/offline mode -- there is no
    endpoint that accepts injected history. See `evaluation/README.md`'s
    "Deployed mode" section for the full methodological difference this
    implies versus local mode.

    Never raises: any HTTP-level failure (non-2xx status, connection
    failure, read timeout, malformed JSON) is captured into
    `EvalResult.error`, exactly like `run_case()`'s own broad catch, so one
    bad case never aborts the whole run.
    """
    start = time.perf_counter()
    try:
        for turn in case.conversation_context:
            if turn.role != "user":
                continue
            prior_response = await http_client.post(
                f"{base_url}/chat", json={"message": turn.content, "conversation_id": conversation_id},
            )
            prior_response.raise_for_status()
            prior_response.json()  # surfaces malformed JSON here exactly like the real call below

        response = await http_client.post(
            f"{base_url}/chat", json={"message": case.user_input, "conversation_id": conversation_id},
        )
        response.raise_for_status()
        payload = response.json()
        # Deliberately wall-clock, not a server-reported duration: this is
        # meant to include network round trip, Render processing, and any
        # cold-start -- see evaluator.py's DEFAULT_HTTP_TIMEOUT_SECONDS and
        # evaluation/README.md's "Deployed mode" section. Per-step
        # `duration_ms` values inside `trace` are preserved unchanged for
        # diagnostics; they are not substituted for this overall figure.
        elapsed = time.perf_counter() - start
        trace = payload.get("trace", [])
        return EvalResult(
            case_id=case.case_id, category=case.category, user_input=case.user_input,
            answer=payload.get("answer", ""), citations=payload.get("citations", []),
            trace=trace, tools_called=_tools_called_from_trace(trace),
            requires_clarification=payload.get("requires_clarification", False),
            requires_confirmation=payload.get("requires_confirmation", False),
            escalated=payload.get("escalated", False),
            latency_seconds=elapsed,
        )
    except Exception as exc:  # noqa: BLE001 -- mirrors run_case()'s deliberately broad catch
        elapsed = time.perf_counter() - start
        return EvalResult(
            case_id=case.case_id, category=case.category, user_input=case.user_input,
            answer="", citations=[], trace=[], tools_called=[],
            requires_clarification=False, requires_confirmation=False, escalated=False,
            latency_seconds=elapsed, error=str(exc),
        )
