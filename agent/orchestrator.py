"""The agent orchestrator: a bounded LLM tool-calling loop.

**This is the one place tool sequencing is decided, and it is not decided
here.** `AgentOrchestrator.run()` does not contain any `if PTO: call A;
call B` branching — on every iteration it hands the LLM the discovered
MCP tool schemas and the conversation so far, and the LLM alone decides
whether to call a tool (and which one, with what arguments), ask a
clarifying question, or give a final answer. The orchestrator's job is
only to: discover tools through the real MCP client, execute whatever
tool call the model requests (through that same client — never by
importing `mcp_server.tools`), feed the result back, record a trace step,
and enforce a hard iteration cap so a confused model can't loop forever.

The deterministic scripts in `scripts/demo_workflows.py` (Phase 2) still
exist and still hard-code their call sequence — they remain useful as a
tool-layer smoke test and as fixtures for the demo, but they are not used
by, or by, this orchestrator.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agent.prompts import SYSTEM_PROMPT
from agent.schemas import AgentResponse, Citation
from agent.tool_results import citations_from_tool_result, dedupe_citations, escalation_from_tool_result, summarize_tool_result
from core.trace import TraceRecorder
from llm.base import ChatMessage, LLMProvider, LLMProviderError, ToolDefinition
from mcp_client.client import MCPClient, MCPClientError, MCPToolError

logger = logging.getLogger("hr_compass_ai.agent.orchestrator")

DEFAULT_MAX_ITERATIONS = 10

# Tools where a "confirmed" argument gates a real (if synthetic) state
# change. Kept as a set, not a single name, so a future action-like tool
# (per CLAUDE.md: "the same principle should apply to any future
# action-like tool") only needs to be added here to get the same
# code-enforced confirmation gate, not a new branch of logic.
_CONFIRMATION_GATED_TOOLS = {"create_mock_hr_ticket"}

_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching HR systems right now, so I can't complete this "
    "request. Please try again in a moment, or contact HR directly for help."
)
_MAX_ITERATIONS_MESSAGE = (
    "I wasn't able to finish gathering everything needed for this request within my "
    "step budget. Please try rephrasing your question, or contact HR directly."
)


def _current_date_context(now: datetime | None = None) -> str:
    """A short, deterministic, server-generated system message giving the
    model an authoritative "today" to anchor relative-date resolution
    against (e.g. "next week", "tomorrow", "in two weeks" in a PTO
    request). Without this, the model has no reliable notion of the
    current date and cannot compute a real notice_business_days figure for
    check_policy_compliance -- it can only hedge ("if you can meet the
    notice requirement"). Kept as a separate system message, not baked
    into the static, versioned SYSTEM_PROMPT string, since this one line
    changes on every request.

    UTC is used (not a per-employee or server-local timezone) since this
    project has no per-user timezone data to ground a "local" date in;
    documented here as a known simplification.
    """
    now = now or datetime.now(timezone.utc)
    return (
        f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}), UTC. Use this as "
        "\"today\" when resolving any relative date or time reference in the user's request "
        "(for example \"next week\", \"tomorrow\", \"in two weeks\") into a concrete number of "
        "business days for check_policy_compliance's notice_business_days argument."
    )


def _parse_control_prefix(text: str) -> tuple[bool, bool, str]:
    """Recognize the two reserved response prefixes documented in the
    system prompt (CLARIFY:/CONFIRM:) and strip them from the visible
    answer. Anything else is a normal final answer."""
    stripped = text.strip()
    for prefix, flags in (("CLARIFY:", (True, False)), ("CONFIRM:", (False, True))):
        if stripped[: len(prefix)].upper() == prefix:
            return flags[0], flags[1], stripped[len(prefix):].strip()
    return False, False, stripped


class AgentOrchestrator:
    """One instance can serve many `run()` calls; it holds no per-request
    state itself (a fresh `TraceRecorder` and message list are built each
    call). Construct with an already-connected `MCPClient` — the caller
    (the FastAPI app, or a test) owns that client's lifecycle."""

    def __init__(
        self,
        llm: LLMProvider,
        mcp_client: MCPClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self.llm = llm
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    async def run(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
        prior_turn_requires_confirmation: bool = False,
    ) -> AgentResponse:
        """Run one turn. `prior_turn_requires_confirmation` should be true
        only when the *immediately preceding* assistant turn in this same
        conversation was a confirmation offer (its `AgentResponse.requires_confirmation`
        was true) — the caller (see `app/main.py`) reads this from
        `ConversationStore`. This is the code-level half of the action-safety
        guardrail: see `_CONFIRMATION_GATED_TOOLS` below for what it gates
        and why prompting alone is not relied on for this.
        """
        trace = TraceRecorder()

        try:
            tools = await self._discover_tools()
        except MCPClientError as exc:
            trace.record_decision(f"MCP tool discovery failed: {exc}", escalation="mcp_unavailable")
            return AgentResponse(answer=_UNAVAILABLE_MESSAGE, trace=trace.as_list(), escalated=True)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="system", content=_current_date_context()),
        ]
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=user_message))

        collected_citations: list[Citation] = []

        for _ in range(self.max_iterations):
            try:
                response = await self.llm.chat(messages, tools=tools)
            except LLMProviderError as exc:
                # The user-facing response below is intentionally generic
                # (never exposes provider internals), so this is the
                # server-side record of *why*. A provider that logs its own
                # sanitized detail (e.g. GroqProvider: exception type, HTTP
                # status code, API error code, model) will already have
                # logged that above this line; this guarantees every LLM
                # failure is visible in server logs even for a provider
                # that doesn't do its own logging.
                logger.error("LLM provider error during agent turn: %s", exc)
                # The trace is returned to, and rendered by, the client's
                # "Agent Activity" panel -- unlike the log line above, it
                # must never carry the raw provider exception text (which
                # can include the upstream API's status/error-code detail).
                # Sanitized here; full detail is server-side only, in logs.
                trace.record_decision(
                    "LLM provider error occurred; unable to reach the language model.",
                    escalation="llm_unavailable",
                )
                return AgentResponse(
                    answer="I'm having trouble reaching the language model right now. Please try again shortly.",
                    citations=collected_citations, snippets=[c.snippet for c in collected_citations],
                    trace=trace.as_list(), escalated=True,
                )

            if not response.message.tool_calls:
                answer_text = response.message.content or ""
                requires_clarification, requires_confirmation, answer = _parse_control_prefix(answer_text)
                escalated = any(step.escalation for step in trace.steps)
                return AgentResponse(
                    answer=answer,
                    citations=collected_citations,
                    snippets=[c.snippet for c in collected_citations],
                    trace=trace.as_list(),
                    requires_clarification=requires_clarification,
                    requires_confirmation=requires_confirmation,
                    escalated=escalated,
                )

            messages.append(response.message)
            try:
                # Each tool call in this batch is handled independently: a
                # clean per-tool MCPToolError still produces a "tool" role
                # result for that call_id (every model-requested tool_call_id
                # must be answered before the next LLM turn, or a
                # OpenAI-compatible API will reject the request), while an
                # MCPClientError (the whole MCP connection is down) aborts
                # the entire turn immediately since no further call in this
                # batch — or any later one — could succeed either.
                for tool_call in response.message.tool_calls:
                    arguments = tool_call.arguments
                    if (
                        tool_call.name in _CONFIRMATION_GATED_TOOLS
                        and arguments.get("confirmed") is True
                        and not prior_turn_requires_confirmation
                    ):
                        # Code-enforced safety net: never let a
                        # confirmed=true call for a gated action through
                        # unless this conversation's immediately preceding
                        # turn was itself a confirmation offer, regardless
                        # of what the model decided. Downgrade to a preview
                        # so the model (and, in the response, the user) sees
                        # exactly what would happen instead.
                        arguments = {**arguments, "confirmed": False}
                    try:
                        result = await self.mcp_client.call_tool(tool_call.name, arguments)
                    except MCPToolError as exc:
                        trace.record_tool_call(tool_call.name, arguments, status="error",
                                                result_summary=exc.message)
                        messages.append(ChatMessage(role="tool", tool_call_id=tool_call.id, name=tool_call.name,
                                                     content=json.dumps({"error": exc.message})))
                        continue

                    summary = summarize_tool_result(tool_call.name, result)
                    sources = [f"{c.document_id} §{c.section}" for c in citations_from_tool_result(tool_call.name, result)]
                    trace.record_tool_call(
                        tool_call.name, arguments, status="success", result_summary=summary,
                        sources=sources, escalation=escalation_from_tool_result(tool_call.name, result),
                    )
                    collected_citations.extend(citations_from_tool_result(tool_call.name, result))
                    collected_citations = dedupe_citations(collected_citations)
                    messages.append(ChatMessage(role="tool", tool_call_id=tool_call.id, name=tool_call.name,
                                                 content=json.dumps(result)))
            except MCPClientError as exc:
                trace.record_decision(f"MCP transport failure: {exc}", escalation="mcp_unavailable")
                return AgentResponse(
                    answer=_UNAVAILABLE_MESSAGE, citations=collected_citations,
                    snippets=[c.snippet for c in collected_citations], trace=trace.as_list(), escalated=True,
                )

        trace.record_decision(
            f"Reached the maximum of {self.max_iterations} tool-call iterations without a final answer.",
            escalation="max_iterations_exceeded",
        )
        return AgentResponse(
            answer=_MAX_ITERATIONS_MESSAGE, citations=collected_citations,
            snippets=[c.snippet for c in collected_citations], trace=trace.as_list(), escalated=True,
        )

    async def _discover_tools(self) -> list[ToolDefinition]:
        mcp_tools = await self.mcp_client.list_tools()
        return [
            ToolDefinition(name=t.name, description=t.description or "", input_schema=t.input_schema)
            for t in mcp_tools
        ]
