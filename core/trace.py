"""Structured operational trace model.

This is the "receipt" a future agent orchestrator (Phase 3) will attach to
every response so a user or reviewer can see exactly which MCP tools were
called, with what arguments, and what evidence backed the final answer —
**without** exposing the model's hidden chain-of-thought. A trace is a flat
list of factual execution events (what happened), never a narration of why
the model decided something or a copy of any internal reasoning text.

Used today (Phase 2) by the deterministic workflow-proof scripts in
`scripts/demo_workflows.py` to record the sequence of real MCP tool calls
they make, so the two required demo workflows can show a concrete,
citation-bearing trace even before an LLM agent exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

StepType = Literal["tool_call", "decision"]
StepStatus = Literal["success", "error"]


@dataclass(frozen=True)
class TraceStep:
    """One factual event in an operational trace.

    Every field is a plain, JSON-serializable fact about what happened —
    never free-text reasoning. `arguments` should already be the exact,
    minimal arguments sent to the tool (no secrets; this project's tool
    arguments are all non-sensitive synthetic identifiers, so no additional
    redaction is currently needed, but callers should still avoid stuffing
    large payloads into it — see `result_summary`/`sources` for that).
    """

    step: int
    timestamp: str
    type: StepType
    tool: str | None
    arguments: dict[str, Any] | None
    status: StepStatus
    result_summary: str
    sources: list[str] = field(default_factory=list)
    escalation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TraceRecorder:
    """Accumulates `TraceStep`s for one workflow run."""

    def __init__(self) -> None:
        self._steps: list[TraceStep] = []

    def record_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        status: StepStatus,
        result_summary: str,
        sources: list[str] | None = None,
        escalation: str | None = None,
    ) -> TraceStep:
        step = TraceStep(
            step=len(self._steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type="tool_call",
            tool=tool,
            arguments=arguments,
            status=status,
            result_summary=result_summary,
            sources=sources or [],
            escalation=escalation,
        )
        self._steps.append(step)
        return step

    def record_decision(
        self,
        result_summary: str,
        escalation: str | None = None,
        sources: list[str] | None = None,
    ) -> TraceStep:
        """Record a non-tool decision point (e.g. 'routing to remote-work workflow')."""
        step = TraceStep(
            step=len(self._steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            type="decision",
            tool=None,
            arguments=None,
            status="success",
            result_summary=result_summary,
            sources=sources or [],
            escalation=escalation,
        )
        self._steps.append(step)
        return step

    @property
    def steps(self) -> list[TraceStep]:
        return list(self._steps)

    def as_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._steps]
