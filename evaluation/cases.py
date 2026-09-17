"""Typed loader for the 25 structured evaluation cases in `cases.json`.

`cases.json` is the single source of truth for the evaluation set (case
metadata, expected tools/documents/flags, and rubric keywords). It is plain
JSON -- deliberately -- so it can be read, diffed, and reviewed without
importing any Python. The *deterministic scripted responses* used to drive
`FakeLLMProvider` in offline mode live separately, in `evaluation/scripts.py`,
since a scripted tool-call sequence isn't itself evaluation data.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

CASES_PATH = Path(__file__).resolve().parent / "cases.json"


class ConversationTurn(BaseModel):
    role: str
    content: str


class EvalCase(BaseModel):
    """One evaluation case. Field meanings mirror the Phase 4 spec:

    - `expected_tools`: tools the agent should call at some point (a subset
      check against the actual trace, not an exact-order match, since real
      models may sequence calls differently while still doing the right
      thing).
    - `forbidden_tools`: tools that must NOT appear in the trace.
    - `expected_policy_documents` / `expected_citation_sections`: used for
      citation accuracy/completeness; sections are only checked "where
      practical" (many cases leave this empty).
    - `gold_answer`/`rubric_keywords`: intentionally NOT an exact-match
      target. `rubric_keywords` is a small set of phrases a *correct* answer
      should contain; groundedness/quality is scored as keyword coverage,
      not string equality -- see evaluation/metrics.py and its documented
      limitations.
    - `prior_turn_requires_confirmation`: fed straight into
      `AgentOrchestrator.run(prior_turn_requires_confirmation=...)` for the
      action-safety cases (category N), reproducing the code-level
      confirmation gate's precondition.
    """

    case_id: str
    category: str
    user_input: str
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    expected_behavior: str
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_policy_documents: list[str] = Field(default_factory=list)
    expected_citation_sections: list[str] = Field(default_factory=list)
    requires_clarification: bool = False
    requires_escalation: bool = False
    expected_action_safety: str | None = None
    prior_turn_requires_confirmation: bool = False
    gold_answer: str | None = None
    rubric_keywords: list[str] = Field(default_factory=list)
    notes: str = ""


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in data["cases"]]


def cases_by_category(cases: list[EvalCase] | None = None) -> dict[str, list[EvalCase]]:
    cases = cases if cases is not None else load_cases()
    by_category: dict[str, list[EvalCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)
    return by_category
