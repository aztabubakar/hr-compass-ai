"""In-memory conversation state.

**This store is deliberately ephemeral.** Per this phase's instructions
("do not introduce Redis or a paid database... an in-memory store is
acceptable if documented as ephemeral"), conversation history lives only
in this process's memory: it is lost on restart/redeploy, and — if the
app is ever run with more than one worker process — is **not** shared
across workers (each worker would have its own, inconsistent view of a
given `conversation_id`). Both are acceptable for this project's scope
(a single-worker free-tier deployment, demo-length conversations) but
would need a real shared store (e.g. Redis) before scaling beyond that.

Only the final user message and final assistant answer of each turn are
retained per conversation — not the intermediate tool-call/tool-result
messages the agent exchanged with the LLM while producing that answer.
Replaying every turn's full tool-call chatter into every future turn
would grow unbounded and mostly restates information already folded into
that turn's final answer text, so it is deliberately dropped.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from llm.base import ChatMessage


@dataclass
class ConversationState:
    history: list[ChatMessage] = field(default_factory=list)
    # Whether the most recent assistant turn was a confirmation offer
    # (CONFIRM: ...) the agent is now waiting on. Read by the orchestrator
    # as a genuine, code-enforced gate — not just a prompt instruction —
    # before it will let a create_mock_hr_ticket(confirmed=true) call
    # through: see AgentOrchestrator.run()'s `prior_turn_requires_confirmation`
    # parameter. This means a model cannot skip straight to creating a
    # ticket in a single turn no matter what it decides; the conversation
    # must have passed through an explicit confirmation offer first.
    last_requires_confirmation: bool = False


class ConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get_state(self, conversation_id: str) -> ConversationState:
        with self._lock:
            state = self._conversations.get(conversation_id)
            return ConversationState(list(state.history), state.last_requires_confirmation) if state else ConversationState()

    def append_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_answer: str,
        requires_confirmation: bool,
    ) -> None:
        with self._lock:
            state = self._conversations.setdefault(conversation_id, ConversationState())
            state.history.append(ChatMessage(role="user", content=user_message))
            state.history.append(ChatMessage(role="assistant", content=assistant_answer))
            state.last_requires_confirmation = requires_confirmation

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(conversation_id, None)
