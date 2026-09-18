"""Manual, live smoke test for the real Groq-backed agent.

    python -m scripts.agent_smoke_test

This is a **manual** script, not part of the automated test suite, and is
never run in CI: it requires `GROQ_API_KEY` to be set and makes real
network calls to Groq. If `GROQ_API_KEY` is not set, it prints setup
guidance and exits cleanly (exit code 0) rather than failing loudly —
this script's job is to help a human verify the live path works, not to
assert anything in an automated pipeline.

It runs two turns against the real agent (real MCP server subprocess,
real Groq-backed LLMProvider):

1. A policy-only question, to check basic grounded retrieval.
2. The PTO workflow scenario, to check the model chooses tools on its own
   (this script does not tell it which tools to call — that's the whole
   point of a *live* smoke test, versus the deterministic `FakeLLMProvider`
   tests).

For each, it prints the final answer, citations, and operational trace.
"""
from __future__ import annotations

import json
import os
import sys

import anyio

from agent.orchestrator import AgentOrchestrator
from llm.base import LLMConfigError, LLMProviderError
from llm.config import LLMConfig, get_provider
from mcp_client.client import MCPClient, MCPClientError


def _print_result(label: str, result) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print("\nANSWER:\n" + result.answer)
    print(f"\nCITATIONS ({len(result.citations)}):")
    for c in result.citations:
        print(f"  - {c.document_id} §{c.section}: {c.snippet[:100]}")
    print(f"\nOPERATIONAL TRACE ({len(result.trace)} step(s)):")
    print(json.dumps(result.trace, indent=2))
    if result.requires_clarification:
        print("\n[requires_clarification = True]")
    if result.requires_confirmation:
        print("\n[requires_confirmation = True]")
    if result.escalated:
        print("\n[escalated = True]")


async def main() -> int:
    config = LLMConfig()
    if config.provider == "groq" and not config.groq_api_key:
        print(
            "GROQ_API_KEY is not set, so this live smoke test cannot run.\n\n"
            "To run it:\n"
            "  1. Get a free API key at https://console.groq.com/keys\n"
            "  2. Copy .env.example to .env and set GROQ_API_KEY (and optionally GROQ_MODEL)\n"
            "     or export GROQ_API_KEY in your shell.\n"
            "  3. Build the RAG index if you haven't: python -m rag.ingest\n"
            "  4. Re-run: python -m scripts.agent_smoke_test\n\n"
            "Exiting cleanly (this is expected/normal without a key)."
        )
        return 0

    try:
        llm = get_provider(config)
    except LLMConfigError as exc:
        print(f"LLM configuration error: {exc}")
        return 0

    print(f"Using LLM provider={config.provider} model={llm.model}")

    try:
        async with MCPClient() as client:
            orchestrator = AgentOrchestrator(llm, client)

            try:
                result_a = await orchestrator.run("What is Solara's paid holiday policy?")
            except LLMProviderError as exc:
                print(f"\nLLM provider error on the policy question: {exc}")
                return 1
            _print_result("1) Policy-only question", result_a)

            try:
                result_b = await orchestrator.run(
                    "My employee ID is EMP-1001. Can I take three days of PTO next week?"
                )
            except LLMProviderError as exc:
                print(f"\nLLM provider error on the PTO workflow question: {exc}")
                return 1
            _print_result("2) PTO workflow (model-selected tools)", result_b)

    except MCPClientError as exc:
        print(f"Could not connect to the MCP server: {exc}")
        print("Make sure the RAG index is built (python -m rag.ingest) and try again.")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
