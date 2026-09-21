"""Live, real-model evaluation run against whichever LLM provider
`LLM_PROVIDER` selects (Gemini or Groq).

    python -m evaluation.run_live
    python -m evaluation.run_live --limit 5          # smaller subset during development
    python -m evaluation.run_live --cases EVAL-006 EVAL-008   # specific cases only

Unlike `run_offline.py`, nothing here is scripted: each case's
`user_input` (and `conversation_context`, if any) is handed to the real
`AgentOrchestrator` backed by a real provider (`GeminiProvider` or
`GroqProvider`, per `LLM_PROVIDER` -- see `llm/config.py`), which
independently decides which MCP tools to call, in what order, and how to
answer -- the same code path `app/main.py`'s `/chat` endpoint uses. Scoring
against each case's `expected_tools`/`expected_policy_documents`/rubric is
therefore a genuine (if still approximate -- see evaluation/README.md)
measurement of how the live model actually behaves, not a self-fulfilling
scripted proof like the offline run. Exactly one provider is used for the
whole run -- never mixed mid-run.

Requires the configured provider's API key to be set (`GEMINI_API_KEY` or
`GROQ_API_KEY`, matching `LLM_PROVIDER`). If it isn't, this script prints
setup guidance and exits cleanly (exit code 0) rather than failing -- it is
a manual/optional tool, never run in CI. Requires the RAG index to already
be built (`python -m rag.ingest`); it does not build one for you, since
live evaluation is meant to run against the same index the real deployment
would use, not a hermetic test copy.

Does not store or print the API key anywhere. Any per-case provider error
(timeout, rate limit, auth) is caught by `evaluation/evaluator.py` and
recorded as that case's `error` field rather than aborting the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio

from evaluation.cases import EvalCase, load_cases
from evaluation.evaluator import run_cases
from evaluation.metrics import aggregate
from llm.base import LLMConfigError
from llm.config import LLMConfig, get_provider, is_llm_configured
from mcp_client.client import MCPClient, MCPClientError

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_SETUP_GUIDANCE = {
    "gemini": (
        "GEMINI_API_KEY is not set, so live evaluation cannot run.\n\n"
        "To run it:\n"
        "  1. Get a free API key at https://aistudio.google.com/apikey\n"
        "  2. Copy .env.example to .env and set GEMINI_API_KEY (and optionally GEMINI_MODEL)\n"
        "     or export GEMINI_API_KEY in your shell.\n"
        "  3. Build the RAG index if you haven't: python -m rag.ingest\n"
        "  4. Re-run: python -m evaluation.run_live\n\n"
        "Exiting cleanly (this is expected/normal without a key)."
    ),
    "groq": (
        "GROQ_API_KEY is not set, so live evaluation cannot run.\n\n"
        "To run it:\n"
        "  1. Get a free API key at https://console.groq.com/keys\n"
        "  2. Copy .env.example to .env and set GROQ_API_KEY (and optionally GROQ_MODEL)\n"
        "     or export GROQ_API_KEY in your shell.\n"
        "  3. Build the RAG index if you haven't: python -m rag.ingest\n"
        "  4. Re-run: python -m evaluation.run_live\n\n"
        "Exiting cleanly (this is expected/normal without a key)."
    ),
}


def _select_cases(all_cases: list[EvalCase], case_ids: list[str] | None, limit: int | None) -> list[EvalCase]:
    cases = all_cases
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
        missing = wanted - {c.case_id for c in cases}
        if missing:
            print(f"Warning: unknown case id(s) ignored: {', '.join(sorted(missing))}")
    if limit is not None:
        cases = cases[:limit]
    return cases


def _print_report(report_dict: dict) -> None:
    print(f"\n{'=' * 70}\nLIVE EVALUATION (real model)\n{'=' * 70}")
    print(f"Cases run: {report_dict['total_cases']}  Errors: {report_dict['errors']}")
    for metric in (
        "groundedness_avg", "citation_recall_avg", "citation_precision_avg", "tool_selection_accuracy",
        "workflow_completion_rate", "clarification_accuracy", "escalation_accuracy", "action_safety_pass_rate",
    ):
        value = report_dict[metric]
        print(f"  {metric}: {value:.2f}" if value is not None else f"  {metric}: n/a")
    lat = report_dict["latency"]
    print(f"  latency (s): min={lat['min']:.3f} p50={lat['p50']:.3f} p95={lat['p95']:.3f} max={lat['max']:.3f} mean={lat['mean']:.3f}")
    print("\nBy category:")
    for category, summary in report_dict["by_category"].items():
        print(f"  {category} (n={summary['n']}): "
              f"groundedness={summary['groundedness_avg']}, tools={summary['tool_selection_accuracy']}, "
              f"workflow={summary['workflow_completion_rate']}, clarify={summary['clarification_accuracy']}, "
              f"escalate={summary['escalation_accuracy']}, safety={summary['action_safety_pass_rate']}")


async def _main(case_ids: list[str] | None, limit: int | None) -> int:
    config = LLMConfig()
    if not is_llm_configured(config):
        guidance = _SETUP_GUIDANCE.get(
            config.provider,
            f"LLM_PROVIDER='{config.provider}' is not configured, so live evaluation cannot run.",
        )
        print(guidance)
        return 0

    try:
        llm = get_provider(config)
    except LLMConfigError as exc:
        print(f"LLM configuration error: {exc}")
        return 0

    cases = _select_cases(load_cases(), case_ids, limit)
    if not cases:
        print("No cases selected.")
        return 0

    print(f"Using LLM provider={config.provider} model={llm.model} on {len(cases)} case(s).")

    try:
        async with MCPClient() as client:
            results = await run_cases(cases, lambda _case: llm, client)
    except MCPClientError as exc:
        print(f"Could not connect to the MCP server: {exc}")
        print("Make sure the RAG index is built (python -m rag.ingest) and try again.")
        return 1

    report = aggregate(cases, results, mode="live")
    report_dict = report.to_dict()
    _print_report(report_dict)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"live_{timestamp}_{llm.model}.json"
    out_path.write_text(json.dumps({
        "mode": "live",
        "generated_at": timestamp,
        "provider": config.provider,
        "model": llm.model,
        "temperature": config.temperature,
        "note": f"Real-model run against the configured {config.provider} model. No API key is stored in this file.",
        "aggregate": report_dict,
        "results": [r.to_dict() for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases.")
    parser.add_argument("--cases", nargs="+", default=None, help="Run only these case IDs (e.g. EVAL-006 EVAL-008).")
    args = parser.parse_args()
    return anyio.run(_main, args.cases, args.limit)


if __name__ == "__main__":
    sys.exit(main())
