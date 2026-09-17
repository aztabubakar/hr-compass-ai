"""Deterministic, offline evaluation run.

    python -m evaluation.run_offline
    python -m evaluation.run_offline --max-cases 5   # fast subset, e.g. for CI

Runs every case in `evaluation/cases.json` through the real
`AgentOrchestrator`, the real MCP server subprocess, and real RAG
retrieval over the actual policy corpus (hashing embedding backend, so it
never needs network access or model weights -- see rag/config.py). The
only thing that isn't real is the "model": each case's tool-call sequence
is scripted in `evaluation/scripts.py` via `FakeLLMProvider`, standing in
for what a well-behaved model should decide.

**This proves the evaluation architecture and the agent/MCP/safety
machinery work end to end -- it is not a measurement of real LLM quality.**
Every script is written to succeed, so a 100% pass rate here means "the
harness and the orchestration code are wired correctly," not "a live
model will answer this well." See `evaluation/run_live.py` for that, and
`evaluation/README.md`'s "Deterministic vs. live" section for the full
distinction. This script never requires GROQ_API_KEY and never makes a
network call, so it is safe to run in CI on every push/PR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anyio

from core.paths import POLICIES_DIR
from evaluation.cases import load_cases
from evaluation.evaluator import run_cases
from evaluation.metrics import aggregate
from evaluation.scripts import get_script
from llm.fake_provider import FakeLLMProvider
from mcp_client.client import MCPClient
from rag.config import RagConfig
from rag.ingest import run_ingest

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _print_report(report_dict: dict) -> None:
    print(f"\n{'=' * 70}\nDETERMINISTIC (OFFLINE) EVALUATION -- architecture check, not a model-quality score\n{'=' * 70}")
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


async def _main(max_cases: int | None) -> int:
    cases = load_cases()
    if max_cases is not None:
        cases = cases[:max_cases]

    def llm_factory(case):
        return FakeLLMProvider(get_script(case.case_id))

    with tempfile.TemporaryDirectory(prefix="hr_compass_eval_index_") as tmp_index_dir:
        index_dir = Path(tmp_index_dir)
        run_ingest(RagConfig(policies_dir=POLICIES_DIR, index_dir=index_dir, embedding_backend="hashing"))
        env = {**os.environ, "EMBEDDING_BACKEND": "hashing", "RAG_INDEX_DIR": str(index_dir)}
        async with MCPClient(env=env) as client:
            results = await run_cases(cases, llm_factory, client)

    report = aggregate(cases, results, mode="offline")
    report_dict = report.to_dict()
    _print_report(report_dict)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"offline_{timestamp}.json"
    out_path.write_text(json.dumps({
        "mode": "offline",
        "generated_at": timestamp,
        "note": "Deterministic/scripted run. Proves architecture, not real-model quality. See evaluation/README.md.",
        "aggregate": report_dict,
        "results": [r.to_dict() for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 1 if report_dict["errors"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=None, help="Run only the first N cases (for a fast CI smoke test).")
    args = parser.parse_args()
    return anyio.run(_main, args.max_cases)


if __name__ == "__main__":
    sys.exit(main())
