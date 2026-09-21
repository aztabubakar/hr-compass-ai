"""Live, real-model evaluation run against whichever LLM provider
`LLM_PROVIDER` selects (Gemini or Groq) -- OR, when `EVAL_BASE_URL` is set,
against a deployed HR Compass AI instance's real `/chat` endpoint instead.

Local mode (default, unchanged from before `EVAL_BASE_URL` support existed):

    python -m evaluation.run_live
    python -m evaluation.run_live --limit 5          # smaller subset during development
    python -m evaluation.run_live --cases EVAL-006 EVAL-008   # specific cases only

Deployed mode (evaluates the actual hosted app; no local RAG index, MCP
client, or LLM provider is constructed -- see `evaluator.run_case_http`):

    EVAL_BASE_URL=https://hr-compass-ai-cosv.onrender.com \\
    python -m evaluation.run_live --delay-between-cases 15

See `evaluation/README.md`'s "Deployed mode" section for what this does and
does not prove, and how it differs methodologically from local mode --
most notably for EVAL-015/EVAL-023, the two cases with
`conversation_context` (their prior turns are replayed through the real
deployed app rather than injected as scripted history; see
`evaluator.run_case_http`'s docstring).

Pacing and resuming a rate-limited run (applies to either mode -- both
ultimately call the same rate-limited Gemini API, one directly and one via
the deployed app):

    python -m evaluation.run_live --delay-between-cases 15
    python -m evaluation.run_live --resume evaluation/results/live_checkpoint_20260101T000000Z.json

`--delay-between-cases SECONDS` (default 0) sleeps *after* each completed
case, outside that case's measured latency -- a pacing knob only; it never
touches scoring, expected results, or per-case latency, and it does not
guarantee avoiding a provider's rate limit, only reduces how fast requests
are fired.

Every run writes an incremental checkpoint after each completed case (see
`_write_checkpoint`): to `evaluation/results/live_checkpoint_<timestamp>.json`
by default for a fresh run, or to the path given by `--resume` when
continuing one. A run interrupted by quota exhaustion, a timeout, or a
crash can be continued with `--resume CHECKPOINT_FILE`: every case_id
already present in that file -- whether it previously succeeded or ended
in `error` -- is skipped (never re-run), and its original recorded result
is carried forward into the final aggregate unchanged. Resuming never
retries a failure; see `_load_checkpoint`'s docstring.

Requires the configured provider's API key in local mode (`GEMINI_API_KEY`
or `GROQ_API_KEY`, matching `LLM_PROVIDER`) and a prebuilt RAG index
(`python -m rag.ingest`); neither is required in deployed mode, since the
deployed app owns both. If local mode is unconfigured, this script prints
setup guidance and exits cleanly (exit code 0) rather than failing -- it is
a manual/optional tool, never run in CI.

Does not store or print any API key or environment secret anywhere,
including in a checkpoint or results file's metadata (only the provider
*name*, never a key, and -- in deployed mode -- no model name at all, since
`/health` does not expose one). Any per-case error (timeout, rate limit,
auth, or a deployed-endpoint HTTP failure) is caught by
`evaluation/evaluator.py` and recorded as that case's `error` field rather
than aborting the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx

from evaluation.cases import EvalCase, load_cases
from evaluation.evaluator import DEFAULT_HTTP_TIMEOUT_SECONDS, EvalResult, run_case, run_case_http
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
        "Exiting cleanly (this is expected/normal without a key). Alternatively, set "
        "EVAL_BASE_URL to a deployed instance's URL to evaluate it over HTTP instead -- "
        "see evaluation/README.md's \"Deployed mode\" section."
    ),
    "groq": (
        "GROQ_API_KEY is not set, so live evaluation cannot run.\n\n"
        "To run it:\n"
        "  1. Get a free API key at https://console.groq.com/keys\n"
        "  2. Copy .env.example to .env and set GROQ_API_KEY (and optionally GROQ_MODEL)\n"
        "     or export GROQ_API_KEY in your shell.\n"
        "  3. Build the RAG index if you haven't: python -m rag.ingest\n"
        "  4. Re-run: python -m evaluation.run_live\n\n"
        "Exiting cleanly (this is expected/normal without a key). Alternatively, set "
        "EVAL_BASE_URL to a deployed instance's URL to evaluate it over HTTP instead -- "
        "see evaluation/README.md's \"Deployed mode\" section."
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


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Load a previous run's checkpoint `results` list (plain dicts, not
    yet `EvalResult` objects) for `--resume`.

    Every case_id present here -- whether that case previously succeeded
    or ended in `error` -- is treated as already completed and is never
    re-run; its original recorded result (error included, unmodified) is
    carried forward into the final aggregate exactly as it was. This is a
    deliberate, narrow contract: resuming continues a run past whatever
    happened, it does not retry failures and must never silently turn an
    errored case into a passing one. To force a specific case to be
    re-attempted, remove its entry from the checkpoint file first.
    """
    if not path.exists():
        raise FileNotFoundError(f"--resume checkpoint file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results", [])


def _write_checkpoint(path: Path, metadata: dict[str, Any], result_dicts: list[dict[str, Any]]) -> None:
    """Overwrite `path` with every result completed so far (already-resumed
    ones plus this run's new ones), so a run interrupted at any point --
    quota exhaustion, a timeout, Ctrl-C -- can be continued with
    `--resume path`. Written atomically (a temp file in the same
    directory, then an atomic rename) so a reader never observes a
    half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": result_dicts}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


async def _http_provider_label(base_url: str, http_client: httpx.AsyncClient) -> str | None:
    """Best-effort, read-only lookup of the deployed app's configured LLM
    provider *name* (never a model string or a key) via its existing,
    unmodified `/health` endpoint -- for results-file metadata only.
    `/health` does not report the exact model in use (see
    `app/models.py`'s `LlmHealth`), so `model` stays unknown in deployed
    mode; that is a metadata gap, not a functional one. Never raises: a
    `/health` failure just leaves the label unknown rather than failing
    the run."""
    try:
        response = await http_client.get(f"{base_url}/health")
        response.raise_for_status()
        return response.json().get("llm", {}).get("provider")
    except Exception:  # noqa: BLE001 -- metadata-only best effort, never fatal
        return None


def _combine(resumed_dicts: list[dict[str, Any]], new_results: list[EvalResult]) -> list[dict[str, Any]]:
    return list(resumed_dicts) + [r.to_dict() for r in new_results]


def _new_http_client() -> httpx.AsyncClient:
    """Construct the `httpx.AsyncClient` used for deployed-mode requests.
    A thin, otherwise-pointless-looking factory kept only as a test seam
    (tests monkeypatch this one function to inject a mocked transport,
    the same pattern `app/main.py`'s `get_llm_provider` FastAPI dependency
    already uses for the same reason) -- production behavior is identical
    to constructing `httpx.AsyncClient` inline."""
    return httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_HTTP_TIMEOUT_SECONDS))


async def _main(
    case_ids: list[str] | None,
    limit: int | None,
    delay_between_cases: float,
    resume_path: Path | None,
    checkpoint_file: Path | None,
) -> int:
    all_cases = load_cases()
    selected = _select_cases(all_cases, case_ids, limit)
    if not selected:
        print("No cases selected.")
        return 0

    resumed_dicts = _load_checkpoint(resume_path) if resume_path is not None else []
    resumed_ids = {r["case_id"] for r in resumed_dicts}
    remaining = [c for c in selected if c.case_id not in resumed_ids]
    if resumed_ids:
        print(f"Resuming from {resume_path}: {len(resumed_ids)} case(s) already completed, "
              f"{len(remaining)} remaining.")

    checkpoint_path = resume_path or checkpoint_file or (
        RESULTS_DIR / f"live_checkpoint_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    base_url = os.getenv("EVAL_BASE_URL")
    execution_mode = "deployed_http" if base_url else "local"
    metadata: dict[str, Any] = {
        "execution_mode": execution_mode,
        "delay_between_cases": delay_between_cases,
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "resumed_from": str(resume_path) if resume_path is not None else None,
    }

    new_results: list[EvalResult] = []

    if base_url:
        # ---- Deployed HTTP mode: no local MCPClient, no local LLM
        # provider, no local RAG index -- see evaluator.run_case_http.
        metadata["base_url"] = base_url
        async with _new_http_client() as http_client:
            metadata["provider"] = await _http_provider_label(base_url, http_client)
            metadata["model"] = None  # not exposed by /health -- see _http_provider_label

            if not remaining:
                print("All selected cases already completed in the checkpoint; nothing to run.")
            else:
                print(f"Using deployed endpoint base_url={base_url} on {len(remaining)} case(s).")

            run_token = uuid.uuid4().hex[:8]
            for index, case in enumerate(remaining):
                # Unique per case AND per run (case_id + a fresh per-run
                # token), so conversation state can never leak between
                # cases, or between this run and any earlier/later one that
                # happens to touch the same case_id (e.g. a re-run after a
                # resume).
                conversation_id = f"eval-{case.case_id}-{run_token}"
                result = await run_case_http(case, base_url, http_client, conversation_id)
                new_results.append(result)
                _write_checkpoint(checkpoint_path, metadata, _combine(resumed_dicts, new_results))
                if delay_between_cases > 0 and index < len(remaining) - 1:
                    await anyio.sleep(delay_between_cases)
    else:
        # ---- Local mode: unchanged from before EVAL_BASE_URL support --
        # same provider construction, same MCPClient, same run_case() calls.
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

        metadata["provider"] = config.provider
        metadata["model"] = llm.model
        metadata["temperature"] = config.temperature

        if not remaining:
            print("All selected cases already completed in the checkpoint; nothing to run.")
        else:
            print(f"Using LLM provider={config.provider} model={llm.model} on {len(remaining)} case(s).")

        try:
            async with MCPClient() as client:
                for index, case in enumerate(remaining):
                    result = await run_case(case, llm, client)
                    new_results.append(result)
                    _write_checkpoint(checkpoint_path, metadata, _combine(resumed_dicts, new_results))
                    if delay_between_cases > 0 and index < len(remaining) - 1:
                        await anyio.sleep(delay_between_cases)
        except MCPClientError as exc:
            print(f"Could not connect to the MCP server: {exc}")
            print("Make sure the RAG index is built (python -m rag.ingest) and try again.")
            return 1

    metadata["run_finished_at"] = datetime.now(timezone.utc).isoformat()

    all_result_dicts = _combine(resumed_dicts, new_results)
    if resumed_dicts:
        # Only reorder when a resume actually merged two separately-ordered
        # sequences -- a plain (no --resume) run keeps its cases in exactly
        # the order _select_cases() returned them, unchanged from before
        # EVAL_BASE_URL/--resume existed.
        canonical_order = {c.case_id: i for i, c in enumerate(all_cases)}
        all_result_dicts.sort(key=lambda d: canonical_order.get(d["case_id"], len(canonical_order)))
    metadata["case_count"] = len(all_result_dicts)

    final_results = [EvalResult(**d) for d in all_result_dicts]
    report = aggregate(all_cases, final_results, mode="live")
    report_dict = report.to_dict()
    _print_report(report_dict)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_label = metadata.get("model") or execution_mode
    note = (
        f"Real-model run against the configured {metadata.get('provider')} model. "
        "No API key is stored in this file."
        if execution_mode == "local" else
        f"Real-model run against the deployed instance at {base_url}. "
        "No API key is stored in this file. Latency includes network round trip, "
        "Render processing, and any cold-start -- see evaluation/README.md's "
        "\"Deployed mode\" section."
    )
    out_path = RESULTS_DIR / f"live_{timestamp}_{model_label}.json"
    out_path.write_text(json.dumps({
        "mode": "live",
        "generated_at": timestamp,
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "temperature": metadata.get("temperature"),
        "note": note,
        "metadata": metadata,
        "aggregate": report_dict,
        "results": all_result_dicts,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Checkpoint file: {checkpoint_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected cases.")
    parser.add_argument("--cases", nargs="+", default=None, help="Run only these case IDs (e.g. EVAL-006 EVAL-008).")
    parser.add_argument(
        "--delay-between-cases", type=float, default=0.0,
        help="Seconds to sleep after each completed case, outside its measured latency (pacing only; "
             "never affects scoring, expected results, or per-case latency). Default 0.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a previous checkpoint file. Case IDs already present there are skipped (never "
             "re-run, including ones that previously errored), and their original recorded results are "
             "kept unchanged in the final aggregate.",
    )
    parser.add_argument(
        "--checkpoint-file", type=Path, default=None,
        help="Where to write the incremental checkpoint for a fresh run (ignored if --resume is given, "
             "which reuses that same file). Defaults to evaluation/results/live_checkpoint_<timestamp>.json.",
    )
    args = parser.parse_args()
    return anyio.run(
        _main, args.cases, args.limit, args.delay_between_cases, args.resume, args.checkpoint_file,
    )


if __name__ == "__main__":
    sys.exit(main())
