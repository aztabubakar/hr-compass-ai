"""RAG retrieval ablation: top_k=3 vs. top_k=5.

    python -m evaluation.ablation

This is the project's primary, required ablation/comparison (PROJECT_SPEC:
"Include at least one ablation/comparison"). It compares retrieval
behavior directly (via `rag.retrieval`) rather than going through the full
agent+LLM loop, for two reasons: it isolates the RAG layer's own
characteristics from whatever a given LLM would do with the results, and
it needs no live model, so it's fully deterministic, offline, and free to
re-run (uses the hashing embedding backend over the real policy corpus,
same as the automated test suite).

A representative subset of the evaluation set is used: every case in
`evaluation/cases.json` that declares `expected_policy_documents` (i.e.
where "the right document(s) were retrieved" is actually checkable), using
each case's `user_input` as the retrieval query.

Metrics compared per top_k:
  - doc hit rate: fraction of queries where at least one expected document
    appears somewhere in the top-k results.
  - doc recall: fraction of each query's expected documents that appear in
    the top-k results, averaged across queries.
  - mean top score: average cosine-similarity score of the single
    best-ranked result per query (a groundedness proxy -- higher means the
    retriever is more confident its best match is actually relevant).
  - latency: wall-clock time per query.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from core.paths import POLICIES_DIR
from evaluation.cases import load_cases
from rag.config import RagConfig
from rag.ingest import run_ingest
from rag.retrieval import PolicyRetriever

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_K_VALUES = (3, 5)


def _representative_queries() -> list[tuple[str, list[str]]]:
    cases = load_cases()
    return [
        (case.user_input, case.expected_policy_documents)
        for case in cases
        if case.expected_policy_documents
    ]


def _run_for_top_k(retriever: PolicyRetriever, queries: list[tuple[str, list[str]]], top_k: int) -> dict:
    hits = 0
    recalls: list[float] = []
    top_scores: list[float] = []
    latencies: list[float] = []

    for query_text, expected_docs in queries:
        start = time.perf_counter()
        response = retriever.query(query_text, top_k=top_k)
        latencies.append(time.perf_counter() - start)

        retrieved_docs = {r.doc_id for r in response.results}
        expected = set(expected_docs)
        found = expected & retrieved_docs
        if found:
            hits += 1
        recalls.append(len(found) / len(expected))
        top_scores.append(response.results[0].score if response.results else 0.0)

    n = len(queries)
    return {
        "top_k": top_k,
        "n_queries": n,
        "doc_hit_rate": hits / n if n else 0.0,
        "doc_recall_avg": statistics.mean(recalls) if recalls else 0.0,
        "mean_top_score": statistics.mean(top_scores) if top_scores else 0.0,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else 0.0,
        "latency_p95_seconds": sorted(latencies)[int(round(0.95 * (len(latencies) - 1)))] if latencies else 0.0,
    }


def _print_report(queries: list[tuple[str, list[str]]], per_k: list[dict]) -> None:
    print(f"\n{'=' * 70}\nRAG RETRIEVAL ABLATION: top_k=3 vs top_k=5\n{'=' * 70}")
    print(f"Representative subset: {len(queries)} queries drawn from evaluation/cases.json "
          f"(cases with expected_policy_documents).\n")
    header = f"{'top_k':>6} | {'doc_hit_rate':>12} | {'doc_recall':>10} | {'mean_top_score':>14} | {'mean_latency_ms':>16} | {'p95_latency_ms':>15}"
    print(header)
    print("-" * len(header))
    for row in per_k:
        print(f"{row['top_k']:>6} | {row['doc_hit_rate']:>12.2f} | {row['doc_recall_avg']:>10.2f} | "
              f"{row['mean_top_score']:>14.3f} | {row['latency_mean_seconds'] * 1000:>16.2f} | "
              f"{row['latency_p95_seconds'] * 1000:>15.2f}")
    print(
        "\nInterpretation: a higher top_k widens the candidate pool the LLM's chosen tool call sees, which "
        "can raise doc_hit_rate/doc_recall (more chances to include the right document) at the cost of more "
        "tokens per tool result and slightly higher retrieval latency. It does not by itself improve "
        "mean_top_score, since that reflects only the single best-ranked result, which top_k does not change."
    )


def main() -> int:
    queries = _representative_queries()
    if not queries:
        print("No evaluation cases declare expected_policy_documents; nothing to compare.")
        return 1

    with tempfile.TemporaryDirectory(prefix="hr_compass_ablation_index_") as tmp_index_dir:
        config = RagConfig(policies_dir=POLICIES_DIR, index_dir=Path(tmp_index_dir), embedding_backend="hashing")
        run_ingest(config)
        retriever = PolicyRetriever(config)

        per_k = [_run_for_top_k(retriever, queries, k) for k in TOP_K_VALUES]

    _print_report(queries, per_k)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"ablation_topk_{timestamp}.json"
    out_path.write_text(json.dumps({
        "comparison": "top_k=3 vs top_k=5 retrieval",
        "generated_at": timestamp,
        "embedding_backend": "hashing",
        "n_queries": len(queries),
        "queries": [q for q, _ in queries],
        "results": per_k,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
