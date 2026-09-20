"""Unit tests for agent/tool_results.py's citation-precision filtering and
LLM-payload trimming, using hand-constructed tool results with realistic,
well-separated relevance scores.

These are deliberately pure unit tests against synthetic score values
(rather than against the real hashing-backend retriever used elsewhere in
this test suite): the hashing backend is a dependency-free bag-of-words
fallback with genuinely weak topic separation (verified by hand -- an
unrelated "International Travel Approval" section can score within a few
points of a real PTO match), so it isn't a meaningful way to prove this
filtering *logic* is correct. Testing the function directly against
realistic, well-separated scores (representative of what the production
`fastembed` backend's real semantic embeddings actually produce) proves
the algorithm itself, independent of any one backend's embedding quality
-- a concern this project's RAG design already documents as a known
backend trade-off (see design-and-evaluation.md).
"""
from __future__ import annotations

from agent.tool_results import (
    CITATION_RELATIVE_SCORE_THRESHOLD,
    citations_from_tool_result,
    summarize_tool_result,
    trim_tool_result_for_llm,
)


def _hit(document_id: str, section: str, score: float, snippet: str = "some policy text") -> dict:
    return {
        "document_id": document_id, "document_title": f"{document_id} title", "section": section,
        "source_path": f"/policies/{document_id.lower()}.md", "snippet": snippet, "score": score,
    }


# ---------------------------------------------------------------------------
# Citation precision: unrelated low-score hits are dropped
# ---------------------------------------------------------------------------
def test_low_score_unrelated_hits_are_excluded_from_citations() -> None:
    # A realistic PTO search: one strong match, plus noise from unrelated
    # documents that merely filled out the remaining top-k slots.
    result = {
        "query": "PTO request notice and manager approval",
        "results": [
            _hit("POL-PTO-001", "Requesting and Approving PTO", 0.82),
            _hit("POL-HOL-001", "Winter Office Closure", 0.19),
            _hit("POL-EQP-001", "Purpose and Scope", 0.14),
            _hit("POL-LEV-001", "Purpose and Scope", 0.11),
        ],
        "has_sufficient_evidence": True,
    }
    citations = citations_from_tool_result("search_policy_documents", result)

    assert len(citations) == 1
    assert citations[0].document_id == "POL-PTO-001"
    assert citations[0].section == "Requesting and Approving PTO"


def test_relative_threshold_is_documented_and_applied_consistently() -> None:
    top_score = 1.0
    just_above = top_score * CITATION_RELATIVE_SCORE_THRESHOLD + 0.001
    just_below = top_score * CITATION_RELATIVE_SCORE_THRESHOLD - 0.001
    result = {
        "results": [
            _hit("POL-A", "S1", top_score),
            _hit("POL-B", "S2", just_above),
            _hit("POL-C", "S3", just_below),
        ],
    }
    citations = citations_from_tool_result("search_policy_documents", result)
    doc_ids = {c.document_id for c in citations}
    assert doc_ids == {"POL-A", "POL-B"}


def test_top_hit_always_survives_even_with_a_single_weak_result() -> None:
    result = {"results": [_hit("POL-PTO-001", "Eligibility", 0.03)]}
    citations = citations_from_tool_result("search_policy_documents", result)
    assert len(citations) == 1


def test_empty_results_produce_no_citations() -> None:
    assert citations_from_tool_result("search_policy_documents", {"results": []}) == []


# ---------------------------------------------------------------------------
# Recall / multi-document capability preserved
# ---------------------------------------------------------------------------
def test_two_comparably_strong_hits_from_different_documents_are_both_kept() -> None:
    # A genuine multi-document match within one call: both scores are
    # close, so both should survive -- this is what preserves recall for a
    # real multi-topic query, distinct from noise filtering.
    result = {
        "results": [
            _hit("POL-RWK-001", "Temporary Remote Work", 0.71),
            _hit("POL-LOC-001", "Temporary Interstate Work", 0.68),
        ],
    }
    citations = citations_from_tool_result("search_policy_documents", result)
    assert {c.document_id for c in citations} == {"POL-RWK-001", "POL-LOC-001"}


def test_get_policy_section_citation_is_never_filtered() -> None:
    # A single named-section fetch is unambiguous by construction -- there
    # is no top-k noise to filter.
    result = {
        "document_id": "POL-PTO-001", "document_title": "PTO Policy", "section": "Eligibility",
        "content": "All full-time employees are eligible.", "source_path": "/policies/pol-pto-001.md",
        "effective_date": "2024-01-01",
    }
    citations = citations_from_tool_result("get_policy_section", result)
    assert len(citations) == 1
    assert citations[0].section == "Eligibility"


def test_check_policy_compliance_citations_are_never_score_filtered() -> None:
    # gather_citations() already builds this from a small set of targeted,
    # scenario-specific queries -- not a broad top-k dump -- so this
    # project's new relevance filter deliberately does not touch it.
    result = {
        "citations": [
            {"document_id": "POL-PTO-001", "document_title": "PTO Policy", "section": "Eligibility",
             "snippet": "text", "score": 0.05},
        ],
    }
    citations = citations_from_tool_result("check_policy_compliance", result)
    assert len(citations) == 1


# ---------------------------------------------------------------------------
# summarize_tool_result reports filtering honestly
# ---------------------------------------------------------------------------
def test_summary_reports_relevant_count_when_hits_were_filtered() -> None:
    result = {
        "results": [
            _hit("POL-PTO-001", "Eligibility", 0.80),
            _hit("POL-EQP-001", "Purpose and Scope", 0.10),
        ],
    }
    summary = summarize_tool_result("search_policy_documents", result)
    assert "2 chunk(s) retrieved" in summary
    assert "1 judged relevant" in summary
    assert "POL-PTO-001" in summary
    assert "POL-EQP-001" not in summary


def test_summary_omits_relevance_note_when_nothing_was_filtered() -> None:
    result = {"results": [_hit("POL-PTO-001", "Eligibility", 0.80), _hit("POL-PTO-001", "Accrual", 0.78)]}
    summary = summarize_tool_result("search_policy_documents", result)
    assert "judged relevant" not in summary
    assert "2 chunk(s) retrieved" in summary


# ---------------------------------------------------------------------------
# LLM-payload trimming: smaller context, same information for synthesis
# ---------------------------------------------------------------------------
def test_trim_removes_source_path_and_filtered_hits_from_search_results() -> None:
    result = {
        "query": "PTO notice",
        "results": [
            _hit("POL-PTO-001", "Eligibility", 0.80),
            _hit("POL-EQP-001", "Purpose and Scope", 0.05),
        ],
        "has_sufficient_evidence": True,
    }
    trimmed = trim_tool_result_for_llm("search_policy_documents", result)

    assert len(trimmed["results"]) == 1  # the noisy hit is dropped
    assert trimmed["results"][0]["document_id"] == "POL-PTO-001"
    assert "source_path" not in trimmed["results"][0]
    assert trimmed["has_sufficient_evidence"] is True  # untouched, unfiltered field
    assert trimmed["query"] == "PTO notice"


def test_trim_removes_source_path_from_get_policy_section() -> None:
    result = {
        "document_id": "POL-PTO-001", "document_title": "PTO Policy", "section": "Eligibility",
        "content": "All full-time employees are eligible.", "source_path": "/policies/pol-pto-001.md",
        "effective_date": "2024-01-01",
    }
    trimmed = trim_tool_result_for_llm("get_policy_section", result)
    assert "source_path" not in trimmed
    assert trimmed["content"] == "All full-time employees are eligible."


def test_trim_is_a_no_op_for_other_tools() -> None:
    result = {"employee_id": "EMP-1001", "full_name": "Jordan Vance"}
    assert trim_tool_result_for_llm("lookup_employee_profile", result) == result


def test_trim_does_not_mutate_the_original_result() -> None:
    # citations_from_tool_result/summarize_tool_result run against the
    # original, untrimmed result -- trimming must never mutate it in place.
    result = {"results": [_hit("POL-PTO-001", "Eligibility", 0.80), _hit("POL-EQP-001", "Purpose and Scope", 0.05)]}
    trim_tool_result_for_llm("search_policy_documents", result)
    assert len(result["results"]) == 2
    assert "source_path" in result["results"][0]
