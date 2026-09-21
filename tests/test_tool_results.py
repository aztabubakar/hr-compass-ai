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

from agent.schemas import Citation
from agent.tool_results import (
    CITATION_RELATIVE_SCORE_THRESHOLD,
    attribute_citations,
    citations_from_tool_result,
    summarize_tool_result,
    trim_tool_result_for_llm,
)


def _hit(document_id: str, section: str, score: float, snippet: str = "some policy text") -> dict:
    return {
        "document_id": document_id, "document_title": f"{document_id} title", "section": section,
        "source_path": f"/policies/{document_id.lower()}.md", "snippet": snippet, "score": score,
    }


def _citation(
    document_id: str, section: str = "Some Section", snippet: str = "supporting text",
    source_tool: str = "search_policy_documents",
) -> Citation:
    return Citation(
        document_id=document_id, document_title=f"{document_id} title", section=section, snippet=snippet,
        source_tool=source_tool,
    )


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


# ---------------------------------------------------------------------------
# attribute_citations: final-answer citation attribution (over-citation fix)
#
# candidate_citations here always models the whole turn's accumulated
# evidence -- e.g. a remote-work question that retrieved location, remote-
# work, and security-policy evidence over several tool calls -- and each
# test checks what AgentResponse.citations should be once the model's
# final answer text is known.
# ---------------------------------------------------------------------------
_LOC = _citation("POL-LOC-001")
_RWK = _citation("POL-RWK-001")
_SEC = _citation("POL-SEC-001")


def test_attribution_narrows_to_the_single_document_the_answer_cites() -> None:
    # A. Single-document attribution.
    candidates = [_LOC, _RWK, _SEC]
    answer = "You may work from that state without additional approval, per POL-LOC-001."

    result = attribute_citations(answer, candidates)

    assert [c.document_id for c in result] == ["POL-LOC-001"]


def test_attribution_preserves_a_legitimate_multi_document_answer() -> None:
    # B. Legitimate multi-document attribution: the answer genuinely rests
    # on two of the three candidates -- both must survive, and the unrelated
    # third (never mentioned) must not.
    candidates = [_LOC, _RWK, _SEC]
    answer = (
        "This is within the interstate threshold under POL-LOC-001, but note it also "
        "exceeds the Work From Anywhere allowance in POL-RWK-001, so it must be routed "
        "as a location change request."
    )

    result = attribute_citations(answer, candidates)

    assert {c.document_id for c in result} == {"POL-LOC-001", "POL-RWK-001"}
    assert "POL-SEC-001" not in {c.document_id for c in result}


def test_attribution_falls_back_to_all_candidates_when_answer_cites_nothing_by_id() -> None:
    # C. Fallback behavior: the answer never repeats an exact policy ID
    # (a real, expected case of model formatting variation) -- attribution
    # must not interpret that as "nothing was used" and wipe out valid
    # evidence; it must conservatively keep every candidate, exactly this
    # project's pre-attribution behavior.
    candidates = [_LOC, _RWK]
    answer = "You may work from that state for up to six weeks without additional sign-off."

    result = attribute_citations(answer, candidates)

    assert {c.document_id for c in result} == {"POL-LOC-001", "POL-RWK-001"}


def test_attribution_never_lets_a_fabricated_citation_through() -> None:
    # D. Fabricated citation protection: the model's answer references a
    # policy ID that was never actually retrieved. It must never appear in
    # the result -- attribute_citations only ever selects *from*
    # candidate_citations, so this holds structurally, not just by luck of
    # this particular input. Since the only ID mentioned doesn't match any
    # real candidate, this is also a fallback case (all real candidates are
    # conservatively kept; the fake one can never be among them).
    candidates = [_LOC, _RWK, _SEC]
    answer = "This is approved per POL-FAKE-999."

    result = attribute_citations(answer, candidates)

    result_ids = {c.document_id for c in result}
    assert "POL-FAKE-999" not in result_ids
    assert result_ids == {"POL-LOC-001", "POL-RWK-001", "POL-SEC-001"}  # conservative fallback


def test_attribution_drops_a_fabricated_id_but_keeps_the_genuine_one_when_mixed() -> None:
    # A mixed case: one real, attributed citation plus one fabricated one in
    # the same answer. The fabricated ID must never leak through, and the
    # genuine one must not be swept into a fallback it doesn't need.
    candidates = [_LOC, _RWK, _SEC]
    answer = "This is approved per POL-LOC-001 and also somehow per POL-FAKE-999."

    result = attribute_citations(answer, candidates)

    assert [c.document_id for c in result] == ["POL-LOC-001"]


def test_attribution_deduplicates_and_preserves_candidate_order() -> None:
    # E. Deduplication / deterministic ordering: a duplicate candidate
    # (e.g. the same section cited via two different tool calls) collapses
    # to one, and survivors keep their original relative order rather than
    # being re-sorted (here, RWK before LOC, matching the input order).
    duplicate_rwk = _citation("POL-RWK-001")
    candidates = [_RWK, duplicate_rwk, _LOC, _SEC]
    answer = "Per POL-RWK-001 and POL-LOC-001, this is within the allowance."

    result = attribute_citations(answer, candidates)

    assert [c.document_id for c in result] == ["POL-RWK-001", "POL-LOC-001"]


def test_attribution_never_matches_by_substring() -> None:
    # Guard against confusing similarly-named IDs: a candidate whose ID is a
    # prefix of another real-looking token in the answer must not be pulled
    # in by a substring match -- POL-LOC-0011 is a different, unrelated
    # document from POL-LOC-001 and is not even a real candidate here.
    candidates = [_LOC]
    answer = "This references POL-LOC-0011, an unrelated document."

    result = attribute_citations(answer, candidates)

    # POL-LOC-001 was never actually mentioned (only the longer, distinct
    # POL-LOC-0011 token was) -- no overlap, so this is a fallback case, and
    # the one real candidate is conservatively kept, not excluded.
    assert [c.document_id for c in result] == ["POL-LOC-001"]


def test_attribution_returns_empty_for_empty_candidates() -> None:
    assert attribute_citations("Per POL-LOC-001.", []) == []


# ---------------------------------------------------------------------------
# Section-level attribution: document-ID attribution alone is not enough
# when several sections of the *same* document are retrieved together (the
# production over-citation case -- see agent/tool_results.py's
# _filter_to_supporting_sections). Real section titles/snippets from this
# project's actual policy corpus are used throughout so these tests are
# grounded in genuine, not synthetic, content.
# ---------------------------------------------------------------------------
_CON_SECTIONS = {
    "Anti-Harassment and Anti-Discrimination": (
        "Harassment or discrimination based on any protected characteristic is strictly "
        "prohibited, whether committed by a manager, coworker, customer, or vendor, and "
        "regardless of whether it occurs in person, in writing, or through digital "
        "communication tools. Prohibited conduct includes unwelcome sexual advances, "
        "derogatory comments or jokes, and creation of a hostile work environment, and "
        "applies equally at in-person offices, virtual meetings, and offsite company events."
    ),
    "Roles and Responsibilities": (
        "The People & Culture Team owns this policy, operates the reporting channels "
        "described in Section 5, and leads or coordinates investigations. Every employee, "
        "manager, and contractor is responsible for treating others with respect and for "
        "reporting suspected violations rather than ignoring them."
    ),
    "Reporting Procedures": (
        "Employees who experience or witness conduct that may violate this policy should "
        "report it to their manager, any People & Culture Team member, or through the "
        "confidential Solara Ethics Hotline, which accepts anonymous reports and is "
        "operated by an independent third party. Reports may be made verbally or in "
        "writing, and employees are never required to confront the individual involved "
        "directly before reporting."
    ),
    "Frequently Asked Questions": (
        "Can I report a concern anonymously? Yes, the Solara Ethics Hotline described in "
        "Section 5 accepts anonymous reports through an independent third party; you do "
        "not have to give your name to have a concern investigated."
    ),
    "Disciplinary Action": (
        "Violations of this policy may result in disciplinary action up to and including "
        "termination of employment, regardless of the violator's position or tenure. The "
        "severity of disciplinary action considers the nature and impact of the violation "
        "and any prior conduct history, and is determined in consultation with the People "
        "& Culture Team and, where appropriate, Legal."
    ),
}


def _con_candidates() -> list[Citation]:
    return [_citation("POL-CON-001", section, snippet) for section, snippet in _CON_SECTIONS.items()]


def test_a_harassment_case_narrows_away_from_unused_sections() -> None:
    """A. The exact production over-citation scenario: all 5 POL-CON-001
    sections were retrieved by one search_policy_documents call, but the
    final answer only grounds reporting-to-People&Culture/Ethics-Hotline
    claims. Disciplinary Action and Roles and Responsibilities (neither
    materially reflected in the answer) must not survive."""
    answer = (
        "I can create a mock HR ticket for you to report this concern to the People & "
        "Culture Team, or you can contact them directly or use the confidential Solara "
        "Ethics Hotline as outlined in our Workplace Conduct & Anti-Harassment Policy "
        "(POL-CON-001). Would you like me to create a mock HR ticket for you?"
    )

    result = attribute_citations(answer, _con_candidates())
    sections = {c.section for c in result}

    assert len(result) < 5, "must narrow away from the full retrieved set"
    assert "Reporting Procedures" in sections  # directly supports the People&Culture/Hotline claims
    assert "Disciplinary Action" not in sections  # the answer never discusses consequences
    assert "Roles and Responsibilities" not in sections  # the answer never discusses policy ownership


def test_b_benefits_case_prefers_part_time_section_over_unrelated_ones() -> None:
    """B. EMP-1004 part-time medical-eligibility answer: must prefer
    "Benefits for Part-Time and Contract Workers" and may retain "Purpose
    and Eligibility" (materially used here: the answer states the
    full-time/part-time distinction that section itself defines), but must
    never retain Waiting Period, Roles and Responsibilities, or Wellness
    Stipend, none of which this answer discusses."""
    candidates = [
        _citation("POL-BEN-001", "Purpose and Eligibility", (
            "Full-time employees (scheduled 30 or more hours per week) are eligible for "
            "medical, dental, and vision coverage beginning the first of the month "
            "following 30 days of continuous employment. Part-time employees scheduled "
            "under 30 hours per week are not eligible for medical, dental, or vision "
            "coverage but are eligible for the 401(k) plan described in Section 4."
        )),
        _citation("POL-BEN-001", "Waiting Period", (
            "New employees must complete a 30-calendar-day waiting period before medical, "
            "dental, and vision coverage begins. Coverage becomes effective on the first "
            "day of the month following the waiting period."
        )),
        _citation("POL-BEN-001", "Benefits for Part-Time and Contract Workers", (
            "As described in Section 1, part-time employees are eligible only for the "
            "401(k) plan and are not eligible for medical, dental, vision, life, or "
            "disability insurance, or the wellness stipend. Contractors and EOR-employed "
            "international staff are not eligible for any benefit described in this "
            "policy, regardless of how many hours they work."
        )),
        _citation("POL-BEN-001", "Wellness Stipend", (
            "Benefits-eligible employees receive a $50 per month wellness stipend, "
            "reimbursable through SolaraExpense, for expenses such as gym memberships, "
            "fitness classes, or mental health app subscriptions."
        )),
        _citation("POL-BEN-001", "Roles and Responsibilities", (
            "The People & Culture Team owns benefits plan selection, runs annual open "
            "enrollment, and is the point of contact for eligibility questions and "
            "qualifying life event documentation. Payroll administers premium deductions, "
            "FSA contributions, and 401(k) contribution processing each pay period."
        )),
    ]
    answer = (
        "You are not currently eligible for medical, dental, or vision benefits as a "
        "part-time employee, since Solara only offers those to full-time employees; you "
        "are still eligible to enroll in the 401(k) plan."
    )

    result = attribute_citations(answer, candidates)
    sections = {c.section for c in result}

    assert "Benefits for Part-Time and Contract Workers" in sections
    assert "Waiting Period" not in sections
    assert "Roles and Responsibilities" not in sections
    assert "Wellness Stipend" not in sections


def _hol_candidates() -> list[Citation]:
    sections = {
        "Observed U.S. Holidays": (
            "Solara Dynamics observes the following 10 paid holidays each year for all "
            "U.S.-based employees, regardless of work location, work arrangement, or "
            "employment type (full-time or part-time): New Year's Day, Martin Luther King "
            "Jr. Day, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving "
            "Day, Day after Thanksgiving, Christmas Eve, Christmas Day."
        ),
        "Floating Holidays": (
            "In addition to the 10 fixed holidays, each U.S.-based employee receives 2 "
            "floating holidays per calendar year to use for holidays of personal, "
            "cultural, or religious significance not on the standard list above."
        ),
        "Holidays for Non-U.S. Employees and Contractors": (
            "Employees engaged through an Employer of Record outside the United States "
            "observe the statutory public holiday schedule of their country of "
            "employment, as documented in their local EOR employment agreement, rather "
            "than the U.S. holiday list in Section 2."
        ),
        "Winter Office Closure": (
            "Solara Dynamics offices are closed from December 26 through December 31 each "
            "year ('Winter Recess'). These days are paid company closure days for all "
            "U.S.-based employees and are not deducted from PTO balances."
        ),
        "Roles and Responsibilities": (
            "The People & Culture Team owns the annual holiday calendar, publishes "
            "observed dates each November, and confirms local public holiday schedules "
            "for EOR-employed international staff each year."
        ),
    }
    return [_citation("POL-HOL-001", section, snippet) for section, snippet in sections.items()]


def test_c_holiday_case_keeps_us_section_and_drops_international_when_unmentioned() -> None:
    """C. A U.S.-only holiday answer must retain Observed U.S. Holidays and
    must not retain the international/EOR section, since the answer never
    discusses international/EOR holiday treatment."""
    answer = (
        "Solara Dynamics observes 10 fixed U.S. holidays each year, including New Year's "
        "Day, Independence Day, and Christmas Day."
    )

    result = attribute_citations(answer, _hol_candidates())
    sections = {c.section for c in result}

    assert "Observed U.S. Holidays" in sections
    assert "Holidays for Non-U.S. Employees and Contractors" not in sections
    assert "Roles and Responsibilities" not in sections


def test_c_holiday_case_keeps_international_section_when_actually_discussed() -> None:
    """C, continued: when the answer genuinely turns on the international/EOR
    treatment, that section must survive (and dominates, since the answer
    doesn't discuss the U.S. holiday list at all here)."""
    answer = (
        "Since you are engaged through Solara's UK Employer of Record, you observe UK "
        "statutory public holidays under your local EOR employment agreement rather than "
        "the standard U.S. holiday list."
    )

    result = attribute_citations(answer, _hol_candidates())
    sections = {c.section for c in result}

    assert "Holidays for Non-U.S. Employees and Contractors" in sections


def _remote_work_candidates() -> list[Citation]:
    loc_sections = {
        "Payroll-Registered States": (
            "Solara Dynamics is currently payroll-registered in the following states: "
            "Texas, Colorado, North Carolina, California, New York, Washington, Illinois, "
            "Georgia, Florida, Massachusetts, Virginia, Arizona, Oregon, and Tennessee."
        ),
        "Temporary Interstate Work": (
            "Temporary work from a payroll-registered state for up to 30 consecutive "
            "calendar days requires only manager notice in Compass HR. From 31 to 90 "
            "consecutive calendar days, Payroll and People & Culture Team approval is "
            "required at least 15 business days in advance. More than 90 consecutive "
            "calendar days is treated as a permanent relocation request requiring VP and "
            "People & Culture Team approval."
        ),
        "International Remote Work": (
            "Employees may work temporarily from an approved Employer-of-Record country "
            "(United Kingdom, Canada, Portugal, Ireland) for up to 6 consecutive weeks (30 "
            "business days) per year, subject to Global Mobility review, at least 20 "
            "business days notice, and a signed International Remote Work Attestation."
        ),
        "Roles and Responsibilities": (
            "The People & Culture Team and Payroll jointly review and approve interstate "
            "and international work location requests. Managers are responsible for "
            "confirming business need and notifying People & Culture of any planned "
            "remote work arrangement."
        ),
    }
    rwk_sections = {
        "Temporary Remote Work ('Work From Anywhere')": (
            "Employees may work remotely from a non-primary location for up to 20 "
            "business days per calendar year without a formal work location change, known "
            "as Work From Anywhere. Requests exceeding 20 business days must be submitted "
            "as a formal Work Location Change Request under POL-LOC-001."
        ),
        "Eligibility Criteria": (
            "All full-time employees in good standing are eligible for the Work From "
            "Anywhere allowance; contractors and interns are not eligible."
        ),
        "Frequently Asked Questions": (
            "Does Work From Anywhere require manager approval? Yes, submit the request in "
            "Compass HR at least 5 business days in advance."
        ),
        "Roles and Responsibilities": (
            "Managers approve Work From Anywhere requests in Compass HR. The People & "
            "Culture Team audits usage annually to ensure compliance with the "
            "20-business-day allowance."
        ),
    }
    return (
        [_citation("POL-LOC-001", section, snippet) for section, snippet in loc_sections.items()]
        + [_citation("POL-RWK-001", section, snippet) for section, snippet in rwk_sections.items()]
    )


def test_d_and_e_colorado_remote_work_preserves_smallest_multi_document_set() -> None:
    """D/E. A Colorado interstate-work answer that also invokes the Work
    From Anywhere allowance must keep evidence from *both* POL-LOC-001 and
    POL-RWK-001 (genuine multi-document support, never collapsed to one
    document), while dropping each document's own clearly-irrelevant
    sections (Payroll-Registered States, and both documents' Roles and
    Responsibilities / FAQ / Eligibility Criteria, none of which this
    answer discusses).

    Known limitation, not concealed: POL-LOC-001's "International Remote
    Work" section can still survive alongside "Temporary Interstate Work"
    here, since the two provisions share a lot of procedural vocabulary
    ("approval", "business days", "Payroll") within the same document --
    see _filter_to_supporting_sections's docstring. This is a lexical, not
    semantic, heuristic; that specific narrowing would need a stronger
    (semantic) signal than this deterministic, dependency-free design uses.
    """
    answer = (
        "Working from Colorado for six weeks (42 calendar days) exceeds the "
        "30-calendar-day no-approval threshold under POL-LOC-001, so it needs Payroll "
        "and People & Culture Team approval; this also exceeds the 20-business-day Work "
        "From Anywhere allowance in POL-RWK-001, so it must be handled as a formal "
        "location change request rather than informal remote work."
    )

    result = attribute_citations(answer, _remote_work_candidates())
    by_document: dict[str, set[str]] = {}
    for c in result:
        by_document.setdefault(c.document_id, set()).add(c.section)

    # Multi-document support: both documents must contribute at least one citation.
    assert set(by_document) == {"POL-LOC-001", "POL-RWK-001"}
    assert "Temporary Interstate Work" in by_document["POL-LOC-001"]
    assert "Payroll-Registered States" not in by_document["POL-LOC-001"]
    assert "Roles and Responsibilities" not in by_document["POL-LOC-001"]
    # Known limitation (see this test's docstring): International Remote Work
    # still survives alongside Temporary Interstate Work here.
    assert by_document["POL-LOC-001"] == {"Temporary Interstate Work", "International Remote Work"}
    # RWK narrows cleanly to just the one section actually used.
    assert by_document["POL-RWK-001"] == {"Temporary Remote Work ('Work From Anywhere')"}


def test_f_weak_signal_falls_back_to_the_full_document_level_set() -> None:
    """F. When the final answer shares no lexical signal at all with any of
    a document's retrieved sections (a real case: a paraphrased answer that
    never repeats a section's distinctive vocabulary), section-level
    filtering must not guess -- it keeps every section from that document
    that document-level attribution already let through, per
    _filter_to_supporting_sections's conservative-fallback rule."""
    candidates = [
        _citation("POL-XYZ-001", "Some Section", "Completely unrelated boilerplate wording zzz qqq."),
        _citation("POL-XYZ-001", "Another Section", "More unrelated boilerplate wording aaa bbb."),
    ]
    answer = "Yes, that is permitted per POL-XYZ-001."

    result = attribute_citations(answer, candidates)

    assert {c.section for c in result} == {"Some Section", "Another Section"}


def test_g_exact_get_policy_section_outranks_broad_search_hit_from_same_document() -> None:
    """G. An exact get_policy_section fetch must survive even when its own
    lexical score would otherwise fall below the document's relative
    threshold -- the model asking for that section by name is itself strong
    evidence of intentional use."""
    candidates = [
        _citation(
            "POL-PTO-001", "Requesting and Approving PTO",
            "Employees submit PTO requests through Compass HR at least 5 business days in "
            "advance for manager approval.",
            source_tool="search_policy_documents",
        ),
        _citation(
            "POL-PTO-001", "Roles and Responsibilities",
            "The People & Culture Team administers the PTO program and audits balances "
            "quarterly.",
            source_tool="get_policy_section",
        ),
    ]
    answer = "To request PTO, submit it through Compass HR for manager approval at least 5 business days ahead."

    result = attribute_citations(answer, candidates)
    sections = {c.section for c in result}

    assert "Requesting and Approving PTO" in sections  # the clearly on-topic section
    assert "Roles and Responsibilities" in sections  # kept solely on exact-section authority


def test_dedup_prefers_the_authoritative_get_policy_section_citation() -> None:
    """Deduplication by (document_id, section): when the same section is
    reached twice -- once as a broad search_policy_documents chunk, once as
    an exact get_policy_section fetch -- the authoritative, fuller citation
    wins, not whichever happened to be collected first."""
    short = _citation("POL-PTO-001", "Eligibility", "All full-time employees are eligible.",
                       source_tool="search_policy_documents")
    full = _citation(
        "POL-PTO-001", "Eligibility",
        "All full-time employees are eligible for PTO beginning on their hire date; "
        "part-time employees accrue PTO on a prorated basis per Section 9.",
        source_tool="get_policy_section",
    )
    answer = "All full-time employees are eligible for PTO per POL-PTO-001."

    result = attribute_citations(answer, [short, full])

    assert len(result) == 1
    assert result[0].source_tool == "get_policy_section"
    assert result[0].snippet == full.snippet


def test_h_trace_sources_are_unaffected_by_final_citation_attribution() -> None:
    """H. The operational trace records each tool call's own retrieved
    evidence (`TraceRecorder.record_tool_call`'s `sources`) *before* the
    final answer exists, so citation attribution -- which only ever runs
    afterward, on a separately-accumulated citations list -- cannot alter
    it. Proven directly (not through the real hashing-backend RAG pipeline,
    which has too little topic separation to reliably produce multiple
    comparably-scored real sections for one query -- see this module's
    docstring) by recording a trace step from the same 5 candidates used in
    the harassment scenario (test A) and confirming its `sources` count
    stays at 5 regardless of how many of those same citations
    attribute_citations goes on to keep for the final answer."""
    from core.trace import TraceRecorder

    candidates = _con_candidates()
    trace = TraceRecorder()
    trace.record_tool_call(
        "search_policy_documents", {"query": "harassment reporting conduct complaint"},
        status="success", result_summary="5 chunk(s) retrieved from POL-CON-001.",
        sources=[f"{c.document_id} §{c.section}" for c in candidates],
    )

    answer = (
        "I can create a mock HR ticket for you to report this concern to the People & "
        "Culture Team, or you can contact them directly or use the confidential Solara "
        "Ethics Hotline as outlined in our Workplace Conduct & Anti-Harassment Policy "
        "(POL-CON-001). Would you like me to create a mock HR ticket for you?"
    )
    final_citations = attribute_citations(answer, candidates)

    tool_call_step = trace.as_list()[0]
    assert len(tool_call_step["sources"]) == 5  # trace: every retrieved chunk, untouched
    assert len(final_citations) < 5  # AgentResponse.citations: narrowed


def test_structured_data_tools_never_produce_policy_citations() -> None:
    """Structured-data facts (employee type, benefit enrollment, PTO
    balance) must never be turned into policy citations -- this is already
    true of citations_from_tool_result (it only handles the three
    policy-evidence tools), unaffected by attribute_citations, which can
    only ever narrow an already-empty set further."""
    assert citations_from_tool_result("lookup_benefits_status", {"has_record": True, "benefits_eligible": False}) == []
    assert citations_from_tool_result("check_pto_balance", {"has_tracked_balance": True, "balance_hours": 20.0}) == []
    assert citations_from_tool_result("lookup_employee_profile", {"employee_id": "EMP-1004"}) == []
