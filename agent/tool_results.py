"""Turns a raw MCP tool result (already a plain dict — see
`mcp_client.MCPClient.call_tool`) into the things the agent loop needs for
its trace, structured output, and the conversation it sends the LLM: a
short human-readable summary, a list of citable evidence (filtered for
relevance — see `_filter_relevant_hits`), a trimmed copy safe to feed back
to the model, and whether the result carries an escalation.

This module knows the *shape* of each of the 8 tools' results (since
that's needed to render a readable trace line and to reliably extract
citations without ever trusting the model's own prose), but contains no
policy logic itself and never decides *which* tool to call — that
decision is the LLM's alone, made in `agent/orchestrator.py`.

`attribute_citations` (below) is a second, independent citation-precision
step, distinct from `_filter_relevant_hits`: that filter narrows *one*
`search_policy_documents` call's raw top-k hits down to genuinely relevant
ones (still working from retrieval scores alone). `attribute_citations`
runs once, after the whole turn's evidence has been gathered across
*every* tool call and the model has produced its final answer, and narrows
the accumulated candidate set down to whatever the final answer actually
attributes a claim to. The two are complementary, not redundant: a model
can legitimately retrieve several documents over several calls while
researching a multi-part question (e.g. an interstate-work question that
also checks the international-work allowance for comparison) without
every one of them ending up relevant to what it actually tells the user.
"""
from __future__ import annotations

import re

from agent.schemas import Citation

# Matches this project's document-ID convention (see policies/content.py):
# an all-caps, dash-separated PREFIX-CODE-NUMBER token, e.g. "POL-PTO-001".
# Deliberately not hard-coded to "POL" or any specific category code, so a
# future document-ID prefix still matches without a code change here.
# Word-bounded on both ends and matched by exact set membership against
# real candidate document_ids in `attribute_citations` below -- never by
# substring containment -- so a similarly-named ID can never be confused
# with another (e.g. a hypothetical "POL-LOC-0010" is never mistaken for
# "POL-LOC-001").
POLICY_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}-[A-Z0-9]{2,6}-\d{2,4}\b")

# How much lower than the single best-scoring hit in one search_policy_documents
# call a hit's score may be and still count as genuinely relevant evidence,
# not just whatever filled out the remaining top-k slots. Relative (not an
# absolute score) so it adapts to whatever range the configured embedding
# backend happens to produce; the single best hit always survives (its own
# ratio to itself is 1.0), so this can never drop every result down to zero.
#
# This is deliberately a general "how much noise is in one ranked list"
# heuristic, not a PTO-specific or topic-specific rule -- the exact same
# threshold applies to every search_policy_documents call regardless of
# what it searched for, which is what keeps this a generic filter rather
# than hard-coded topic branching. Genuine multi-document evidence is not
# lost by this: this project's demo workflows (and the tests that prove
# them) already retrieve multi-document evidence via *separate*
# document_id-scoped search_policy_documents/get_policy_section calls, and
# a separate call's own top hit always clears its own bar.
CITATION_RELATIVE_SCORE_THRESHOLD = 0.75

# How much lower than the best-evidenced section *within the same document*
# a candidate's own section-relevance score (see `_section_relevance_score`)
# may be and still count as materially used by the final answer, rather than
# merely belonging to the same, otherwise-relevant document. Relative, per
# document -- exactly the same "how much noise is in this ranked list"
# pattern as CITATION_RELATIVE_SCORE_THRESHOLD above, one stage later: that
# constant separates a relevant document's chunks from another document's
# noise within one search call; this one separates a document's *actually
# cited* sections from its other, merely-retrieved ones. The single
# best-scoring section in a document always survives (its ratio to itself is
# 1.0), so a document with only weak lexical signal everywhere still keeps
# its best candidate rather than losing everything.
SECTION_RELATIVE_SCORE_THRESHOLD = 0.6

# A small, generic function-word list used only to keep _section_relevance_score
# from being dominated by words that appear in nearly every sentence
# regardless of topic. Deliberately not a topic/domain word list of any
# kind -- removing "the" or "and" never depends on what policy or question
# is involved, so this introduces no per-case or per-document tuning.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "as", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "you",
    "your", "yours", "i", "we", "our", "per", "not", "no", "do", "does", "did", "if", "so", "by",
    "at", "from", "can", "will", "would", "should", "may", "might", "must", "have", "has", "had",
    "also", "about", "into", "than", "any", "all", "up", "out", "one", "two",
})
# Tokens must start with a letter -- this excludes bare numbers ("20", "30")
# from scoring. Several unrelated provisions across this project's policy
# corpus happen to use similar-looking day/week thresholds (e.g. a 20- and a
# 30-business-day allowance defined in two different sections), so a bare
# number matching between an answer and an unrelated section is coincidental
# noise, not a genuine content signal, in a way a real content word is not.
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*")
# A modest, fixed bonus (not tuned per case) added to a citation's score when
# its own section *title* shares a word with the answer -- "section-title
# term overlap" is a distinct, independent signal from snippet-body overlap
# (a model may name a section's topic in its own words -- e.g. echoing a
# document's title -- without repeating any of that section's body text).
# Kept small and additive, rather than a large multiplicative weight, so one
# section's title match cannot on its own dominate the relative comparison
# for every other section in the same document (see
# `_filter_to_supporting_sections`).
_TITLE_OVERLAP_BONUS = 0.15


def _normalized_tokens(text: str) -> set[str]:
    """Lowercase, punctuation-and-number-stripped, stopword-filtered word
    tokens -- the smallest normalization that lets `_section_relevance_score`
    compare a citation's own wording against the answer's wording without
    being thrown off by case, punctuation, near-universal filler words, or
    coincidental numeric-threshold overlap (see `_TOKEN_PATTERN`). Not a
    stemmer/lemmatizer -- "approval" and "approved" are different tokens
    here -- which is a deliberate simplicity trade-off (see
    `_section_relevance_score`'s docstring for why this is still robust
    enough for its purpose)."""
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS and len(token) > 1}


def _section_relevance_score(
    content_tokens: set[str], title_tokens: set[str], answer_tokens: set[str], document_frequency: dict[str, int],
) -> float:
    """How much of a candidate citation's own vocabulary (its section title
    plus its retrieved snippet -- the two fields that describe what this
    specific section actually says) is echoed in the final answer text,
    weighted so a word shared by *most sections of the same document* (e.g.
    a policy's standard "People & Culture Team owns this" boilerplate,
    repeated near-verbatim in several sections' Roles/Reporting/Disciplinary
    text) counts for much less than a word that is comparatively distinctive
    *within that document's own candidate set* -- `document_frequency` is
    exactly `_filter_to_supporting_sections`'s per-document token->section-count
    map, an inverse-document-frequency-style weighting computed fresh from
    that document's own candidates each time, never a fixed/hard-coded word
    list of any kind.

    This is deliberately a coarse, deterministic lexical-overlap signal, not
    exact-string or semantic matching: a real model rarely repeats a
    section's exact heading, but an answer genuinely grounded in a section
    tends to reuse several of its distinctive content words (e.g. "Ethics
    Hotline", "anonymous" for a reporting-procedures section), while an
    unrelated section from the same document shares little of its
    *distinctive* vocabulary with the answer, even if both happen to share
    generic organizational boilerplate. `_TITLE_OVERLAP_BONUS` is added on
    top when the section's own title shares a word with the answer -- see
    that constant's docstring.

    Scored as a fraction of the citation's own (weighted) vocabulary so a
    short, focused section isn't penalized relative to a longer one, and
    compared only *relatively*, against other sections of the same document
    (see `_filter_to_supporting_sections`), so no single absolute cutoff
    needs to work identically well for every document or embedding backend.
    """
    if not content_tokens:
        return 0.0
    weight_total = sum(1.0 / document_frequency[token] for token in content_tokens)
    weight_matched = sum(1.0 / document_frequency[token] for token in content_tokens if token in answer_tokens)
    base = (weight_matched / weight_total) if weight_total else 0.0
    title_overlap = bool(title_tokens & answer_tokens)
    return base + (_TITLE_OVERLAP_BONUS if title_overlap else 0.0)


def _filter_to_supporting_sections(answer: str, citations: list[Citation]) -> list[Citation]:
    """Second-stage, section-level narrowing: for each *document* still
    present after `attribute_citations`'s document-level stage, keep only
    the sections whose own content is actually echoed in the final answer,
    relative to that same document's best-evidenced section --
    `SECTION_RELATIVE_SCORE_THRESHOLD` applied to `_section_relevance_score`,
    exactly the same relative-ranking pattern `_filter_relevant_hits` already
    uses one stage earlier in the pipeline. Document-frequency weighting
    (see `_section_relevance_score`) is computed fresh, per document, from
    that document's own surviving candidates only -- never across documents
    and never from any fixed word list -- so it adapts automatically to
    whatever policy corpus is actually loaded.

    Two things are always exempt from being dropped by score alone:
    - A citation from `get_policy_section` (`Citation.source_tool`): the
      model explicitly asked for that exact section by name, which is
      stronger evidence of intentional use than lexical overlap can
      establish either way -- see this function's "exact-section authority"
      handling below.
    - Every section of a document whose sections all score 0 against the
      answer (no lexical signal at all, e.g. the answer never repeats any
      distinctive word from anything retrieved): conservative fallback,
      keeping the whole document-level group unchanged rather than guessing
      with no evidence.

    Never drops a whole document to zero citations while any of its
    sections clears the bar, and never removes a document's *only*
    candidate (a single-section group's one citation always has a
    score-to-itself ratio of 1.0, so it always clears its own threshold).
    Groups (by `document_id`) are processed in first-seen order and each
    group's own relative order is preserved -- no re-sorting.

    Known limitation, stated plainly rather than hidden: this is a lexical
    heuristic, not a semantic one. Two genuinely distinct provisions within
    the same document that both use a lot of shared *procedural* vocabulary
    (e.g. two different approval-threshold sections that both talk about
    "approval", "business days", and "Payroll") can still both clear a
    reasonable relative threshold even when the answer's substance really
    only concerns one of them -- narrowing that further would require a
    semantic (not lexical) comparison, or a much stricter threshold that
    would itself start rejecting genuine multi-section answers elsewhere.
    """
    if not citations:
        return citations

    answer_tokens = _normalized_tokens(answer)

    groups: dict[str, list[Citation]] = {}
    order: list[str] = []
    for citation in citations:
        if citation.document_id not in groups:
            groups[citation.document_id] = []
            order.append(citation.document_id)
        groups[citation.document_id].append(citation)

    kept: list[Citation] = []
    for document_id in order:
        group = groups[document_id]
        content_token_sets = [_normalized_tokens(c.section) | _normalized_tokens(c.snippet) for c in group]
        title_token_sets = [_normalized_tokens(c.section) for c in group]

        document_frequency: dict[str, int] = {}
        for tokens in content_token_sets:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        scored = [
            (citation, _section_relevance_score(content_tokens, title_tokens, answer_tokens, document_frequency))
            for citation, content_tokens, title_tokens in zip(group, content_token_sets, title_token_sets)
        ]
        top_score = max(score for _, score in scored)
        if top_score <= 0.0:
            # No lexical signal anywhere in this document's candidates --
            # can't confidently distinguish "used" from "unused" sections,
            # so conservatively keep the whole document-level group (this is
            # exactly the case rule 5's fallback requirement targets: a
            # grounded answer must never lose all its evidence just because
            # this particular scoring signal came up empty).
            kept.extend(group)
            continue
        threshold = top_score * SECTION_RELATIVE_SCORE_THRESHOLD
        for citation, score in scored:
            if citation.source_tool == "get_policy_section" or score >= threshold:
                kept.append(citation)
    return kept


def _most_useful_citation(a: Citation, b: Citation) -> Citation:
    """Tie-break between two citations for the same (document_id, section)
    -- see `_dedupe_by_document_and_section`. An exact `get_policy_section`
    fetch is preferred over a broader `search_policy_documents` chunk of
    the same section (it named the section explicitly and typically returns
    fuller section content); failing that, the longer/more complete snippet
    wins. Never compares snippet *content* beyond length -- this is a
    tie-break between two citations already known to name the same section,
    not a relevance judgment."""
    a_authoritative = a.source_tool == "get_policy_section"
    b_authoritative = b.source_tool == "get_policy_section"
    if a_authoritative != b_authoritative:
        return a if a_authoritative else b
    return a if len(a.snippet) >= len(b.snippet) else b


def _dedupe_by_document_and_section(citations: list[Citation]) -> list[Citation]:
    """Collapse citations that name the same (document_id, section) -- e.g.
    the same section reaching the candidate set once via
    `search_policy_documents` and once via a later `get_policy_section` call
    -- to the single most useful one (`_most_useful_citation`), rather than
    `dedupe_citations`'s snippet-prefix-sensitive key, which would treat two
    different excerpts of the same section as distinct citations. First-seen
    order is preserved for the surviving, merged entries."""
    best: dict[tuple[str, str], Citation] = {}
    order: list[tuple[str, str]] = []
    for citation in citations:
        key = (citation.document_id, citation.section)
        if key not in best:
            best[key] = citation
            order.append(key)
        else:
            best[key] = _most_useful_citation(citation, best[key])
    return [best[key] for key in order]


def _filter_relevant_hits(hits: list[dict], relative_threshold: float = CITATION_RELATIVE_SCORE_THRESHOLD) -> list[dict]:
    """Keep only the hits from one ranked `search_policy_documents` result
    that are genuinely comparable in relevance to the single best match,
    not just whatever else happened to fill out the remaining top-k slots
    (e.g. an unrelated equipment or holiday-schedule chunk turning up
    alongside a PTO-relevant one). Hits are assumed already ranked
    best-first, as the retriever returns them."""
    if not hits:
        return []
    top_score = hits[0].get("score", 0.0)
    if top_score <= 0:
        return hits[:1]
    return [hit for hit in hits if hit.get("score", 0.0) >= top_score * relative_threshold]


def trim_tool_result_for_llm(tool_name: str, result: dict) -> dict:
    """Return a copy of `result` with fields the model doesn't need for
    reasoning/synthesis removed, before it's added to the conversation
    history that gets resent, in full, on every subsequent turn. This is
    what keeps a request with several tool calls from accumulating an
    unnecessarily large context: `source_path` is a filesystem path the
    model never needs to read or cite (citation source paths are already
    captured server-side via `citations_from_tool_result`, called with the
    *original*, untrimmed result), and a `search_policy_documents` hit
    list is narrowed to the same relevance-filtered subset
    `citations_from_tool_result` uses, so the model reasons over (and the
    conversation carries the token cost of) only the evidence that's
    actually likely to matter.

    The full, untrimmed result is still what `summarize_tool_result`,
    `citations_from_tool_result`, and the operational trace use — this
    function only affects what accumulates in the LLM's own context.
    """
    if tool_name == "search_policy_documents":
        relevant_hits = _filter_relevant_hits(result.get("results", []))
        trimmed_hits = [{k: v for k, v in hit.items() if k != "source_path"} for hit in relevant_hits]
        return {**result, "results": trimmed_hits}
    if tool_name == "get_policy_section":
        return {k: v for k, v in result.items() if k != "source_path"}
    return result


def summarize_tool_result(tool_name: str, result: dict) -> str:
    """A short, human-readable one-liner for the operational trace."""
    if tool_name == "search_policy_documents":
        all_hits = result.get("results", [])
        relevant_hits = _filter_relevant_hits(all_hits)
        docs = sorted({hit["document_id"] for hit in relevant_hits})
        if len(relevant_hits) == len(all_hits):
            return f"{len(all_hits)} chunk(s) retrieved" + (f" from {', '.join(docs)}." if docs else ".")
        return (
            f"{len(all_hits)} chunk(s) retrieved, {len(relevant_hits)} judged relevant"
            + (f" from {', '.join(docs)}." if docs else ".")
        )
    if tool_name == "get_policy_section":
        return f"Retrieved {result.get('document_id')} §{result.get('section')}."
    if tool_name == "lookup_employee_profile":
        return f"{result.get('full_name')} ({result.get('employment_type')}, {result.get('department_name')})."
    if tool_name == "check_pto_balance":
        if result.get("has_tracked_balance"):
            return f"Balance: {result.get('balance_hours')} {result.get('unit', 'hours')} available."
        return result.get("note") or "No tracked PTO balance."
    if tool_name == "lookup_benefits_status":
        if result.get("has_record"):
            plan = result.get("medical_plan") or "no medical plan selected"
            return f"Benefits eligible: {result.get('benefits_eligible')}; medical plan: {plan}."
        return result.get("note") or "No benefits record on file."
    if tool_name == "check_policy_compliance":
        reasons = result.get("reasons") or []
        return f"status={result.get('status')}" + (f": {reasons[0]}" if reasons else "")
    if tool_name == "draft_hr_email":
        return f"Draft prepared to {result.get('to')}: '{result.get('subject')}'. {result.get('note', '')}"
    if tool_name == "create_mock_hr_ticket":
        return result.get("message", f"status={result.get('status')}")
    # Unknown/future tool: fall back to a generic, bounded summary rather
    # than assuming a shape.
    return str(result)[:200]


def citations_from_tool_result(tool_name: str, result: dict) -> list[Citation]:
    """Extract citable evidence, if any, from a tool result. Only ever
    reads fields the corresponding MCP tool actually returns (see
    `mcp_server/schemas.py`) — never invents a citation.

    For `search_policy_documents`, only relevance-filtered hits become
    citations (`_filter_relevant_hits`) — the UI's citation panel should
    reflect evidence that plausibly supports the answer, not every raw
    top-k chunk the retriever happened to return alongside it (e.g. an
    equipment or holiday-schedule chunk that scored far below the actual
    match for an unrelated query). `get_policy_section` and
    `check_policy_compliance` are unaffected: a section fetch already
    names one exact section, and `check_policy_compliance`'s citations
    come from the tool's own small set of targeted, scenario-specific
    retrieval queries (`mcp_server.compliance.gather_citations`), not a
    single broad top-k dump.
    """
    if tool_name == "search_policy_documents":
        return [
            Citation(
                document_id=hit["document_id"],
                document_title=hit["document_title"],
                section=hit["section"],
                snippet=hit["snippet"],
                source_path=hit.get("source_path"),
                source_tool=tool_name,
            )
            for hit in _filter_relevant_hits(result.get("results", []))
        ]
    if tool_name == "get_policy_section":
        content = result.get("content", "")
        snippet = content if len(content) <= 220 else content[:220].rsplit(" ", 1)[0] + "..."
        return [
            Citation(
                document_id=result["document_id"],
                document_title=result["document_title"],
                section=result["section"],
                snippet=snippet,
                source_path=result.get("source_path"),
                source_tool=tool_name,
            )
        ]
    if tool_name == "check_policy_compliance":
        return [
            Citation(
                document_id=c["document_id"],
                document_title=c["document_title"],
                section=c["section"],
                snippet=c["snippet"],
                source_tool=tool_name,
            )
            for c in result.get("citations", [])
        ]
    return []


def escalation_from_tool_result(tool_name: str, result: dict) -> str | None:
    """A short escalation note for the trace, if this result carries one."""
    if tool_name == "check_policy_compliance" and result.get("escalation_required"):
        approvals = result.get("required_approvals") or []
        return f"Requires: {', '.join(approvals)}" if approvals else "Escalation required."
    return None


def dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Citation] = []
    for citation in citations:
        key = (citation.document_id, citation.section, citation.snippet[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def attribute_citations(answer: str, candidate_citations: list[Citation]) -> list[Citation]:
    """Narrow `candidate_citations` -- every citation accumulated from tool
    results over the whole turn (see `citations_from_tool_result`,
    collected across calls in `agent/orchestrator.py`) -- down to the ones
    the model's *final answer text* actually attributes a claim to, so
    `AgentResponse.citations` reflects evidence actually used to ground the
    answer, not merely evidence that was retrieved somewhere along the way.
    Two stages, applied in order:

    1. **Document-level**: keep only candidates whose `document_id` the
       answer mentions by exact ID (see `POLICY_ID_PATTERN`, exact set
       membership, never substring containment -- a document whose ID
       happens to prefix or resemble another's can never be confused with
       it). Falls back to the full candidate set if the answer mentions no
       candidate's ID at all (model formatting variation, or a rarer case
       where only a non-candidate/fabricated ID is mentioned) -- this can
       only ever *remove* a candidate, never add one, so a fabricated
       policy ID the model writes but never actually retrieved can never
       appear in the result.
    2. **Section-level** (`_filter_to_supporting_sections`): a model can
       legitimately retrieve several *sections of the same document* while
       researching a question (e.g. a harassment report retrieving a
       document's reporting-procedures, FAQ, and disciplinary-action
       sections together, since they're all topically close) without its
       final answer actually resting on all of them. Stage 1 alone cannot
       catch this, since every one of them shares the same document_id the
       answer legitimately mentions -- stage 2 narrows *within* each
       surviving document to the sections whose own content the answer
       actually echoes, relative to that document's best-evidenced section,
       with the same kind of conservative, per-document fallback as stage 1
       (see that function's docstring) plus an unconditional exemption for
       an exact `get_policy_section` fetch.

    Deterministic throughout: candidates are deduplicated first
    (`dedupe_citations`, then `_dedupe_by_document_and_section` to collapse
    the same section reached via two different tool calls to the single
    most useful instance), and every stage preserves first-seen relative
    order -- never re-sorted or shuffled.

    Conservative by construction at every stage: neither stage can ever
    reduce a case's citations to fewer than what genuinely has no lexical
    or ID evidence against it, and it must never silently drop every
    citation a case otherwise had simply because the model's wording didn't
    happen to repeat something verbatim.
    """
    deduped = dedupe_citations(candidate_citations)
    deduped = _dedupe_by_document_and_section(deduped)
    if not deduped:
        return deduped

    mentioned_ids = set(POLICY_ID_PATTERN.findall(answer))
    document_attributed = [c for c in deduped if c.document_id in mentioned_ids]
    document_attributed = document_attributed if document_attributed else deduped

    return _filter_to_supporting_sections(answer, document_attributed)
