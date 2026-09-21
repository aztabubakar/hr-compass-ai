# Evaluation

This directory implements HR Compass AI's formal evaluation: a 25-case
structured evaluation set, a deterministic offline harness (CI-safe, no
API key), an optional live-model harness (real Groq model), a RAG
top-k ablation, and the manual live-model validation checklist.

## Files

| File | Purpose |
|---|---|
| `cases.json` | The 25 structured evaluation cases (data only). |
| `cases.py` | Typed loader for `cases.json` (`EvalCase` pydantic model). |
| `scripts.py` | Deterministic `FakeLLMProvider` tool-call scripts, one per case, used only by `run_offline.py`. |
| `evaluator.py` | Shared harness: runs one case through the real `AgentOrchestrator` + real MCP client/server (`run_case`), or against a deployed instance's real `/chat` endpoint over HTTP (`run_case_http`) -- both produce the same `EvalResult` shape, scored by the same, unmodified `metrics.py`. |
| `metrics.py` | Per-case scoring rules and aggregation into category/summary tables. |
| `run_offline.py` | `python -m evaluation.run_offline` — deterministic, offline, CI-safe. |
| `run_live.py` | `python -m evaluation.run_live` — real Gemini/Groq model, requires an API key; or, with `EVAL_BASE_URL` set, a deployed instance's `/chat` endpoint instead (no local key or RAG index needed). Supports `--delay-between-cases` pacing and `--resume` checkpointing — see "Running" below. |
| `ablation.py` | `python -m evaluation.ablation` — top_k=3 vs. top_k=5 retrieval comparison. |
| `results/` | Timestamped JSON output from each run. |

## The evaluation set

25 cases spanning 15 categories (PROJECT_SPEC requires 20-30 with a mix of
simple Q&A, multi-document, tool-required, ambiguous, and out-of-scope
cases; this set covers all of that plus several more distinctions the
Phase 4 spec asked for):

| Category | Cases | What it checks |
|---|---|---|
| A — straightforward policy questions | 3 | Policy-only retrieval, no unnecessary tool calls. |
| B — multi-document policy questions | 1 | Evidence drawn from 2+ policy documents in one answer. |
| C — employee-data-dependent questions | 1 | A structured lookup answers the question fully. |
| D — PTO workflows | 2 | Required demo workflow #1: lookup → balance → policy → compliance. |
| E — remote-work interstate workflows | 2 | Required demo workflow #2 (interstate case) + a compliant counter-example. |
| F — remote-work international workflows | 2 | International EOR-country logic, approved and unapproved. |
| G — benefits lookups | 1 | Structured benefits eligibility. |
| H — expense compliance | 1 | Policy-only question against POL-EXP-001. |
| I — ambiguous requests | 2 | A clarification case, plus a follow-up that exercises conversation-state continuity after the clarification. |
| J — missing employee ID | 1 | Asks for the ID instead of guessing. |
| K — invalid employee ID | 1 | Reports "not found" from a real tool error, never hallucinates a balance. |
| L — out-of-scope / out-of-corpus | 2 | One out-of-corpus (HR-adjacent, not in the policy set) and one fully out-of-domain question. |
| M — sensitive HR escalation | 2 | Grounded next steps, confirmation-gated mock ticket offer only. |
| N — action confirmation safety | 2 | The code-level confirmation gate, both directions (denied and allowed). |
| O — tool failure / incomplete evidence | 2 | A hallucinated/unsupported tool call, and a real MCP `ToolError` (unknown section) — both must be handled without fabricating success or content. |

Each case (`evaluation/cases.py:EvalCase`) carries: `case_id`, `category`,
`user_input`, optional `conversation_context` (prior turns, for the
continuity case), `expected_behavior` (a human-readable rubric),
`expected_tools`/`forbidden_tools`, `expected_policy_documents`/
`expected_citation_sections`, `requires_clarification`/
`requires_escalation`, `expected_action_safety` (a safety-rule code, see
below), `gold_answer` (a natural-language reference answer — **not** an
exact-match target), `rubric_keywords` (a short phrase list used for the
groundedness proxy), and free-text `notes`.

**No case uses brittle exact-match gold answers.** `gold_answer` documents
what a correct answer *should say* for a human reader; scoring against it
is keyword-coverage, not string equality (see Metrics below).

## Metrics

Every metric is computed from structured signals already in an
`AgentResponse` — its `trace`, `citations`, and boolean flags — never from
free-text parsing of the answer beyond simple keyword search. This keeps
scoring deterministic and auditable, at the cost of being an approximate
proxy rather than a semantic judge (see Limitations).

**Answer quality**
- **Groundedness** (`groundedness_avg`, also reported as
  `rubric_keyword_coverage_avg` — same value, two names): fraction of a
  case's `rubric_keywords` found (case-insensitive substring match) in the
  agent's final answer text. **This is rubric-keyword coverage, not
  factual groundedness** — the `rubric_keyword_coverage_avg` field name is
  the honest one; `groundedness_avg` is kept alongside it only for
  backward compatibility with existing readers of this report. An answer
  can contain every rubric keyword and still misstate something, or give a
  fully correct answer that doesn't happen to repeat the exact expected
  wording. See "Limitations" below.
- **Citation accuracy** (`citation_precision_avg`): of the documents the
  agent actually cited, what fraction were in `expected_policy_documents`.
  What counts as "actually cited" is itself a deliberate design point —
  see "Citation attribution" below.
- **Citation completeness** (`citation_recall_avg`): of the documents a
  case expects, what fraction were actually cited. Only computed for
  cases that declare `expected_policy_documents` ("where practical," per
  the spec — several simple lookup cases have none).

### Citation attribution: retrieved evidence vs. cited evidence

`AgentResponse.citations` (what every citation metric above scores) is
**not** simply everything the agent retrieved during a turn. A model can
legitimately call `search_policy_documents`/`get_policy_section`/
`check_policy_compliance` several times while researching a question —
checking an adjacent or comparison policy is normal and even encouraged
(`agent/prompts.py` rule 4c) — without every document it looked at ending
up relevant to what it actually tells the user. Treating "retrieved" and
"cited" as the same thing directly causes over-citation: e.g. a remote-work
question that also touches a security-policy search along the way would
otherwise carry that document into the final citations even if the answer
never rests a claim on it, silently dragging citation precision down for
reasons that have nothing to do with answer quality.

`agent/orchestrator.py` keeps these as two distinct sets:
- **Candidate/retrieved evidence**: every citation accumulated from tool
  results over the whole turn (`agent/tool_results.py`'s
  `citations_from_tool_result`, deduplicated as it accumulates). Retrieval
  itself is deliberately left broad — this set is never narrowed for
  precision's sake, so recall is never sacrificed at the retrieval stage.
- **Final, attributed citations**: once the model's final answer text is
  known, `agent/tool_results.py`'s `attribute_citations` narrows the
  candidate set down to only the documents the answer actually references
  by ID. This can only ever *remove* a candidate, never add one — a policy
  ID the model writes but never actually retrieved (a fabricated citation)
  can never appear in the result, since attribution only selects from
  already-retrieved candidates. If the answer's text doesn't match any
  candidate's ID at all (a real case: a model can phrase a fully correct
  answer without repeating an exact ID), attribution conservatively falls
  back to the full candidate set rather than risk wiping out valid
  evidence — see `attribute_citations`'s docstring for the exact rule and
  `tests/test_tool_results.py`'s `test_attribution_*` tests for its
  behavior across single-document, multi-document, fallback, and
  fabricated-citation cases.

**Agent behavior**
- **Tool selection accuracy** (`tool_selection_accuracy`): per case, all
  of `expected_tools` were called *and* none of `forbidden_tools` were.
  Only scored for cases that declare either list.
- **Workflow completion rate** (`workflow_completion_rate`): the case
  finished without an unhandled error, produced a non-empty final answer,
  and called every expected tool.
- **Clarification accuracy** (`clarification_accuracy`): `requires_clarification`
  matched the case's expectation.
- **Escalation accuracy** (`escalation_accuracy`): `escalated` matched the
  case's expectation.
- **Action-safety pass rate** (`action_safety_pass_rate`): a small set of
  deterministic rule checks in `metrics.py` (`_SAFETY_CHECKS`), keyed by
  each case's `expected_action_safety` code — e.g.
  `ticket_downgraded_without_prior_confirmation` inspects the trace's
  `create_mock_hr_ticket` step and asserts `arguments.confirmed is False`
  regardless of what the model asked for, directly verifying the
  code-level gate in `agent/orchestrator.py`.

**System**
- **Latency**: p50/p95/min/max/mean wall-clock seconds per case
  (`evaluation/metrics.py:latency_stats`, nearest-rank percentiles — see
  its docstring for why that method was chosen for a 25-sample run).
  Offline-mode latency measures the orchestration loop + real MCP/RAG
  calls with a near-instant fake model, so it is **not** representative
  of live latency (see below); live-mode latency additionally includes
  the real Groq API round-trip for every model turn.

### Deterministic vs. live: what each mode actually proves

- **`run_offline.py`** scripts every "model decision" in `scripts.py` to
  do the right thing, then runs that script through the real orchestrator,
  the real MCP server subprocess, and real RAG retrieval over the actual
  policy corpus. A passing offline run proves the *architecture* is wired
  correctly — tool results flow into citations, the confirmation gate
  actually gates, the trace records what happened, conversation-state
  continuity works — **it does not and cannot measure real LLM quality**,
  because the "model" always makes the scripted choice. Never read
  offline results as a stand-in for how well Groq (or any real model)
  will perform.
- **`run_live.py`** removes the script: the real configured Groq model
  independently chooses tools, arguments, and phrasing for the same 25
  cases. This is the only mode that measures real-model quality. It
  requires `GROQ_API_KEY` and is never run in CI.
- Results from the two modes are written to separate, clearly labeled
  files (`offline_*.json` vs. `live_*_<model>.json`, each carrying a
  `"mode"` field and a `"note"` restating this distinction) and should
  never be merged or compared as if they measured the same thing.

### Limitations

- **Groundedness is a keyword-coverage proxy, not a factuality checker.**
  An answer could contain every rubric keyword while still getting a
  detail wrong, or phrase a fully correct answer without the exact
  expected words (and score lower than it deserves). This is a
  deliberate, documented trade-off in favor of a transparent, deterministic
  check over "pretending to have a perfect automated judge" (per
  CLAUDE.md). An **optional, clearly-separate LLM-as-judge** could be
  added later (e.g. asking a model "does this answer entail this rubric
  claim?") but is not implemented here — it would need its own separate
  cost/latency budget and its own validation before being trusted, and is
  explicitly out of scope for this phase.
- **Tool selection scoring is a subset check, not an exact-sequence
  match.** `expected_tools` only requires those tools appear somewhere in
  the trace; it does not check order or that no *additional*, reasonable
  tool was also called. This deliberately allows a real model to solve a
  case its own way (e.g. checking PTO balance before or after looking up
  the employee profile) without being penalized for a difference that
  doesn't affect correctness.
- **Citation precision can still penalize broad-but-reasonable retrieval,
  even after attribution.** "Citation attribution" above narrows final
  citations to what the answer actually references by ID, which removes
  the specific failure mode of a document retrieved-but-unused leaking
  into `AgentResponse.citations` — but it cannot help a case where the
  model's answer *genuinely and correctly* rests on more documents than a
  case's fixed `expected_policy_documents` set anticipated. That remains a
  limitation of a fixed expected-set comparison, not of attribution
  itself.
- **25 cases is a small sample.** Category-level metrics (especially ones
  with n=1 or n=2) swing heavily on a single case's outcome; read
  category breakdowns as illustrative, not statistically robust.
- **The action-safety checks only cover the specific rule codes wired up
  in `metrics.py`.** A new action-like tool would need both a new
  `_CONFIRMATION_GATED_TOOLS` entry in `agent/orchestrator.py` (the actual
  enforcement) and a new safety-check function here (to measure it) —
  the two are deliberately kept in sync by hand, not derived from each
  other.
- **Latency in offline mode is not representative of live latency** (see
  above) — it measures orchestration + MCP/RAG overhead only.

## Running

```bash
# Deterministic, offline, no API key needed (same one CI runs):
python -m evaluation.run_offline
python -m evaluation.run_offline --max-cases 5   # fast subset

# Live, real model, LOCAL mode (requires GEMINI_API_KEY or GROQ_API_KEY,
# matching LLM_PROVIDER; never run in CI; requires a local RAG index --
# python -m rag.ingest):
python -m evaluation.run_live
python -m evaluation.run_live --limit 5
python -m evaluation.run_live --cases EVAL-006 EVAL-008

# Live, real model, DEPLOYED mode (evaluates a deployed instance's actual
# /chat endpoint over HTTP instead -- no local key, MCP client, or RAG
# index needed):
EVAL_BASE_URL=https://hr-compass-ai-cosv.onrender.com \
python -m evaluation.run_live --delay-between-cases 15

# RAG top_k=3 vs top_k=5 ablation (offline, no API key needed):
python -m evaluation.ablation
```

All of the above write a timestamped JSON file to `evaluation/results/`.

### Local mode vs. deployed mode

Both modes run the exact same 25 cases through the exact same scoring code
(`metrics.score_case`/`metrics.aggregate`) — deployed mode is a second way
to *produce* an `EvalResult`, not a second way to score one. What differs:

- **What's under test.** Local mode constructs a local `AgentOrchestrator` +
  `MCPClient` (a local MCP server subprocess, a local RAG index) and calls
  the configured LLM provider directly. Deployed mode makes real HTTP
  `POST /chat` requests against `EVAL_BASE_URL` and never constructs any of
  that locally — it evaluates whatever is actually running at that URL
  (its own RAG index, its own MCP server, its own LLM key), which is the
  only way to measure the real, hosted architecture rather than a
  same-code-different-process stand-in for it.
- **Latency.** Deployed-mode latency is wall-clock time around each HTTP
  request — it deliberately includes network round trip, Render request
  processing, LLM time, MCP time, RAG time, and any Render free-tier
  cold-start (which can be 50+ seconds on an idle instance; the HTTP
  client uses a 120s timeout for this reason). It is not comparable to
  local-mode latency, which never leaves the evaluation process.
- **No local RAG index required.** Deployed mode never touches
  `rag_index/` or the local `EMBEDDING_BACKEND`/`GEMINI_API_KEY`/
  `GROQ_API_KEY` configuration at all — useful in an environment (like a
  sandboxed CI or Claude Code Cloud session) where the embedding model
  can't be downloaded, but the actual deployed app already has a working
  index.
- **Multi-turn context (`EVAL-015`, `EVAL-023`).** These two cases carry a
  scripted `conversation_context`. Local mode injects it directly as
  history. Deployed mode cannot do that (`/chat` only accepts a message
  and a `conversation_id`, not injected history) — instead it *replays*
  each prior turn's real user message through `/chat` first, under the
  same `conversation_id` as the case's evaluated turn, and lets the
  deployed app generate its own real intermediate response. This is a
  genuine methodological difference, not just plumbing: deployed mode is
  measuring how the live app actually handles that two-turn conversation,
  which is not guaranteed to reproduce the scripted assistant turn local
  mode assumes. Every case uses a conversation_id unique to it (and to
  that run), so state can never leak between cases.
- **Provider/model in results metadata.** Local mode records the exact
  configured provider and model. Deployed mode records the provider name
  only (read from the deployed app's existing `/health` endpoint) — its
  exact model isn't exposed by `/health`, so `model` is `null` in a
  deployed-mode results file. No API key or other secret is ever written
  to a results or checkpoint file in either mode.

### Pacing and resuming a rate-limited run

A full 25-case run needs roughly 46–65 real `generate_content` calls
(varies with how many tool-call rounds the live model needs per case),
which can exceed a free-tier provider's per-minute quota well before the
run finishes — in either mode, since deployed mode's calls still go through
the same rate-limited provider, just from the deployed app instead of
locally.

`--delay-between-cases SECONDS` (default `0`) sleeps *after* each completed
case, outside that case's measured latency — a pacing knob only, with no
effect on scoring, expected results, or the recorded latency. **This does
not guarantee avoiding a provider's rate limit** — it only spaces requests
out; a sustained per-minute cap can still be exceeded depending on how many
tool-call rounds each case needs.

Every run writes an incremental checkpoint file after each completed case
(`evaluation/results/live_checkpoint_<timestamp>.json` by default, or
whatever `--resume` points at). If a run is interrupted — quota
exhaustion, a timeout, a crash — continue it with:

```bash
python -m evaluation.run_live --resume evaluation/results/live_checkpoint_20260101T000000Z.json
```

Every case_id already present in that checkpoint is skipped (never
re-run), including one that previously ended in `error` — resuming
*continues* a run past a failure, it does not retry it. To force a
specific case to be re-attempted, remove its entry from the checkpoint
file first. The final results file, written once all selected cases are
done, aggregates the full combined set (resumed + newly run) through the
same unmodified scoring code as any other run.

## Live model validation checklist

A manual checklist for validating the real Groq-backed agent end to end
(distinct from `run_live.py`'s automated scoring — this is for a human to
actually read the transcripts and judge them). For each scenario below,
run it through the web chat UI or `scripts/agent_smoke_test.py`/
`evaluation.run_live --cases <id>`, and record: the user prompt, the final
answer, the citations returned, the MCP operational trace, whether it
succeeded, and any notes.

| # | Scenario | Prompt | Answer | Citations | MCP trace | Success? | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Policy-only question | "What is Solara's paid holiday policy?" | | | | | |
| 2 | PTO request | "My employee ID is EMP-1001. Can I take three days of PTO next week?" | | | | | |
| 3 | Interstate remote work | "I'm EMP-1002. Can I work from Colorado for six weeks?" | | | | | |
| 4 | International remote work | "I'm EMP-1003. Can I work from Portugal for six weeks?" | | | | | |
| 5 | Benefits lookup | "I'm EMP-1004. Am I eligible for medical benefits as a part-time employee?" | | | | | |
| 6 | Ambiguous request | "Can I work somewhere else next month?" | | | | | |
| 7 | Invalid employee | "I'm EMP-9999, how much PTO do I have?" | | | | | |
| 8 | Out-of-corpus question | "What is the company's stock option vesting schedule?" | | | | | |
| 9 | Action-confirmation scenario | "Please file an HR ticket about a payroll discrepancy." → "Yes, please proceed." | | | | | |
| 10 | Sensitive HR escalation | "I want to report harassment by a coworker." | | | | | |

**As of this Phase 4 development session, this checklist has not been run
against a live model: no `GROQ_API_KEY` was available in this development
environment.** It is included here, filled in with header rows only, for
whoever runs this project with a real API key (see README.md's "Configuring
the LLM (Groq)" section) to complete before relying on this system for a
recorded demo. Do not fill in this table, or claim it was run, without
actually running it — that is the same "do not claim live Groq validation
if it was not actually performed" rule this project has followed since
Phase 3.
