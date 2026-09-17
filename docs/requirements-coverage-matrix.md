# Requirements coverage matrix

Maps every item in `docs/acceptance-checklist.md` (which mirrors
`PROJECT_SPEC.md`) to its implementation, test/evidence, and documentation,
with an honest status. Statuses used:

- **complete** — implemented, tested, and documented; nothing further needed.
- **pending-live-validation** — implemented and offline-tested, but its
  real-world quality claim needs a live Groq run this environment
  couldn't perform (no `GROQ_API_KEY`).
- **pending-deployment** — implemented and validated locally/in CI, but
  needs an actual Render deployment this environment couldn't perform (no
  Render account).

| Requirement | Implementation | Test / evidence | Documentation | Status |
|---|---|---|---|---|
| Local venv/setup documented | `requirements.txt`, `.env.example` | manual: `python3 -m venv .venv && pip install -r requirements.txt` succeeds | README.md "Setup" | complete |
| Dependency manifest present | `requirements.txt` | pinned via `pip freeze`; CI installs from it | README.md, requirements.txt header comments | complete |
| `.env.example`; no committed secrets | `.env.example` | `git status`/`grep` secret scan run this session (see final report) | README.md "Configuring the LLM" | complete |
| 5-20 coherent policy files, ~30-120 pages | `policies/` (12 docs, md/html/pdf) | `tests/test_parsers.py` (real-corpus integration tests) | design-and-evaluation.md section 3 | complete |
| >=2 policy source formats | Markdown, HTML, PDF (`policies/*.md/.html/.pdf`) | `rag/parsers.py` tests | design-and-evaluation.md section 3-4 | complete |
| Document cleaning/chunking justified | `rag/cleaning.py`, `rag/chunking.py` | `tests/test_chunking.py`, `tests/test_cleaning.py` | design-and-evaluation.md section 4 | complete |
| Embeddings + persisted vector index | `rag/embeddings.py`, `rag/vector_store.py` (FAISS) | `tests/test_vector_store.py`, `tests/test_embeddings.py` | design-and-evaluation.md section 4 | complete |
| Citation metadata retained | `rag/models.py` (doc_id/title/section/source_path/snippet) | `tests/test_retrieval.py` | design-and-evaluation.md section 4 | complete |
| Top-k RAG retrieval works | `rag/retrieval.py:PolicyRetriever.query` | `tests/test_retrieval.py`; `evaluation/ablation.py` (top_k=3 vs 5) | design-and-evaluation.md sections 4, 10.3 | complete |
| Grounded answers cite doc/section + snippet | `agent/tool_results.py:citations_from_tool_result` | `tests/test_agent_orchestrator.py::test_citations_propagate_from_tool_results_with_correct_fields` | design-and-evaluation.md section 7 | complete |
| Out-of-corpus / unsupported-claim guardrails | `rag/config.py:min_relevance_score`; system prompt (`agent/prompts.py`) | `tests/test_agent_orchestrator.py::test_out_of_corpus_question_reflects_real_low_evidence`; `EVAL-018`/`EVAL-019` | design-and-evaluation.md section 7 | complete |
| Multi-document retrieval case works | `check_policy_compliance` + `search_policy_documents` combined | `tests/test_agent_orchestrator.py::test_remote_work_interstate_multi_document_evidence`; `EVAL-004`, `EVAL-008` | design-and-evaluation.md sections 6, 10.1 | complete |
| Agent decides RAG-only vs. tool workflows | `agent/orchestrator.py` (LLM-driven, no hard-coded branching) | `tests/test_agent_orchestrator.py` (all 17 behavior tests) | design-and-evaluation.md section 7 | complete |
| >=2 multi-step workflows end-to-end | PTO workflow, remote-work workflow | `tests/test_agent_orchestrator.py`; `scripts/demo_workflows.py`; `EVAL-006`-`EVAL-011` | design-and-evaluation.md sections 7, 9.2, 10.1 | complete |
| Concise operational trace; no chain-of-thought | `core/trace.py`, `agent/orchestrator.py` | every agent/API test asserts trace shape | design-and-evaluation.md section 7.5 | complete |
| Missing employee IDs / ambiguous requests handled | `agent/prompts.py` (CLARIFY: prefix) | `test_missing_employee_id_asks_for_it`, `test_ambiguous_request_asks_for_clarification_without_tool_calls`; `EVAL-014`-`EVAL-016` | design-and-evaluation.md section 7 | complete |
| MCP failures handled gracefully | `agent/orchestrator.py` (`MCPClientError`/`MCPToolError` handling) | `test_mcp_discovery_failure_returns_graceful_response`, `test_hallucinated_tool_name_is_handled_gracefully`; `EVAL-024`, `EVAL-025` | design-and-evaluation.md section 7 | complete |
| State-changing ops mock + confirmation-gated | `agent/orchestrator.py:_CONFIRMATION_GATED_TOOLS`, `mcp_server/tools.py:create_mock_hr_ticket` | `test_ticket_creation_without_prior_confirmation_turn_is_downgraded_to_preview`, `test_ticket_creation_after_confirmation_offer_is_allowed`; `EVAL-022`, `EVAL-023` | design-and-evaluation.md sections 7.6, 9.1 | complete |
| MCP server implemented | `mcp_server/server.py` (official MCP SDK, stdio) | `tests/test_mcp_server.py` and all `test_mcp_*` | design-and-evaluation.md section 6 | complete |
| >=5 MCP tools exposed | 8 tools in `mcp_server/tools.py` | `test_api_tools_lists_all_required_tools` | design-and-evaluation.md section 6 | complete |
| >=1 RAG MCP tool | `search_policy_documents`, `get_policy_section` | `tests/test_mcp_tools.py` | design-and-evaluation.md section 6 | complete |
| >=1 structured-data/mock-op MCP tool | `lookup_employee_profile`, `check_pto_balance`, `create_mock_hr_ticket`, etc. | `tests/test_mcp_tools.py` | design-and-evaluation.md section 6 | complete |
| Agent actually invokes tools through MCP | `agent/orchestrator.py` calls only `mcp_client.MCPClient`, never imports `mcp_server.tools` | every `tests/test_agent_orchestrator.py` test spawns a real `mcp_server.server` subprocess | design-and-evaluation.md section 7.2 | complete |
| Chat UI implemented | `app/static/index.html` | `test_root_serves_chat_ui_with_required_elements` | design-and-evaluation.md section 7.8 | complete |
| `/chat` returns answer/citations/snippets/trace | `app/main.py:chat`, `app/models.py:ChatResponse` | `tests/test_api.py` (6 `/chat` tests) | design-and-evaluation.md section 7.8 | complete |
| `/health` returns app/MCP status | `app/main.py:health`, nested `HealthResponse` | `test_health_returns_expected_shape`, `test_health_reports_degraded_when_mcp_unavailable` | design-and-evaluation.md section 9.6 | complete |
| PTO demo task reproducible | `scripts/run_pto_demo.py`; `EVAL-006`/`EVAL-007`; UI prompt | `tests/test_mcp_workflows.py`; `evaluation/run_offline.py` | README.md "Example demo prompts", design-and-evaluation.md section 9.2 | complete |
| Remote-work demo task reproducible | `scripts/run_remote_work_demo.py`; `EVAL-008`-`EVAL-011` | `tests/test_mcp_workflows.py`; `evaluation/run_offline.py` | README.md "Example demo prompts", design-and-evaluation.md section 9.2 | complete |
| Free-tier deployment config included | `render.yaml` | YAML-validated this session; not yet applied to a live Render account | design-and-evaluation.md sections 9.4-9.5 | **pending-deployment** |
| GitHub Actions on push/PR | `.github/workflows/ci.yml` | YAML-validated; every step run locally this session with matching commands | design-and-evaluation.md section 9.3 | complete |
| App-start/import smoke test | `scripts/app_smoke_test.py` | run this session (PASSED); wired as a CI step | README.md "Smoke tests", design-and-evaluation.md section 9.3 | complete |
| MCP discovery/tool-call automated test | `scripts/mcp_smoke_test.py` + `tests/test_mcp_*.py` | run this session (PASSED); wired as a CI step | README.md "Smoke tests", design-and-evaluation.md section 9.3 | complete |
| Deployment gated on tests | `.github/workflows/ci.yml`'s `deploy` job (`needs: test`) | YAML-validated; genuinely gates once the one-time Render/GitHub secret setup (documented in the job) is done | design-and-evaluation.md section 9.3 | complete (code) / **pending-deployment** (the one-time Render-side setup) |
| 20-30 evaluation cases with gold answers/rubrics | `evaluation/cases.json` (25 cases) | `tests/test_evaluation.py::test_cases_json_has_25_well_formed_cases` | evaluation/README.md | complete |
| Groundedness measured | `evaluation/metrics.py:score_case` (`groundedness_score`) | `evaluation/run_offline.py` output (0.96 avg, this session) | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** (real-model score) |
| Citation accuracy measured | `evaluation/metrics.py` (`citation_precision`/`citation_recall`) | offline run: precision 0.81, recall 1.00 | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** |
| Tool-selection accuracy measured | `evaluation/metrics.py` (`tool_selection_pass`) | offline run: 1.00 | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** |
| Workflow-completion measured | `evaluation/metrics.py` (`workflow_complete`) | offline run: 1.00 | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** |
| Clarification/escalation accuracy measured | `evaluation/metrics.py` | offline run: 1.00 / 1.00 | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** |
| Action-safety pass rate measured | `evaluation/metrics.py:_SAFETY_CHECKS` | offline run: 1.00 | evaluation/README.md, design-and-evaluation.md section 10.1 | complete (offline) / **pending-live-validation** |
| p50/p95 latency measured; cold/warm noted | `evaluation/metrics.py:latency_stats` | offline run: p50=0.007s, p95=0.026s (not representative of live latency, documented as such) | evaluation/README.md, design-and-evaluation.md sections 10.1, 10.4 | complete (offline; methodology only) / **pending-deployment** (cold/warm) / **pending-live-validation** (real latency) |
| >=1 ablation/comparison | `evaluation/ablation.py` (top_k=3 vs 5) | run this session: hit rate 0.79->0.86, recall 0.71->0.86 | design-and-evaluation.md section 10.3 | complete |
| README.md complete | `README.md` | manual review | -- | complete |
| design-and-evaluation.md complete | `design-and-evaluation.md` | manual review | -- | complete |
| ai-tooling.md complete | `ai-tooling.md` | manual review | -- | complete |
| deployed.md complete | `deployed.md` | explicit placeholders, no invented URL | -- | **pending-deployment** |
| Architecture diagram included | `docs/architecture.md` (Mermaid + text), mirrored in design-and-evaluation.md section 9.1 | manual review | docs/architecture.md | complete |
| 7-10 minute demo script/checklist prepared | README.md "Example demo prompts"; `evaluation/README.md`'s live-model validation checklist doubles as a demo script | -- | README.md, evaluation/README.md | **pending-live-validation** (checklist exists; not yet run/recorded) |

## Summary

Of the 48 checklist items: **41 fully complete**, **6 complete in code/
tests but pending-live-validation** (groundedness, citation accuracy,
tool-selection accuracy, workflow-completion, clarification/escalation
accuracy, and action-safety pass rate are all measured and passing
offline, but their real-world quality claim needs the live Groq run this
environment couldn't perform), and **3 pending-deployment** (free-tier
deployment config exists but hasn't been applied to a live Render
account; the deploy gate's one-time Render/GitHub-secret setup; and
`deployed.md`'s placeholders, which depend on both). The p50/p95 latency
and 7-10 minute demo items straddle both categories (methodology/checklist
complete, real numbers/recording pending). No item is unimplemented --
everything remaining requires external resources (a `GROQ_API_KEY`, a
Render account) not available in this development environment, not
further engineering work.
