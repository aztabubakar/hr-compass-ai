# Instructions for Claude Code

You are building **HR Compass AI**, a Quantic AI Engineering project. Read `PROJECT_SPEC.md`, `docs/architecture.md`, and `docs/acceptance-checklist.md` before making architectural decisions.

## Priorities
1. Correctness and reproducibility.
2. Real MCP-based tool invocation; do not fake MCP calls or substitute direct function calls for required MCP behavior.
3. Grounded RAG responses with verifiable document/section citations and snippets.
4. Clear operational traces (tool name, arguments, result summary, retrieved sources, escalation/confirmation decisions) without hidden chain-of-thought.
5. Free-tier-friendly deployment and modest resource use.
6. Automated tests and evaluation.

## Preferred stack
Use Python unless a requirement clearly benefits from another choice. Prefer:
- FastAPI for API + health endpoints.
- Streamlit for a simple chat UI OR a minimal FastAPI-served frontend if that produces a cleaner single-service deployment.
- Official/current Python MCP SDK for MCP client/server.
- SentenceTransformers or another zero-cost embedding model.
- FAISS or Chroma for a small local persisted vector index.
- JSON/SQLite for synthetic structured data.
- Provider abstraction for LLMs configured by environment variables; make it straightforward to use a low-cost/free-tier provider.
- pytest for tests.
- GitHub Actions for CI.
- Render-compatible deployment configuration.

Do not blindly follow these preferences if package/runtime compatibility makes another free-tier-friendly design materially more reliable. Document deviations.

## Engineering rules
- Keep modules small and testable.
- Never commit secrets.
- Provide `.env.example`.
- Synthetic data only.
- Add type hints and useful error messages.
- Add deterministic settings/seeds where applicable.
- Persist source metadata needed for citations.
- Build graceful fallback behavior when an LLM/MCP dependency is unavailable.
- Mock write actions and require explicit confirmation before any simulated state-changing action.
- Do not expose chain-of-thought. Traces must be concise operational events only.
- Keep the repository runnable locally from documented commands.
- Update README/design docs as implementation changes.

## Development method
Work in phases. Before each phase, inspect existing files and tests. After each phase, run tests and report what changed, commands run, and remaining blockers. Avoid giant rewrites when incremental changes are sufficient.
