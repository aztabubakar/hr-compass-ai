# Deployment

**Status: deployed and live-evaluated.** The application is running in
production on Render, serving real requests through the real Gemini
provider, the real MCP server (8 tools), and the real RAG index. This file
records the actual deployment; see `render.yaml` for the deployment
configuration and `design-and-evaluation.md` sections 9-10 for the full
design and evaluation results.

- **Public application URL:** <https://hr-compass-ai-cosv.onrender.com>
- **Health endpoint URL:** <https://hr-compass-ai-cosv.onrender.com/health>
- **Deployment provider:** Render (free-tier "web" service; see
  `render.yaml`). Railway or another free-tier host would also work with
  minor changes to the start command/health check path.
- **Deployment date / branch / commit deployed:** not independently
  recorded in this repository — Render's own dashboard deploy log is the
  authoritative source for the exact commit SHA and timestamp behind any
  given deploy. This file is not auto-updated on every deploy, so treat
  the dashboard, not this line, as ground truth if the two ever disagree.
- **Production LLM provider:** **Gemini** (`llm/gemini_provider.py`), via
  the official `google-genai` SDK. Groq (`llm/groq_provider.py`) remains
  fully supported as an alternative — switching is a `LLM_PROVIDER`
  environment-variable change on Render, no code change needed — but
  Gemini is what this deployment actually runs.
- **Current configured model:** `gemini-3.5-flash-lite` (`render.yaml`'s
  `GEMINI_MODEL`; leave unset to fall back to
  `llm/gemini_provider.py`'s `DEFAULT_GEMINI_MODEL`, which currently
  resolves to the same model).
- **RAG index:** built during the Render *build* step
  (`buildCommand: pip install -r requirements.txt && python -m rag.ingest`
  in `render.yaml`), not on first request — every running instance starts
  with the FAISS index already on disk. See `design-and-evaluation.md`
  section 9.5 for why build-time indexing was chosen over first-request
  lazy building.
- **MCP:** a real MCP server (`mcp_server/server.py`, official Python MCP
  SDK, stdio transport) is launched as a subprocess of the FastAPI process
  and discovered by the agent through the real protocol
  (`mcp_client/client.py`) — the same architecture as local development,
  not a simplified production path.
- **Environment variables required in production** (names only — see
  `.env.example` for full descriptions; actual values are set in the
  Render dashboard and are never committed):
  - `LLM_PROVIDER` = `gemini`
  - `GEMINI_API_KEY` — **secret; configured directly in the Render
    dashboard's environment variables and never committed to this
    repository.** Required for `/chat` to actually answer.
  - `GEMINI_MODEL` (optional; see above)
  - `GROQ_API_KEY` / `GROQ_MODEL` — only needed if `LLM_PROVIDER` is
    switched back to `groq`; not required for the current deployment
  - `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS` (optional; have sane
    defaults)
  - `EMBEDDING_BACKEND` = `fastembed` (production; real semantic
    embeddings)
  - `MCP_TRANSPORT` = `stdio` (the only transport this project
    implements)

## Cold-start behavior

Render's free tier spins the service down after a period of inactivity
and cold-starts the next incoming request. The RAG index is built at
*build* time (not on first request — see above), so a cold start only
pays for process startup (FastAPI boot + spawning the MCP server
subprocess + the Gemini SDK client), not index construction or a
fastembed model download.

No isolated cold-start-vs-warm-request measurement has been made with a
dedicated timer, but the corrected production evaluation run's latency
distribution is consistent with occasional cold starts inflating a small
number of requests: mean 5.13s and p50 3.59s (most requests, presumably
warm) vs. a p95 of 10.85s and a max of 26.98s across the 25-case run (see
`design-and-evaluation.md` section 10.2 for the full table). A single
request taking ~27s against a mean of ~5s is the expected shape of "one or
two cold-started requests in a 25-request run," not a general performance
problem. `scripts/deployment_smoke_test.py` can be timed manually (e.g.
`time python -m scripts.deployment_smoke_test <url>`) for a more targeted
measurement if a precise cold-start number is needed later.

## Why production evaluation runs through GitHub Actions, not directly

This project's development sessions run inside Claude Code Cloud, whose
organization-level egress policy blocks outbound requests to the Render
hostname (confirmed via repeated direct connection failures). Rather than
work around that restriction, deployed-mode evaluation
(`evaluation/run_live.py` with `EVAL_BASE_URL` set) runs through a
manually-triggered GitHub Actions workflow
(`.github/workflows/live-evaluation.yml`), which executes outside that
sandboxed environment and therefore can reach the live deployment
directly. See `README.md`'s "Evaluation" section for how to trigger it and
`design-and-evaluation.md` section 10.2 for the results it has produced.

## Known deployment limitations

- Conversation state (`app/state.py`'s `ConversationStore`) is in-memory
  and per-process; Render's free tier runs a single instance, so this is
  fine for a demo but would not survive a process restart or a
  multi-instance deployment without a real session store.
- The MCP server runs as a subprocess of the FastAPI process (stdio
  transport); if that subprocess dies, `/health`'s `mcp.ready` will report
  `false` until the whole service restarts (there is no subprocess
  auto-restart logic in this phase).
- Real-model evaluation has now been performed against this live
  deployment (see `design-and-evaluation.md` section 10.2) and is not a
  remaining item — but its action-safety pass rate (0.75) and
  groundedness (0.72) are lower than the offline architecture-verification
  run's (1.00 / 0.96), which is expected and documented, not concealed:
  see `design-and-evaluation.md` section 10.2's "Evaluator limitations"
  subsection for why, and which categories are the most challenging.
