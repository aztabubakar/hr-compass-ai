# Deployment

**Status: not yet deployed.** The fields below are placeholders/instructions
to fill in once the project is actually deployed to Render (see
`render.yaml` and `design-and-evaluation.md` section 9 for the deployment
design and section 10 for the RAG index build strategy). No URL has been
invented here -- everything below is either `TBD` or an instruction for
what to record once it exists.

- **Public application URL:** TBD (fill in after deploying; e.g.
  `https://hr-compass-ai.onrender.com`, whatever Render assigns this
  service).
- **Health endpoint URL:** TBD (the same base URL + `/health`).
- **Deployment provider:** Render (free-tier "web" service; see
  `render.yaml`). Railway or another free-tier host would also work with
  minor changes to the start command/health check path.
- **Deployment date:** TBD (record the date of the first successful
  production deploy).
- **Branch / commit deployed:** TBD (record the git branch and commit SHA
  Render built from -- visible in the Render dashboard's deploy log, and
  worth cross-checking against `git log -1` at deploy time).
- **Environment variables required in production** (names only -- see
  `.env.example` for descriptions; set actual values in the Render
  dashboard, never commit them):
  - `LLM_PROVIDER` (set to `groq`)
  - `GROQ_API_KEY` (secret; required for `/chat` to actually answer)
  - `GROQ_MODEL` (optional)
  - `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS` (optional; have sane defaults)
  - `EMBEDDING_BACKEND` (recommend `fastembed` in production; `render.yaml`
    sets this)
  - `MCP_TRANSPORT` (`stdio`; the only transport this project implements)
- **Cold-start behavior:** TBD once observed. Expected, based on the
  deployment design: Render's free tier spins the service down after a
  period of inactivity and takes some number of seconds to cold-start the
  next request. The RAG index is built at Render *build* time (not on
  first request -- see `design-and-evaluation.md` section 10), so a cold
  start should only pay for process startup (FastAPI app boot + spawning
  the MCP server subprocess), not index construction or a fastembed model
  download. Record actual observed cold-start vs. warm-request latency
  here once deployed (see `evaluation/README.md`'s latency methodology,
  and `scripts/deployment_smoke_test.py` for a repeatable way to measure
  it against the live URL).
- **Known deployment limitations:**
  - Conversation state (`app/state.py`'s `ConversationStore`) is
    in-memory and per-process; Render's free tier runs a single instance,
    so this is fine for a demo but would not survive a process restart or
    a multi-instance deployment without a real session store.
  - The MCP server runs as a subprocess of the FastAPI process (stdio
    transport); if that subprocess dies, `/health`'s `mcp.ready` will
    report `false` until the whole service restarts (there is no
    subprocess auto-restart logic in this phase).
  - No live Groq validation has been performed as of this Phase 4
    development session (no `GROQ_API_KEY` was available in the
    development environment) -- see `evaluation/README.md`'s live model
    validation checklist, which must be completed against the real
    deployment before treating this system as demo-ready.
