"""Automated FastAPI app-start + endpoint smoke test.

    python -m scripts.app_smoke_test

Builds a hermetic RAG index (hashing backend, no network access needed),
then actually starts the FastAPI app (entering its `lifespan()` -- the
same startup path Render/uvicorn use in production, including connecting
the real MCP server subprocess) via Starlette's `TestClient` as a context
manager, and exercises `/health`, `/api/tools`, and `/` (the chat UI).
Never calls a real LLM, so it needs no `GROQ_API_KEY` and is safe for CI.

This is PROJECT_SPEC's required "app-start test" -- a dedicated,
individually-named CI step that proves the application actually boots and
serves traffic, distinct from the unit-level `tests/test_api.py` (which
also does this per-test, but isn't a single standalone "did the app
start" check a CI log can point to).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from core.paths import POLICIES_DIR
from rag.config import RagConfig
from rag.ingest import run_ingest


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hr_compass_app_smoke_") as tmp_dir:
        index_dir = Path(tmp_dir)
        print(f"Building a hermetic RAG index (hashing backend) at {index_dir} ...")
        run_ingest(RagConfig(policies_dir=POLICIES_DIR, index_dir=index_dir, embedding_backend="hashing"))

        os.environ["EMBEDDING_BACKEND"] = "hashing"
        os.environ["RAG_INDEX_DIR"] = str(index_dir)

        # Imported only after the env vars above are set, since app.main
        # (via rag.config's module-level RagConfig() default) reads some of
        # this at import time.
        from app.main import app

        print("Starting the app (entering lifespan -- connects the real MCP server subprocess) ...")
        with TestClient(app, raise_server_exceptions=False) as client:
            print("Checking GET /health ...")
            health = client.get("/health")
            if health.status_code != 200:
                print(f"FAIL: /health returned {health.status_code}: {health.text}")
                return 1
            data = health.json()
            if not data["mcp"]["ready"]:
                print(f"FAIL: /health reports mcp.ready=false: {data}")
                return 1
            if not data["rag"]["ready"]:
                print(f"FAIL: /health reports rag.ready=false: {data}")
                return 1
            print(f"OK: /health -> status={data['status']}, mcp.tools_discovered={data['mcp']['tools_discovered']}, "
                  f"llm.configured={data['llm']['configured']}")

            print("Checking GET /api/tools ...")
            tools = client.get("/api/tools")
            if tools.status_code != 200 or not tools.json()["tools"]:
                print(f"FAIL: /api/tools returned {tools.status_code}: {tools.text}")
                return 1
            print(f"OK: /api/tools -> {len(tools.json()['tools'])} tool(s).")

            print("Checking GET / (chat UI) ...")
            root = client.get("/")
            if root.status_code != 200 or "HR Compass AI" not in root.text:
                print(f"FAIL: / did not serve the expected chat UI (status={root.status_code}).")
                return 1
            print("OK: / serves the chat UI.")

    print("\nFastAPI app-start smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
