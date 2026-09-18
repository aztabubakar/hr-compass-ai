"""Central, single-source-of-truth filesystem locations for the project.

Every module that needs to find the policy corpus, the mock data
directory, or the persisted vector index should import from here rather
than re-deriving paths with ad-hoc `Path(__file__).parent` logic, so the
project can be relocated or restructured without hunting down scattered
path assumptions.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "policies"
MOCK_DATA_DIR = PROJECT_ROOT / "mock_data"

# Default location for the persisted vector index (a build artifact; not
# committed to git — see .gitignore). Overridable via RagConfig for tests.
DEFAULT_RAG_INDEX_DIR = PROJECT_ROOT / "rag_index"

# Where confirmed mock HR tickets created via the MCP `create_mock_hr_ticket`
# tool are persisted. Kept separate from the read-only committed
# `mock_data/hr_tickets.json` fixture so running the server never mutates
# committed synthetic data. Overridable so tests can isolate ticket state
# per test run instead of writing into the real project directory.
RUNTIME_TICKETS_PATH = Path(
    os.getenv("HR_COMPASS_RUNTIME_TICKETS_PATH", str(MOCK_DATA_DIR / "runtime_tickets.json"))
)
