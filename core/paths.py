"""Central, single-source-of-truth filesystem locations for the project.

Every module that needs to find the policy corpus, the mock data
directory, or the persisted vector index should import from here rather
than re-deriving paths with ad-hoc `Path(__file__).parent` logic, so the
project can be relocated or restructured without hunting down scattered
path assumptions.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = PROJECT_ROOT / "policies"
MOCK_DATA_DIR = PROJECT_ROOT / "mock_data"

# Default location for the persisted vector index (a build artifact; not
# committed to git — see .gitignore). Overridable via RagConfig for tests.
DEFAULT_RAG_INDEX_DIR = PROJECT_ROOT / "rag_index"
