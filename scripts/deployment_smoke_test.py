"""Smoke test a deployed HR Compass AI instance (e.g. on Render).

    python -m scripts.deployment_smoke_test https://your-app.onrender.com
    python -m scripts.deployment_smoke_test https://your-app.onrender.com --test-chat

Checks (in order): `/` (chat UI), `/health`, `/api/tools`, and -- only if
`--test-chat` is passed *and* `/health` reports `llm.configured: true` on
that deployment -- a real `POST /chat` round trip. `--test-chat` is opt-in
and gated on the deployment's own reported LLM configuration (never
guessed) because a chat call may use the deployment's real Groq quota.

Uses only the Python standard library (`urllib.request`) -- deliberately
no `requests`/`httpx` dependency, so this script has zero extra install
cost for a one-off post-deploy check. Prints a clear PASS/FAIL per check
and exits non-zero if anything failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 15


def _get(url: str) -> tuple[int, dict | str]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, f"connection failed: {exc.reason}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def _post_json(url: str, payload: dict) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 0, f"connection failed: {exc.reason}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="e.g. https://your-app.onrender.com (no trailing slash needed)")
    parser.add_argument("--test-chat", action="store_true",
                         help="Also POST /chat, but only if /health reports llm.configured=true on this deployment.")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    failures = 0

    print(f"Smoke-testing {base_url} ...\n")

    print("GET / (chat UI) ...")
    status, body = _get(f"{base_url}/")
    if status == 200 and isinstance(body, str) and "HR Compass AI" in body:
        print(f"  PASS ({status})")
    else:
        print(f"  FAIL ({status}): {str(body)[:200]}")
        failures += 1

    print("GET /health ...")
    status, health = _get(f"{base_url}/health")
    if status == 200 and isinstance(health, dict) and "status" in health:
        print(f"  PASS ({status}): status={health.get('status')}, "
              f"mcp.ready={health.get('mcp', {}).get('ready')}, "
              f"rag.ready={health.get('rag', {}).get('ready')}, "
              f"llm.configured={health.get('llm', {}).get('configured')}")
        if health.get("status") != "ok":
            print("  WARNING: deployment reports degraded status -- check MCP/RAG readiness above.")
    else:
        print(f"  FAIL ({status}): {str(health)[:200]}")
        failures += 1
        health = {}

    print("GET /api/tools ...")
    status, tools = _get(f"{base_url}/api/tools")
    if status == 200 and isinstance(tools, dict) and tools.get("tools"):
        print(f"  PASS ({status}): {len(tools['tools'])} tool(s) discovered.")
    else:
        print(f"  FAIL ({status}): {str(tools)[:200]}")
        failures += 1

    if args.test_chat:
        llm_configured = isinstance(health, dict) and health.get("llm", {}).get("configured") is True
        print("POST /chat ...")
        if not llm_configured:
            print("  SKIPPED: /health reports llm.configured=false on this deployment (no GROQ_API_KEY set there).")
        else:
            status, chat = _post_json(f"{base_url}/chat", {"message": "What is Solara's paid holiday policy?"})
            if status == 200 and isinstance(chat, dict) and chat.get("answer"):
                print(f"  PASS ({status}): answer received ({len(chat['answer'])} chars), "
                      f"{len(chat.get('citations', []))} citation(s).")
            else:
                print(f"  FAIL ({status}): {str(chat)[:200]}")
                failures += 1
    else:
        print("POST /chat ... SKIPPED (pass --test-chat to include it).")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
