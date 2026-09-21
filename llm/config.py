"""LLM provider configuration and factory.

All settings come from environment variables (loaded from `.env` via
python-dotenv, same convention as `rag.config`). Constructing a real
provider (`get_provider()`) never happens implicitly at import time —
only when explicitly called — so importing this module, or any module
that imports it, never makes a network call or requires an API key. That
is what keeps the normal offline test suite free of any Groq dependency:
tests construct `llm.fake_provider.FakeLLMProvider` directly instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from llm.base import LLMConfigError, LLMProvider

load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY") or None
    groq_model: str | None = os.getenv("GROQ_MODEL") or None
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str | None = os.getenv("GEMINI_MODEL") or None
    # Low/near-deterministic by default: appropriate for HR policy answers,
    # where consistent, literal grounding matters more than creative variety.
    temperature: float = _env_float("LLM_TEMPERATURE", 0.2)
    timeout_seconds: float = _env_float("LLM_TIMEOUT_SECONDS", 30.0)


def is_llm_configured(config: LLMConfig | None = None) -> bool:
    """Whether enough configuration is present to construct a real
    provider — used by `/health` to report `llm_configured` without
    constructing a client or making a network call."""
    config = config or LLMConfig()
    if config.provider == "groq":
        return bool(config.groq_api_key)
    if config.provider == "gemini":
        return bool(config.gemini_api_key)
    return False


def get_provider(config: LLMConfig | None = None) -> LLMProvider:
    """Construct the configured real LLM provider. Raises `LLMConfigError`
    (never a bare `KeyError`/`ValueError`) if configuration is missing or
    the provider name is unrecognized."""
    config = config or LLMConfig()
    if config.provider == "groq":
        from llm.groq_provider import GroqProvider

        return GroqProvider(
            api_key=config.groq_api_key,
            model=config.groq_model,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )
    if config.provider == "gemini":
        from llm.gemini_provider import GeminiProvider

        return GeminiProvider(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )
    raise LLMConfigError(f"Unknown LLM_PROVIDER '{config.provider}'. Supported providers: 'groq', 'gemini'.")
