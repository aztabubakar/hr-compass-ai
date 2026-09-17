"""Provider-agnostic LLM abstraction for the HR Compass AI agent.

`agent/` depends only on `llm.base.LLMProvider`. `llm.config.get_provider()`
constructs the configured real provider (Groq by default); tests use
`llm.fake_provider.FakeLLMProvider` instead and never touch `llm.config`
or make a network call.
"""
