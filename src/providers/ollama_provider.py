"""Ollama - local models via its OpenAI-compatible server. No API key (the SDK
still needs a non-empty string), base_url points at the local daemon."""

from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"
    default_base_url = "http://localhost:11434/v1"
    supports_stream_usage = False