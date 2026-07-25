"""Groq - OpenAI-compatible, fast inference for open-weight models. Supports
usage-in-stream; no other quirks today."""

from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    supports_stream_usage = True