"""OpenAI (direct API). The base class already IS the OpenAI wire format, so this
subclass only pins identity and capabilities."""

from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    default_base_url = None  # the SDK's default (api.openai.com)
    supports_stream_usage = True