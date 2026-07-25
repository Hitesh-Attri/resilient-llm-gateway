"""Gemini via Google's OpenAI-compatibility endpoint.

This is the provider with real divergences, which is exactly why it earns its own
class:
  - its compat layer rejects `stream_options`, so usage-in-stream is off
  - it is a reasoning model whose thinking tokens share the max_tokens budget,
    so `_extra_create_kwargs` is the seam where reasoning_effort/thinking_level
    control will go (deferred to the reasoning-effort slice)
"""

from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    supports_stream_usage = False

    # def _extra_create_kwargs(self, request):
    #     return {"reasoning_effort": self._reasoning_effort}