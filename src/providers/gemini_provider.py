"""Gemini via Google's OpenAI-compatibility endpoint.

This is the provider with real divergences, which is why it earns its own class:
  - its compat layer rejects `stream_options`, so usage-in-stream is off
  - it is a reasoning model whose thinking tokens share the max_tokens budget, so
    it opts into reasoning_effort (mapped to Gemini's thinking_level by Google's
    compat layer). Defaulting to `low` keeps thinking from eating the whole
    budget and truncating the answer - the fix for the mid-word cutoffs we saw.

The default is overridable per instance from config (GEMINI_REASONING_EFFORT) and
per request (ChatRequest.reasoning_effort).
"""

from __future__ import annotations

from core.types import ReasoningEffort
from providers.openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    supports_stream_usage = False
    supports_reasoning_effort = True
    default_reasoning_effort = ReasoningEffort.low