"""reasoning_effort tests.

The resolution logic is a pure function, so the interesting cases (request
overrides default, default applies when unset, non-reasoning providers never get
it) are tested with zero SDK and zero network.
"""

from __future__ import annotations

from core.types import ReasoningEffort
from providers.gemini_provider import GeminiProvider
from providers.groq_provider import GroqProvider
from providers.openai_compatible import resolve_reasoning_kwargs
from providers.openai_provider import OpenAIProvider


def test_non_reasoning_provider_never_sends_it():
    # Even if a caller sets it, a provider that doesn't support it must drop it -
    # sending reasoning_effort to Llama on Groq would error.
    assert resolve_reasoning_kwargs(False, ReasoningEffort.high, ReasoningEffort.low) == {}


def test_request_value_overrides_default():
    out = resolve_reasoning_kwargs(True, ReasoningEffort.high, ReasoningEffort.low)
    assert out == {"reasoning_effort": "high"}


def test_default_applies_when_request_unset():
    out = resolve_reasoning_kwargs(True, None, ReasoningEffort.low)
    assert out == {"reasoning_effort": "low"}


def test_nothing_sent_when_no_request_and_no_default():
    assert resolve_reasoning_kwargs(True, None, None) == {}


def test_capability_flags_per_provider():
    # Only reasoning-capable providers opt in.
    assert GeminiProvider.supports_reasoning_effort is True
    assert GroqProvider.supports_reasoning_effort is False
    # OpenAI stays off for now: reasoning_effort there is model-dependent, and
    # sending it to a non-reasoning OpenAI model would error.
    assert OpenAIProvider.supports_reasoning_effort is False


def test_gemini_defaults_to_low():
    assert GeminiProvider.default_reasoning_effort is ReasoningEffort.low