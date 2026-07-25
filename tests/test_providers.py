"""Provider wiring tests.

These stay SDK-free: importing a provider module doesn't import its vendor SDK
(that's lazy, inside __init__), and class attributes are readable without
constructing anything. So we can assert identity/capabilities and the config's
missing-key guards without any API keys or packages installed.
"""

from __future__ import annotations

import pytest

from core.config import Settings, _construct_provider
from providers.gemini_provider import GeminiProvider
from providers.groq_provider import GroqProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_compatible import OpenAICompatibleProvider
from providers.openai_provider import OpenAIProvider


def test_each_provider_has_its_own_identity():
    assert OpenAIProvider.name == "openai"
    assert GroqProvider.name == "groq"
    assert GeminiProvider.name == "gemini"
    assert OllamaProvider.name == "ollama"


def test_base_urls_are_distinct_and_correct():
    assert OpenAIProvider.default_base_url is None  # SDK default
    assert GroqProvider.default_base_url.endswith("groq.com/openai/v1")
    assert "generativelanguage.googleapis.com" in GeminiProvider.default_base_url
    assert OllamaProvider.default_base_url.endswith("11434/v1")


def test_stream_usage_capability_differs_per_provider():
    # The whole reason capability lives per-class: these genuinely differ.
    assert OpenAIProvider.supports_stream_usage is True
    assert GroqProvider.supports_stream_usage is True
    assert GeminiProvider.supports_stream_usage is False   # compat layer rejects it
    assert OllamaProvider.supports_stream_usage is False


def test_all_share_the_wire_protocol_base():
    for cls in (OpenAIProvider, GroqProvider, GeminiProvider, OllamaProvider):
        assert issubclass(cls, OpenAICompatibleProvider)


@pytest.mark.parametrize(
    "name,key_env",
    [
        ("openai", "OPENAI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_missing_key_fails_fast_with_helpful_message(name, key_env):
    # The key check runs before any client is constructed, so this needs no SDK.
    with pytest.raises(ValueError, match=key_env):
        _construct_provider(name, Settings(_env_file=None))


def test_unknown_provider_name_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        _construct_provider("nonesuch", Settings(_env_file=None))