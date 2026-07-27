"""finish_reason normalization tests.

The point of normalizing: a caller checks `== FinishReason.length` to detect a
truncated answer, without caring that OpenAI calls it 'length' while Anthropic
and Bedrock call it 'max_tokens'. This is the exact signal that would have made
the Gemini thinking-budget truncation obvious instead of a mystery.
"""

from __future__ import annotations

import pytest

from core.types import FinishReason, normalize_finish_reason


@pytest.mark.parametrize(
    "raw,expected",
    [
        # natural completion, across providers
        ("stop", FinishReason.stop),            # OpenAI
        ("end_turn", FinishReason.stop),        # Anthropic / Bedrock
        ("stop_sequence", FinishReason.stop),
        # TRUNCATION - the case that matters most
        ("length", FinishReason.length),        # OpenAI
        ("max_tokens", FinishReason.length),    # Anthropic / Bedrock
        # safety
        ("content_filter", FinishReason.content_filter),   # OpenAI
        ("content_filtered", FinishReason.content_filter),  # Bedrock
        ("guardrail_intervened", FinishReason.content_filter),
        ("refusal", FinishReason.content_filter),           # Anthropic
        # tools
        ("tool_calls", FinishReason.tool_call),  # OpenAI
        ("tool_use", FinishReason.tool_call),    # Anthropic / Bedrock
    ],
)
def test_provider_reasons_map_to_normalized(raw, expected):
    assert normalize_finish_reason(raw) is expected


def test_case_insensitive():
    assert normalize_finish_reason("MAX_TOKENS") is FinishReason.length


def test_unknown_and_none_become_other():
    assert normalize_finish_reason("some_new_reason") is FinishReason.other
    assert normalize_finish_reason(None) is FinishReason.other


def test_finish_reason_serializes_as_its_string_value():
    # str-enum, so it JSON-serializes to "length" in responses and SSE events.
    assert FinishReason.length == "length"