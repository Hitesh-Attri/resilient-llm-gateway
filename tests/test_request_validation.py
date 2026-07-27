"""ChatRequest validation tests - these double as executable documentation of
the request contract (defaults and bounds)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.types import ChatRequest, Message, Role


def _msgs():
    return [Message(role=Role.user, content="hi")]


def test_max_tokens_defaults_to_4096():
    # Generous enough not to truncate; safe across every provider.
    assert ChatRequest(messages=_msgs()).max_tokens == 4096


def test_max_tokens_lower_bound():
    with pytest.raises(ValidationError):
        ChatRequest(messages=_msgs(), max_tokens=0)


def test_max_tokens_upper_bound():
    with pytest.raises(ValidationError):
        ChatRequest(messages=_msgs(), max_tokens=100_000)  # above the 32768 cap


def test_max_tokens_within_bounds_ok():
    assert ChatRequest(messages=_msgs(), max_tokens=16_000).max_tokens == 16_000


def test_at_least_one_message_required():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_temperature_bounds():
    with pytest.raises(ValidationError):
        ChatRequest(messages=_msgs(), temperature=3.0)  # above 2.0