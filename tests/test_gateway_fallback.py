"""Fallback behavior tests. These use fake providers - no real API keys, no SDKs -
because the gateway depends only on the Provider protocol, not on any vendor.
That decoupling is the payoff of using a Protocol: the test is a plain object
with a `name` attribute and a `complete` coroutine, and it just works.
"""

from __future__ import annotations

import pytest

from core.gateway import AllProvidersFailedError, LLMGateway, Target
from core.provider import ProviderError
from core.types import ChatRequest, ChatResponse, Message, Role, Usage


class FakeProvider:
    """Structurally satisfies the Provider protocol. Configurable to succeed or
    to raise a chosen ProviderError, and records whether it was called."""

    def __init__(self, name: str, *, error: ProviderError | None = None) -> None:
        self.name = name
        self._error = error
        self.calls = 0

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ChatResponse(
            content=f"answer from {self.name}",
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=1, output_tokens=1),
            latency_ms=1.0,
        )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role=Role.user, content="hi")])


def _retryable(name: str) -> ProviderError:
    # retryable but not transient -> fails over without wasting retries
    return ProviderError("rate limited", provider=name, retryable=True, status_code=429)


def _fatal(name: str) -> ProviderError:
    return ProviderError("bad request", provider=name, retryable=False, status_code=405)


@pytest.mark.asyncio
async def test_primary_success_never_touches_fallback():
    primary = FakeProvider("primary")
    backup = FakeProvider("backup")
    gw = LLMGateway([Target(primary, "m1"), Target(backup, "m2")])

    resp = await gw.complete(_request())

    assert resp.provider == "primary"
    assert primary.calls == 1
    assert backup.calls == 0  # fallback untouched when primary works


@pytest.mark.asyncio
async def test_retryable_failure_fails_over_to_next():
    primary = FakeProvider("primary", error=_retryable("primary"))
    backup = FakeProvider("backup")
    gw = LLMGateway([Target(primary, "m1"), Target(backup, "m2")])

    resp = await gw.complete(_request())

    assert resp.provider == "backup"  # served by the fallback
    assert primary.calls == 1
    assert backup.calls == 1


@pytest.mark.asyncio
async def test_non_retryable_fails_fast_without_failover():
    primary = FakeProvider("primary", error=_fatal("primary"))
    backup = FakeProvider("backup")
    gw = LLMGateway([Target(primary, "m1"), Target(backup, "m2")])

    with pytest.raises(ProviderError):
        await gw.complete(_request())

    assert primary.calls == 1
    assert backup.calls == 0  # a fail-fast request must NOT be retried elsewhere


@pytest.mark.asyncio
async def test_all_retryable_raises_aggregate_with_full_trail():
    a = FakeProvider("a", error=_retryable("a"))
    b = FakeProvider("b", error=_retryable("b"))
    gw = LLMGateway([Target(a, "m1"), Target(b, "m2")])

    with pytest.raises(AllProvidersFailedError) as exc:
        await gw.complete(_request())

    assert len(exc.value.errors) == 2  # both failures captured for debugging
    assert a.calls == 1 and b.calls == 1


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        LLMGateway([])