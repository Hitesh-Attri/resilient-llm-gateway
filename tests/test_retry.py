"""Retry + backoff tests.

The gateway takes injectable `sleep` and `rand`, so these run instantly (no real
delay) and deterministically (no real randomness). Sleep durations are captured
so we can assert on the backoff schedule without waiting for it.
"""

from __future__ import annotations

import pytest

from core.gateway import AllProvidersFailedError, LLMGateway, Target
from core.provider import ProviderError
from core.retry import RetryPolicy, full_jitter_delay
from core.types import ChatRequest, ChatResponse, Message, Role, Usage


# ---- pure backoff function ------------------------------------------------

def test_full_jitter_grows_exponentially_with_rand_one():
    policy = RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=100.0)
    ceilings = [full_jitter_delay(i, policy, rand=lambda: 1.0) for i in range(4)]
    assert ceilings == [0.5, 1.0, 2.0, 4.0]  # base * 2**i


def test_full_jitter_is_capped_by_max_delay():
    policy = RetryPolicy(max_attempts=10, base_delay=1.0, max_delay=8.0)
    # 2**5 = 32 but the cap is 8
    assert full_jitter_delay(5, policy, rand=lambda: 1.0) == 8.0


def test_full_jitter_scales_with_rand():
    policy = RetryPolicy(base_delay=4.0, max_delay=100.0)
    assert full_jitter_delay(0, policy, rand=lambda: 0.0) == 0.0   # uniform lower bound
    assert full_jitter_delay(0, policy, rand=lambda: 0.5) == 2.0   # half the ceiling


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


# ---- gateway retry behavior ----------------------------------------------

class FlakyProvider:
    """Fails with `error` for the first `fail_times` calls, then succeeds."""

    def __init__(self, name: str, *, error: ProviderError, fail_times: int) -> None:
        self.name = name
        self._error = error
        self._fail_times = fail_times
        self.calls = 0

    async def complete(self, request, *, model):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return ChatResponse(
            content=f"ok from {self.name}", model=model, provider=self.name,
            usage=Usage(input_tokens=1, output_tokens=1), latency_ms=1.0,
        )


def _req():
    return ChatRequest(messages=[Message(role=Role.user, content="hi")])


def _transient(name):
    return ProviderError("rate limited", provider=name, retryable=True,
                         status_code=429, transient=True)


def _permanent_failover(name):
    # retryable but NOT transient: e.g. no credits. Must fail over without retry.
    return ProviderError("no credits", provider=name, retryable=True,
                         status_code=429, transient=False)


def _gateway(chain, **kw):
    captured: list[float] = []

    async def fake_sleep(d):
        captured.append(d)

    gw = LLMGateway(chain, sleep=fake_sleep, rand=lambda: 1.0, **kw)
    return gw, captured


@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_succeeds():
    p = FlakyProvider("groq", error=_transient("groq"), fail_times=2)
    gw, sleeps = _gateway([Target(p, "m")], retry=RetryPolicy(max_attempts=3, base_delay=0.5))

    resp = await gw.complete(_req())

    assert resp.provider == "groq"
    assert p.calls == 3               # 2 failures + 1 success, same provider
    assert sleeps == [0.5, 1.0]       # backoff before each retry (rand=1.0)


@pytest.mark.asyncio
async def test_transient_exhausted_then_falls_over():
    primary = FlakyProvider("groq", error=_transient("groq"), fail_times=99)
    backup = FlakyProvider("gemini", error=_transient("gemini"), fail_times=0)
    gw, sleeps = _gateway(
        [Target(primary, "m1"), Target(backup, "m2")],
        retry=RetryPolicy(max_attempts=3),
    )

    resp = await gw.complete(_req())

    assert resp.provider == "gemini"  # fell over after exhausting the primary
    assert primary.calls == 3         # tried the max, no more
    assert len(sleeps) == 2           # 2 backoffs on the primary, none needed on backup


@pytest.mark.asyncio
async def test_non_transient_failover_does_not_retry():
    primary = FlakyProvider("groq", error=_permanent_failover("groq"), fail_times=99)
    backup = FlakyProvider("gemini", error=_transient("gemini"), fail_times=0)
    gw, sleeps = _gateway(
        [Target(primary, "m1"), Target(backup, "m2")],
        retry=RetryPolicy(max_attempts=3),
    )

    resp = await gw.complete(_req())

    assert resp.provider == "gemini"
    assert primary.calls == 1         # no retries wasted on a no-credits error
    assert sleeps == []               # never backed off


@pytest.mark.asyncio
async def test_all_targets_exhausted_raises_aggregate():
    a = FlakyProvider("a", error=_transient("a"), fail_times=99)
    b = FlakyProvider("b", error=_transient("b"), fail_times=99)
    gw, _ = _gateway([Target(a, "m1"), Target(b, "m2")], retry=RetryPolicy(max_attempts=2))

    with pytest.raises(AllProvidersFailedError) as exc:
        await gw.complete(_req())

    assert len(exc.value.errors) == 2
    assert a.calls == 2 and b.calls == 2  # each target retried to its max