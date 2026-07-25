"""Streaming tests - the important ones prove the commit boundary:

  - before the first token: retry the same target, then fail over (recoverable)
  - after the first token:  no failover possible, a mid-stream error just surfaces

Fake streaming providers make this testable with no SDKs and no real delay (the
gateway's sleep/rand are injected).
"""

from __future__ import annotations

import pytest

from core.gateway import AllProvidersFailedError, LLMGateway, Target
from core.provider import ProviderError
from core.retry import RetryPolicy
from core.types import ChatRequest, Message, Role, StreamChunk, Usage


class FakeStreamProvider:
    def __init__(
        self,
        name: str,
        *,
        deltas=("a", "b", "c"),
        open_error: ProviderError | None = None,
        open_fail_times: int = 0,
        mid_error_at: int | None = None,
    ) -> None:
        self.name = name
        self.deltas = deltas
        self.open_error = open_error
        self.open_fail_times = open_fail_times
        self.mid_error_at = mid_error_at
        self.open_attempts = 0  # increments each time the generator body starts

    async def stream(self, request, *, model):
        self.open_attempts += 1
        # An open failure is raised BEFORE any yield (like a failed API call on
        # the first __anext__), so the gateway can still recover.
        if self.open_error is not None and self.open_attempts <= self.open_fail_times:
            raise self.open_error
        for i, d in enumerate(self.deltas):
            if self.mid_error_at is not None and i == self.mid_error_at:
                raise ProviderError(
                    "mid-stream drop", provider=self.name, retryable=True,
                    status_code=503, transient=True,
                )
            yield StreamChunk(delta=d)
        yield StreamChunk(
            finished=True, provider=self.name, model=model,
            usage=Usage(input_tokens=1, output_tokens=len(self.deltas)), latency_ms=1.0,
        )


def _req():
    return ChatRequest(messages=[Message(role=Role.user, content="hi")])


def _transient_open(name):
    return ProviderError("503 opening stream", provider=name, retryable=True,
                         status_code=503, transient=True)


def _permanent_open(name):
    # retryable but not transient (e.g. no credits): fail over, don't retry
    return ProviderError("no credits", provider=name, retryable=True,
                         status_code=429, transient=False)


def _gateway(chain, **kw):
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    return LLMGateway(chain, sleep=fake_sleep, rand=lambda: 1.0, **kw), sleeps


async def _collect(gw):
    """Drive gw.stream, splitting deltas / final / error."""
    deltas: list[str] = []
    final: StreamChunk | None = None
    error: Exception | None = None
    try:
        async for c in gw.stream(_req()):
            (deltas.append(c.delta) if not c.finished else None)
            if c.finished:
                final = c
    except (ProviderError, AllProvidersFailedError) as e:
        error = e
    return deltas, final, error


@pytest.mark.asyncio
async def test_stream_success_yields_deltas_then_final():
    p = FakeStreamProvider("groq", deltas=("Hel", "lo", "!"))
    gw, _ = _gateway([Target(p, "m")])

    deltas, final, error = await _collect(gw)

    assert error is None
    assert deltas == ["Hel", "lo", "!"]
    assert final is not None and final.provider == "groq"
    assert final.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_stream_retries_open_then_succeeds():
    p = FakeStreamProvider("groq", open_error=_transient_open("groq"), open_fail_times=2)
    gw, sleeps = _gateway([Target(p, "m")], retry=RetryPolicy(max_attempts=3, base_delay=0.5))

    deltas, final, error = await _collect(gw)

    assert error is None
    assert deltas == ["a", "b", "c"]
    assert p.open_attempts == 3       # retried the open twice on the same provider
    assert sleeps == [0.5, 1.0]       # backoff before each retry


@pytest.mark.asyncio
async def test_stream_fails_over_before_first_token():
    primary = FakeStreamProvider("groq", open_error=_permanent_open("groq"), open_fail_times=99)
    backup = FakeStreamProvider("gemini", deltas=("x", "y"))
    gw, sleeps = _gateway([Target(primary, "m1"), Target(backup, "m2")],
                          retry=RetryPolicy(max_attempts=3))

    deltas, final, error = await _collect(gw)

    assert error is None
    assert deltas == ["x", "y"]       # served entirely by the fallback
    assert final.provider == "gemini"
    assert primary.open_attempts == 1  # not transient -> no retries wasted
    assert sleeps == []


@pytest.mark.asyncio
async def test_no_failover_after_first_token_commits():
    # Primary streams two deltas, then dies mid-stream. We are committed - the
    # backup must NOT be used, and the client sees the partial output + an error.
    primary = FakeStreamProvider("groq", deltas=("a", "b", "c"), mid_error_at=2)
    backup = FakeStreamProvider("gemini", deltas=("should", "not", "appear"))
    gw, _ = _gateway([Target(primary, "m1"), Target(backup, "m2")])

    deltas, final, error = await _collect(gw)

    assert deltas == ["a", "b"]        # partial output already delivered
    assert final is None              # never reached the finished chunk
    assert isinstance(error, ProviderError)  # surfaced, not failed over
    assert backup.open_attempts == 0  # fallback untouched after commit


@pytest.mark.asyncio
async def test_stream_all_targets_fail_to_open():
    a = FakeStreamProvider("a", open_error=_transient_open("a"), open_fail_times=99)
    b = FakeStreamProvider("b", open_error=_transient_open("b"), open_fail_times=99)
    gw, _ = _gateway([Target(a, "m1"), Target(b, "m2")], retry=RetryPolicy(max_attempts=2))

    deltas, final, error = await _collect(gw)

    assert deltas == []
    assert isinstance(error, AllProvidersFailedError)
    assert a.open_attempts == 2 and b.open_attempts == 2  # each retried to its max