"""The gateway: an ordered chain of (provider, model) targets, with two loops.

  INNER (retry):    on a TRANSIENT failure (rate limit, 5xx, timeout), retry the
                    SAME target a few times with exponential backoff + jitter.
  OUTER (fallback): when a target is exhausted or fails with a non-transient but
                    retryable error, fall OVER to the next target.

Outcomes:
  - success               -> return, tagged with which target served it
  - transient failure     -> back off and retry the same target
  - retryable (not transient) -> fall over to the next target immediately
  - fail-fast failure     -> raise now (a broken request helps nowhere)
  - every target fails    -> AllProvidersFailedError with the full error trail

Same shape as the retry/fallback logic in your SQS-Lambda pipeline: retry the
flaky downstream a bounded number of times, then route around it.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from core.log import get_logger
from core.provider import Provider, ProviderError
from core.retry import RetryPolicy, full_jitter_delay
from core.types import ChatRequest, ChatResponse, StreamChunk

logger = get_logger(__name__)


@dataclass(frozen=True)
class Target:
    """One rung of the fallback ladder: a provider plus the model to ask it for."""

    provider: Provider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider.name}:{self.model}"


class AllProvidersFailedError(Exception):
    def __init__(self, errors: list[ProviderError]) -> None:
        self.errors = errors
        trail = " | ".join(f"{e.provider}: {e}" for e in errors)
        super().__init__(f"all fallback targets failed -> {trail}")


class LLMGateway:
    def __init__(
        self,
        chain: list[Target],
        *,
        retry: RetryPolicy | None = None,
        # Injectable so tests run with no real delay and deterministic jitter.
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        if not chain:
            raise ValueError("gateway requires at least one target in the chain")
        self._chain = chain
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rand = rand

    @property
    def chain_labels(self) -> list[str]:
        return [t.label for t in self._chain]

    async def complete(self, request: ChatRequest) -> ChatResponse:
        errors: list[ProviderError] = []

        for index, target in enumerate(self._chain):
            try:
                response = await self._attempt_target(target, request)
            except ProviderError as error:
                errors.append(error)

                if not error.retryable:
                    # The request is broken in a way no provider can serve.
                    logger.error("non-retryable failure on %s: %s", target.label, error)
                    raise

                logger.warning(
                    "target %s exhausted (%s); falling over to next target",
                    target.label,
                    error,
                )
                continue

            if index > 0:
                logger.warning("request served by fallback target %s", target.label)
            return response

        raise AllProvidersFailedError(errors)

    async def _attempt_target(self, target: Target, request: ChatRequest) -> ChatResponse:
        """The inner retry loop: try one target up to max_attempts times, backing
        off between transient failures. Non-transient errors (and the final
        attempt) propagate straight to the outer fallback loop."""
        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                return await target.provider.complete(request, model=target.model)
            except ProviderError as error:
                is_last = attempt == self._retry.max_attempts
                if not error.transient or is_last:
                    raise  # let the outer loop decide fail-over vs fail-fast

                delay = full_jitter_delay(attempt - 1, self._retry, self._rand)
                logger.warning(
                    "transient failure on %s (attempt %d/%d): %s; retrying in %.2fs",
                    target.label,
                    attempt,
                    self._retry.max_attempts,
                    error,
                    delay,
                )
                await self._sleep(delay)

        # Unreachable: the loop either returns or raises on the last attempt.
        raise AssertionError("retry loop exited without returning or raising")

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Streaming counterpart to complete().

        The critical difference: retry and fallback can ONLY happen before the
        first token reaches the caller. Once we yield a delta, the client has
        partial output and we are committed to that target - a mid-stream failure
        can no longer fail over, it can only surface as an error.

        So the machinery below concentrates all the retry/fallback logic into
        _open_stream (which gets the first chunk in hand), and after that just
        relays. This is the streaming analogue of "buffer until you can commit."
        """
        errors: list[ProviderError] = []

        for index, target in enumerate(self._chain):
            try:
                first, remaining = await self._open_stream(target, request)
            except ProviderError as error:
                errors.append(error)
                if not error.retryable:
                    logger.error("non-retryable stream failure on %s: %s", target.label, error)
                    raise
                logger.warning(
                    "stream open failed on %s (%s); falling over to next target",
                    target.label,
                    error,
                )
                continue

            # Committed to this target: the client is about to see a token.
            if index > 0:
                logger.warning("stream served by fallback target %s", target.label)
            yield first
            async for chunk in remaining:  # a failure here can no longer fail over
                yield chunk
            return

        raise AllProvidersFailedError(errors)

    async def _open_stream(
        self, target: Target, request: ChatRequest
    ) -> tuple[StreamChunk, AsyncIterator[StreamChunk]]:
        """Open a stream and pull its first chunk, retrying transient open-time
        failures on the SAME target with backoff. Returns (first_chunk, rest).

        Everything that can go wrong AND be recovered from must happen here,
        while nothing has been emitted to the caller yet."""
        for attempt in range(1, self._retry.max_attempts + 1):
            agen = target.provider.stream(request, model=target.model)
            try:
                first = await agen.__anext__()
                return first, agen
            except StopAsyncIteration:
                await agen.aclose()
                raise ProviderError(
                    "provider produced an empty stream",
                    provider=target.provider.name,
                    retryable=True,
                    transient=False,
                ) from None
            except ProviderError as error:
                await agen.aclose()
                is_last = attempt == self._retry.max_attempts
                if not error.transient or is_last:
                    raise
                delay = full_jitter_delay(attempt - 1, self._retry, self._rand)
                logger.warning(
                    "transient stream-open failure on %s (attempt %d/%d): %s; retrying in %.2fs",
                    target.label,
                    attempt,
                    self._retry.max_attempts,
                    error,
                    delay,
                )
                await self._sleep(delay)

        raise AssertionError("stream retry loop exited without returning or raising")