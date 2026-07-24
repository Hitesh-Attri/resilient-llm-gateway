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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from core.provider import Provider, ProviderError
from core.retry import RetryPolicy, full_jitter_delay
from core.types import ChatRequest, ChatResponse
from core.log import get_logger

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
    