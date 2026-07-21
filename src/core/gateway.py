"""The gateway: an ordered chain of (provider, model) targets tried in sequence.

This is the piece you asked to build. The logic is small on purpose - the value
is in getting the failure semantics exactly right:

  - success            -> return immediately, tagged with which target served it
  - retryable failure  -> log and try the next target
  - non-retryable      -> raise now; failing over cannot help a malformed/unauthorized request
  - every target fails  -> raise AllProvidersFailedError with the full error trail

You'll recognize the shape: it's the same "try, classify the failure, decide
retry-vs-abort" pattern you already run across your SQS-Lambda pipeline, just
with LLM providers as the unreliable downstream instead of a flaky integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.provider import Provider, ProviderError
from core.types import ChatRequest, ChatResponse

logger = logging.getLogger("llm_gateway")


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
    def __init__(self, chain: list[Target]) -> None:
        if not chain:
            raise ValueError("gateway requires at least one target in the chain")
        self._chain = chain

    @property
    def chain_labels(self) -> list[str]:
        return [t.label for t in self._chain]

    async def complete(self, request: ChatRequest) -> ChatResponse:
        errors: list[ProviderError] = []

        for index, target in enumerate(self._chain):
            try:
                response = await target.provider.complete(request, model=target.model)
            except ProviderError as error:
                errors.append(error)

                if not error.retryable:
                    # The request/config is the problem. Another provider would
                    # reject it too, so don't waste a call - surface it now.
                    logger.error("non-retryable failure on %s: %s", target.label, error)
                    raise

                logger.warning(
                    "retryable failure on %s (%s); falling over to next target",
                    target.label,
                    error,
                )
                continue

            if index > 0:
                logger.warning("request served by fallback target %s", target.label)
            return response

        raise AllProvidersFailedError(errors)