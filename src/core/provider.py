"""The contract every provider adapter must satisfy, plus the one error type
the gateway reasons about.

Two design decisions live here:

1. `Provider` is a `typing.Protocol`, not an ABC. Adapters conform structurally -
   they don't inherit from anything. This keeps them decoupled and trivially
   mockable in tests (a plain object with the right shape passes).

2. Every provider maps its vendor-specific exceptions onto ONE `ProviderError`
   carrying a `retryable` flag. That flag is the entire basis for the gateway's
   fail-over-vs-fail-fast decision. Adapters own the classification because only
   they understand their vendor's error taxonomy.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import ChatRequest, ChatResponse


class ProviderError(Exception):
    """Normalized provider failure.

    The question `retryable` answers is NOT "was this transient?" but:
    "could a DIFFERENT provider succeed with this same request?"

    retryable=True  -> the provider can't serve us right now: rate limit, 5xx,
                       timeout, connection, exhausted credits, bad/expired key.
                       Fail OVER - another provider is likely fine.
    retryable=False -> the REQUEST itself is invalid (malformed body, unknown
                       model, context-length exceeded). Every provider would
                       reject it, so fail FAST rather than burn the whole chain.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


# Statuses that mean "this exact request is malformed and EVERY provider will
# reject it identically" - so walking the chain only wastes calls. This is the
# short list. Everything NOT here fails over.
#
# Why the list is short: ChatRequest is Pydantic-validated at the API boundary,
# so structurally broken requests are already rejected with a 422 before any
# provider is called. By the time an adapter runs, the request is well-formed,
# and nearly every remaining 4xx is a TARGET problem (retired/unknown model,
# no credits, bad key, prompt too long FOR THIS model) - exactly what the next
# rung exists to survive. The model ID belongs to the rung, not the request, so
# "model not found" says nothing about the next provider's model.
_FAIL_FAST_STATUSES = frozenset({
    405,  # method not allowed - wrong verb, wrong everywhere
    406,  # not acceptable
    414,  # URI too long
    415,  # unsupported media type
})


def should_failover(status_code: int, message: str) -> bool:
    """Should this HTTP failure fail OVER to the next provider (True) or fail
    FAST and surface now (False)?

    Default is FAIL OVER. For a multi-provider gateway, availability is the whole
    point, and most 4xx errors that reach here are specific to the current target
    (its model ID, its credits, its key), not intrinsic to the request. We fail
    fast only for the narrow set of statuses that would break identically on
    every provider.

    Shared by the OpenAI and Anthropic adapters (HTTP/status based). Bedrock
    classifies on botocore error codes instead.

    Tradeoff: a genuinely broken request now costs one call per rung (N x latency
    and cost) before erroring. That's acceptable because the failure is bounded
    and the aggregate error trail is preserved in AllProvidersFailedError.
    """
    return status_code not in _FAIL_FAST_STATUSES


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        ...