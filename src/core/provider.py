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
    """Normalized provider failure carrying TWO orthogonal decisions.

    They answer different questions, and conflating them was an earlier bug:

    retryable -> "could a DIFFERENT provider succeed?" Drives the outer FALLBACK
                 loop (walk to the next rung). True for almost everything, since
                 the model/credits/key belong to the rung, not the request.

    transient -> "might the SAME provider succeed if I wait and retry?" Drives the
                 inner RETRY loop (backoff on this rung before moving on). True
                 only for genuinely temporary conditions: rate limits, 5xx,
                 timeouts, connection errors.

    Invariant: transient implies retryable (worth retrying here => worth failing
    over after). The classification helpers below guarantee it.

    Examples:
      429 rate limit      -> transient=True,  retryable=True   (retry, then fail over)
      429 insufficient_quota / 400 no credits
                          -> transient=False, retryable=True   (fail over now, no retry)
      401 bad key         -> transient=False, retryable=True   (fail over now)
      404 model retired   -> transient=False, retryable=True   (fail over now)
      405 method not allowed
                          -> transient=False, retryable=False  (fail fast)
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retryable: bool,
        status_code: int | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code
        self.transient = transient


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


# Substrings that mark a 429 as "no more money", not "slow down". Retrying these
# never helps, so the inner retry loop must skip them and fail over immediately.
#
# Note: this is the same fragile message-sniffing we deleted from should_failover
# last round - but it is SAFE here in a way it was not there. On the failover
# path a wrong guess changed the OUTCOME (chain stopped early). Here it only
# affects EFFICIENCY: if a marker is missed, we waste a couple of backoff sleeps
# and then fail over correctly anyway. Same technique, very different blast
# radius - fragility is acceptable on an efficiency-only path.
_NON_TRANSIENT_MARKERS = ("insufficient_quota", "credit balance", "quota", "billing")


def is_transient(status_code: int, message: str) -> bool:
    """Should the SAME provider be retried after a short backoff?

    True only for genuinely temporary conditions. 429 is the tricky one: a real
    rate limit is transient (wait and it clears), but an exhausted quota wearing
    a 429 is not - so we look at the message for that single case.
    """
    if status_code >= 500:
        return True
    if status_code in (408, 429):
        return not any(m in message.lower() for m in _NON_TRANSIENT_MARKERS)
    return False


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        ...