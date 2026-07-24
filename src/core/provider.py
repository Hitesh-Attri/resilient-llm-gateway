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


# Substrings that mark a 4xx as an ACCOUNT problem rather than a request problem.
# Providers signal "you're out of money" with a 400, which is indistinguishable
# from a malformed request by status code alone - so we look at the message.
# Message sniffing is fragile, which is why it is confined to this one place.
_ACCOUNT_FAILURE_MARKERS = (
    "credit balance",
    "billing",
    "quota",
    "insufficient_quota",
    "payment",
    "exceeded your current quota",
)


def should_failover(status_code: int, message: str) -> bool:
    """Decide whether an HTTP failure should fail OVER to the next provider.

    Shared by the OpenAI and Anthropic adapters (both are HTTP/status based).
    Bedrock classifies on botocore error codes instead.
    """
    if status_code >= 500:
        return True
    if status_code in (408, 429):  # timeout, rate limited
        return True
    if status_code in (401, 403):
        # Bad or expired key for THIS provider. Another provider may well work,
        # so prefer availability - but this must be alarmed on, not swallowed.
        return True
    if status_code == 400 and any(m in message.lower() for m in _ACCOUNT_FAILURE_MARKERS):
        return True
    # Genuine bad request: malformed body, unknown model, context too long.
    return False


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        ...