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

from .types import ChatRequest, ChatResponse


class ProviderError(Exception):
    """Normalized provider failure.

    retryable=True  -> transient (429, 5xx, timeout, connection). Fail OVER to the
                       next target; a different provider may succeed.
    retryable=False -> the request or config is the problem (400 bad request,
                       401/403 auth). Failing over won't help, so fail FAST and
                       surface it.
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


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        ...