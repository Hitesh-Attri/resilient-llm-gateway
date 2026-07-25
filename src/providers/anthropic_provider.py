"""Anthropic (direct API) adapter.

Anthropic's shape:
  - `system` is a top-level kwarg (matches our model - no translation needed).
  - messages are [{"role", "content"}] with roles user/assistant only.
  - response text lives in resp.content, a list of blocks; we join the text ones.
  - usage is resp.usage.input_tokens / output_tokens.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from core.provider import ProviderError, is_transient, should_failover
from core.types import ChatRequest, ChatResponse, StreamChunk, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        from anthropic import AsyncAnthropic  # lazy: only import if this adapter is used

        self._client = AsyncAnthropic(api_key=api_key)

    def _map_error(self, e: Exception) -> ProviderError:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        if isinstance(e, RateLimitError):
            return ProviderError(
                str(e), provider=self.name, retryable=True,
                status_code=429, transient=is_transient(429, str(e)),
            )
        if isinstance(e, APIConnectionError):
            return ProviderError(str(e), provider=self.name, retryable=True, transient=True)
        if isinstance(e, APIStatusError):
            # Exhausted credits arrive as a 400, so status alone isn't enough -
            # should_failover/is_transient also inspect the message.
            return ProviderError(
                str(e), provider=self.name,
                retryable=should_failover(e.status_code, str(e)),
                status_code=e.status_code,
                transient=is_transient(e.status_code, str(e)),
            )
        raise e

    def _build_kwargs(self, request: ChatRequest, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        }
        if request.system:
            kwargs["system"] = request.system
        return kwargs

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from anthropic import APIError

        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(**self._build_kwargs(request, model))
        except APIError as e:
            raise self._map_error(e) from e
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResponse(
            content=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            latency_ms=latency_ms,
        )

    async def stream(self, request: ChatRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        from anthropic import APIError

        start = time.perf_counter()
        usage = Usage()
        try:
            # messages.stream() gives an async text_stream plus a final message
            # carrying usage - cleaner than parsing raw SSE events by hand.
            async with self._client.messages.stream(**self._build_kwargs(request, model)) as s:
                async for text in s.text_stream:
                    yield StreamChunk(delta=text)
                final = await s.get_final_message()
                usage = Usage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                )
        except APIError as e:
            raise self._map_error(e) from e

        yield StreamChunk(
            finished=True, provider=self.name, model=model, usage=usage,
            latency_ms=(time.perf_counter() - start) * 1000,
        )