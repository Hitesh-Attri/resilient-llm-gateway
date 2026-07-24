"""Anthropic (direct API) adapter.

Anthropic's shape:
  - `system` is a top-level kwarg (matches our model - no translation needed).
  - messages are [{"role", "content"}] with roles user/assistant only.
  - response text lives in resp.content, a list of blocks; we join the text ones.
  - usage is resp.usage.input_tokens / output_tokens.
"""

from __future__ import annotations

import time
from typing import Any

from core.provider import ProviderError, should_failover
from core.types import ChatRequest, ChatResponse, Usage


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        from anthropic import AsyncAnthropic  # lazy: only import if this adapter is used

        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from anthropic import APIConnectionError, APIStatusError, RateLimitError

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        }
        if request.system:
            kwargs["system"] = request.system

        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(**kwargs)
        except RateLimitError as e:
            raise ProviderError(str(e), provider=self.name, retryable=True, status_code=429) from e
        except APIConnectionError as e:
            raise ProviderError(str(e), provider=self.name, retryable=True) from e
        except APIStatusError as e:
            # Note: exhausted credits arrive as a 400, so status alone is not
            # enough - should_failover also inspects the message.
            raise ProviderError(
                str(e),
                provider=self.name,
                retryable=should_failover(e.status_code, str(e)),
                status_code=e.status_code,
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResponse(
            content=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            latency_ms=latency_ms,
        )