"""OpenAI (direct API) adapter.

OpenAI's shape differs from ours in two ways the adapter must absorb:
  - There is no top-level system field: the system prompt is a message with
    role="system", prepended to the list. This is exactly why we kept `system`
    separate in ChatRequest - so each adapter can place it where its vendor wants.
  - usage fields are prompt_tokens / completion_tokens (not input/output).

Provider-quirk note: some newer OpenAI models rename `max_tokens` to
`max_completion_tokens` and reject `temperature`. The adapter is precisely where
you'd branch on model name to handle that - the gateway and callers never see it.
"""

from __future__ import annotations

import time

from core.provider import ProviderError
from core.types import ChatRequest, ChatResponse, Usage


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        from openai import AsyncOpenAI  # lazy import

        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from openai import APIConnectionError, APIStatusError, RateLimitError

        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role.value, "content": m.content} for m in request.messages)

        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except RateLimitError as e:
            raise ProviderError(str(e), provider=self.name, retryable=True, status_code=429) from e
        except APIConnectionError as e:
            raise ProviderError(str(e), provider=self.name, retryable=True) from e
        except APIStatusError as e:
            raise ProviderError(
                str(e),
                provider=self.name,
                retryable=e.status_code >= 500,
                status_code=e.status_code,
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000

        choice = resp.choices[0]
        usage = resp.usage
        return ChatResponse(
            content=choice.message.content or "",
            model=model,
            provider=self.name,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            latency_ms=latency_ms,
        )