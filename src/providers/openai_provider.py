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
from collections.abc import AsyncIterator
from typing import Any

from core.provider import ProviderError, is_transient, should_failover
from core.types import ChatRequest, ChatResponse, StreamChunk, Usage


class OpenAIProvider:
    """Serves OpenAI *and* any OpenAI-compatible endpoint (Groq, Gemini's compat
    layer, OpenRouter, Cerebras, Ollama, vLLM) by pointing `base_url` elsewhere.

    A large slice of the ecosystem cloned OpenAI's wire format, so one adapter
    plus a URL covers many vendors. This is the payoff of the adapter pattern:
    four new providers for zero new adapter code.

    `name` is overridable so logs and ChatResponse.provider say "groq", not
    "openai" - otherwise you can't tell who actually served a request.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        name: str = "openai",
        stream_usage: bool = False,
    ) -> None:
        from openai import AsyncOpenAI  # lazy import

        self.name = name
        # Not every OpenAI-compatible endpoint accepts stream_options, so whether
        # to request usage during streaming is a per-provider capability flag.
        self._stream_usage = stream_usage
        # max_retries=0: the SDK retries any 429 by default, including
        # insufficient_quota which can never succeed. The gateway owns retry and
        # fallback policy, so the vendor SDK should not silently retry underneath it.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def _map_error(self, e: Exception) -> ProviderError:
        """Translate an OpenAI SDK exception into our normalized ProviderError.
        Shared by complete() and stream() so classification lives in one place."""
        from openai import APIConnectionError, APIStatusError, RateLimitError

        if isinstance(e, RateLimitError):
            # 429 is both real rate limits (transient) and insufficient_quota
            # (not) - is_transient reads the message to tell them apart.
            return ProviderError(
                str(e), provider=self.name, retryable=True,
                status_code=429, transient=is_transient(429, str(e)),
            )
        if isinstance(e, APIConnectionError):
            return ProviderError(str(e), provider=self.name, retryable=True, transient=True)
        if isinstance(e, APIStatusError):
            return ProviderError(
                str(e), provider=self.name,
                retryable=should_failover(e.status_code, str(e)),
                status_code=e.status_code,
                transient=is_transient(e.status_code, str(e)),
            )
        raise e  # unknown - let it propagate

    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role.value, "content": m.content} for m in request.messages)
        return messages

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from openai import APIError

        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=self._build_messages(request),  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except APIError as e:
            raise self._map_error(e) from e
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

    async def stream(self, request: ChatRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        from openai import APIError

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        if self._stream_usage:
            # Ask for a final usage-only chunk. Only enabled for providers known
            # to support it, since some compatible endpoints reject the option.
            kwargs["stream_options"] = {"include_usage": True}

        start = time.perf_counter()
        usage: Usage | None = None
        try:
            events = await self._client.chat.completions.create(**kwargs)
            async for event in events:
                # The usage-only final chunk has empty choices.
                if getattr(event, "usage", None):
                    usage = Usage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    )
                if event.choices:
                    piece = event.choices[0].delta.content
                    if piece:
                        yield StreamChunk(delta=piece)
        except APIError as e:
            raise self._map_error(e) from e

        yield StreamChunk(
            finished=True,
            provider=self.name,
            model=model,
            usage=usage or Usage(),
            latency_ms=(time.perf_counter() - start) * 1000,
        )