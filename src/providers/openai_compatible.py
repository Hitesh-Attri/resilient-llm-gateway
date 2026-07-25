"""Base class for every provider that speaks OpenAI's wire format.

It owns the parts that are IDENTICAL across OpenAI, Groq, Gemini, and Ollama:
message building, the buffered call, the streaming loop, and error mapping.
Concrete providers subclass this and override only what differs - identity
(`name`, `default_base_url`), capabilities (`supports_stream_usage`), and, via
the `_extra_create_kwargs` hook, any provider-specific request params (this is
the seam where Gemini's reasoning_effort/thinking_level will plug in later).

Design note: this is inheritance used well - a true is-a relationship where
subclasses specialize behavior, not share unrelated code. The wire protocol
lives here once; the divergences live in the subclasses.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from core.provider import ProviderError, is_transient, should_failover
from core.types import ChatRequest, ChatResponse, StreamChunk, Usage, normalize_finish_reason


class OpenAICompatibleProvider:
    # --- identity / capabilities: subclasses override these ---
    name: str = "openai"
    default_base_url: str | None = None
    supports_stream_usage: bool = False

    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        from openai import AsyncOpenAI  # lazy import: only if this family is used

        # max_retries=0: the SDK retries any 429 by default, including
        # insufficient_quota which can never succeed. The gateway owns retry and
        # fallback policy, so the vendor SDK must not silently retry underneath it.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.default_base_url,
            max_retries=0,
        )

    # --- hooks for subclasses ---
    def _extra_create_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        """Provider-specific params merged into every create() call. Empty by
        default; e.g. GeminiProvider will return {"reasoning_effort": ...}."""
        return {}

    # --- shared machinery ---
    def _build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role.value, "content": m.content} for m in request.messages)
        return messages

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

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from openai import APIError

        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=self._build_messages(request),  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                **self._extra_create_kwargs(request),
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
            finish_reason=normalize_finish_reason(choice.finish_reason),
        )

    async def stream(self, request: ChatRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        from openai import APIError

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            **self._extra_create_kwargs(request),
        }
        if self.supports_stream_usage:
            # Ask for a final usage-only chunk. Only enabled for providers known
            # to support it, since some compatible endpoints reject the option.
            kwargs["stream_options"] = {"include_usage": True}

        start = time.perf_counter()
        usage: Usage | None = None
        finish_raw: str | None = None
        try:
            events = await self._client.chat.completions.create(**kwargs)
            async for event in events:
                if getattr(event, "usage", None):  # usage-only final chunk
                    usage = Usage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    )
                if event.choices:
                    choice = event.choices[0]
                    if choice.finish_reason:  # set on the last content chunk
                        finish_raw = choice.finish_reason
                    piece = choice.delta.content
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
            finish_reason=normalize_finish_reason(finish_raw),
        )