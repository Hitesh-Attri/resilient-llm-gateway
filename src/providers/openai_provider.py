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

from core.provider import ProviderError, should_failover
from core.types import ChatRequest, ChatResponse, Usage


class OpenAIProvider:
    """Serves OpenAI *and* any OpenAI-compatible endpoint (Groq, Gemini's compat
    layer, OpenRouter, Cerebras, Ollama, vLLM) by pointing `base_url` elsewhere.

    A large slice of the ecosystem cloned OpenAI's wire format, so one adapter
    plus a URL covers many vendors. This is the payoff of the adapter pattern:
    four new providers for zero new adapter code.

    `name` is overridable so logs and ChatResponse.provider say "groq", not
    "openai" - otherwise you can't tell who actually served a request.
    """

    def __init__(self, api_key: str, *, base_url: str | None = None, name: str = "openai") -> None:
        from openai import AsyncOpenAI  # lazy import

        self.name = name
        # max_retries=0: the SDK retries any 429 by default, including
        # insufficient_quota which can never succeed. The gateway owns retry and
        # fallback policy, so the vendor SDK should not silently retry underneath it.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

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
            # insufficient_quota also arrives as a 4xx, so classify on the
            # message too - see should_failover.
            raise ProviderError(
                str(e),
                provider=self.name,
                retryable=should_failover(e.status_code, str(e)),
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