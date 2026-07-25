"""AWS Bedrock adapter, using the Converse API.

Converse is the right choice here: it gives one message shape across every model
on Bedrock (Claude, Llama, Mistral, ...), so this single adapter covers many
models. Its shape:
  - system is a list: [{"text": ...}]
  - messages are [{"role", "content": [{"text": ...}]}]
  - inference params go under inferenceConfig (maxTokens, temperature)
  - response text: resp["output"]["message"]["content"][0]["text"]
  - usage: resp["usage"]["inputTokens"] / ["outputTokens"]

boto3 is synchronous, so we offload the blocking call to a thread with
asyncio.to_thread to keep the event loop free. For higher throughput you'd swap
in aioboto3, but to_thread is honest and production-acceptable for now.
Auth is your standard AWS credential chain (IAM role on ECS) - no API key here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from core.provider import ProviderError
from core.types import ChatRequest, ChatResponse, StreamChunk, Usage

# Bedrock signals failures with botocore error codes, not HTTP status, so it
# classifies here rather than via core.provider.should_failover - but with the
# same default: FAIL OVER unless the error means every provider would reject the
# request identically. Since ChatRequest is already validated upstream, the only
# botocore code that qualifies is a genuinely malformed Converse payload.
# (Unknown model IDs also raise ValidationException, but those are per-rung and
# SHOULD fail over - so we accept the rare false fail-fast rather than sniff
# messages here. If that bites, narrow it by inspecting the message.)
_FAIL_FAST_CODES = frozenset({
    "SerializationException",  # malformed request structure - broken everywhere
})

# Codes worth RETRYING on the same target (transient). Everything else that
# fails over (access denied, quota) is permanent-for-this-target, so retrying it
# only wastes backoff before falling over.
_TRANSIENT_CODES = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
})


class BedrockProvider:
    name = "bedrock"

    def __init__(self, region: str) -> None:
        import boto3  # lazy import

        self._client = boto3.client("bedrock-runtime", region_name=region)

    def _build_kwargs(self, request: ChatRequest, model: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": [
                {"role": m.role.value, "content": [{"text": m.content}]} for m in request.messages
            ],
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]
        return kwargs

    def _map_error(self, e: Exception) -> ProviderError:
        from botocore.exceptions import ClientError, EndpointConnectionError

        if isinstance(e, EndpointConnectionError):
            return ProviderError(str(e), provider=self.name, retryable=True, transient=True)
        if isinstance(e, ClientError):
            code = e.response.get("Error", {}).get("Code", "")
            return ProviderError(
                str(e), provider=self.name,
                retryable=code not in _FAIL_FAST_CODES,
                transient=code in _TRANSIENT_CODES,
            )
        raise e

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        start = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                self._client.converse, **self._build_kwargs(request, model)
            )
        except (ClientError, BotoCoreError) as e:
            raise self._map_error(e) from e
        latency_ms = (time.perf_counter() - start) * 1000

        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp["usage"]
        return ChatResponse(
            content=text,
            model=model,
            provider=self.name,
            usage=Usage(input_tokens=usage["inputTokens"], output_tokens=usage["outputTokens"]),
            latency_ms=latency_ms,
        )

    async def stream(self, request: ChatRequest, *, model: str) -> AsyncIterator[StreamChunk]:
        """converse_stream returns a *synchronous*, blocking event stream. We
        can't iterate it directly in async code without blocking the event loop,
        so we run the blocking iteration in a worker thread and hand items back
        through an asyncio.Queue. This sync->async bridge is the general pattern
        for consuming any blocking iterator from asyncio."""
        from botocore.exceptions import BotoCoreError, ClientError

        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        _DONE = object()  # sentinel marking normal end of stream

        def produce() -> None:
            # Runs in a worker thread; must not touch the event loop directly,
            # so it hands items back via loop.call_soon_threadsafe.
            try:
                resp = self._client.converse_stream(**self._build_kwargs(request, model))
                for event in resp["stream"]:
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except (ClientError, BotoCoreError) as e:
                loop.call_soon_threadsafe(queue.put_nowait, self._map_error(e))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        usage = Usage()
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, ProviderError):
                    raise item
                event: dict[str, Any] = item  # type: ignore[assignment]
                if "contentBlockDelta" in event:
                    piece = event["contentBlockDelta"]["delta"].get("text", "")
                    if piece:
                        yield StreamChunk(delta=piece)
                elif "metadata" in event:
                    u = event["metadata"].get("usage", {})
                    usage = Usage(
                        input_tokens=u.get("inputTokens", 0),
                        output_tokens=u.get("outputTokens", 0),
                    )
        finally:
            await producer  # ensure the worker thread is joined

        yield StreamChunk(
            finished=True, provider=self.name, model=model, usage=usage,
            latency_ms=(time.perf_counter() - start) * 1000,
        )