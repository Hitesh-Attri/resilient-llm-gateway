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
from typing import Any

from core.provider import ProviderError
from core.types import ChatRequest, ChatResponse, Usage

_RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
}


class BedrockProvider:
    name = "bedrock"

    def __init__(self, region: str) -> None:
        import boto3  # lazy import

        self._client = boto3.client("bedrock-runtime", region_name=region)

    async def complete(self, request: ChatRequest, *, model: str) -> ChatResponse:
        from botocore.exceptions import ClientError, EndpointConnectionError

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

        start = time.perf_counter()
        try:
            resp = await asyncio.to_thread(self._client.converse, **kwargs)
        except EndpointConnectionError as e:
            raise ProviderError(str(e), provider=self.name, retryable=True) from e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            raise ProviderError(
                str(e),
                provider=self.name,
                retryable=code in _RETRYABLE_CODES,
            ) from e
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