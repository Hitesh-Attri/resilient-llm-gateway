"""HTTP surface: POST /v1/chat (buffered) and POST /v2/chat (SSE streaming).

The routes stay thin - validate input (Pydantic does it for free), delegate to
the gateway, and map results onto HTTP / SSE. All interesting logic is in core.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.gateway import AllProvidersFailedError
from core.provider import ProviderError
from core.types import ChatRequest, StreamChunk

router = APIRouter(prefix="/v2", tags=["chat-stream"])


def _sse(payload: dict) -> str:
    """Format one Server-Sent Event. The wire format is literally
    'data: <text>\\n\\n' per event - the blank line terminates the event."""
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_to_event(chunk: StreamChunk) -> dict:
    if chunk.finished:
        return {
            "type": "done",
            "provider": chunk.provider,
            "model": chunk.model,
            "usage": chunk.usage.model_dump() if chunk.usage else None,
            "latency_ms": chunk.latency_ms,
        }
    return {"type": "delta", "content": chunk.delta}


@router.post("/chat")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    gateway = request.app.state.gateway

    async def event_source() -> AsyncIterator[str]:
        try:
            async for chunk in gateway.stream(body):
                yield _sse(_chunk_to_event(chunk))
        except (ProviderError, AllProvidersFailedError) as e:
            # Two cases land here:
            #  - failure before any token: the whole stream is this one error event
            #  - failure mid-stream (post-commit): some deltas were already sent,
            #    and this error event tells the client the stream ended abnormally.
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        },
    )
    