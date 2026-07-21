"""HTTP surface. One endpoint for now: POST /v1/chat.

The route is deliberately thin - it validates input (Pydantic does this for
free), delegates to the gateway, and maps gateway failures onto HTTP status
codes. All the interesting logic lives in the core, not here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from core.gateway import AllProvidersFailedError
from core.provider import ProviderError
from core.types import ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    gateway = request.app.state.gateway
    try:
        return await gateway.complete(body)
    except ProviderError as e:
        # Non-retryable error surfaced straight through (e.g. bad request / auth).
        status = e.status_code or 502
        raise HTTPException(status_code=status, detail=str(e)) from e
    except AllProvidersFailedError as e:
        # Every rung of the chain failed transiently -> upstream unavailable.
        raise HTTPException(status_code=503, detail=str(e)) from e