"""Provider-agnostic request and response types.

The whole point of the gateway is that callers speak ONE vocabulary and the
provider adapters translate it into each vendor's dialect. These models are that
vocabulary. Nothing here knows about OpenAI, Anthropic, or Bedrock.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    """What a caller sends. Note `system` is separate from `messages`.

    This is a deliberate normalization decision: Anthropic and Bedrock treat the
    system prompt as a top-level field, while OpenAI treats it as a message with
    role="system". We pick the cleaner model (separate field) and let the OpenAI
    adapter fold it back into the messages list.
    """

    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatResponse(BaseModel):
    """What every provider returns, normalized. `provider` and `model` tell you
    which target actually served the request - critical once fallback is in play."""

    content: str
    model: str
    provider: str
    usage: Usage
    latency_ms: float


class StreamChunk(BaseModel):
    """One item in a streamed response.

    Two shapes flow through the same type:
      - a text piece:   delta="Hello", finished=False
      - the final event: finished=True, with provider/model/usage/latency filled in

    Keeping both in one type lets a provider's stream be a single async iterator
    the gateway can relay without special-casing, and the SSE layer maps each to
    a `delta` or `done` event."""

    delta: str = ""
    finished: bool = False
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None