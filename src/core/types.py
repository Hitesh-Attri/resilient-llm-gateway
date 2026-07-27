"""Provider-agnostic request and response types.

The whole point of the gateway is that callers speak ONE vocabulary and the
provider adapters translate it into each vendor's dialect. These models are that
vocabulary. Nothing here knows about OpenAI, Anthropic, or Bedrock.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class ReasoningEffort(str, Enum):
    """How hard a reasoning-capable model should think before answering. Maps to
    OpenAI's `reasoning_effort` and, via Google's compat layer, to Gemini's
    `thinking_level`. Lower effort = fewer thinking tokens = less chance of the
    thinking budget eating your whole max_tokens and truncating the answer."""

    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class ChatRequest(BaseModel):
    """What a caller sends. Note `system` is separate from `messages`.

    This is a deliberate normalization decision: Anthropic and Bedrock treat the
    system prompt as a top-level field, while OpenAI treats it as a message with
    role="system". We pick the cleaner model (separate field) and let the OpenAI
    adapter fold it back into the messages list.
    """

    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    # 4096 default: generous enough not to truncate thorough answers, yet safe
    # across every provider (Anthropic requires max_tokens and older Claude models
    # cap output at 4096). Cap at 32768 as an abuse guard - a per-model-aware or
    # configurable cap belongs with the rate-limiting/budgets slice, not here.
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # Optional per-request override. When None, a reasoning-capable provider uses
    # its configured default; non-reasoning providers ignore it entirely.
    reasoning_effort: ReasoningEffort | None = None
    # A JSON Schema. When set, the gateway asks the provider for conforming JSON,
    # validates it server-side, and fails over if the model violates the schema.
    response_schema: dict[str, Any] | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class FinishReason(str, Enum):
    """Normalized reason generation stopped. Each provider has its own vocabulary
    (OpenAI 'length', Anthropic/Bedrock 'max_tokens', ...); we map them all here
    so a caller can check `finish_reason == FinishReason.length` without knowing
    which provider served the request. `length` is the one that means 'truncated'."""

    stop = "stop"                      # natural completion
    length = "length"                  # hit max_tokens - output was cut off
    content_filter = "content_filter"  # blocked by safety/guardrails
    tool_call = "tool_call"            # stopped to call a tool
    other = "other"                    # anything unmapped


# All providers' raw stop reasons -> our normalized enum. The raw strings don't
# collide across providers, so one table covers them all.
_FINISH_REASON_MAP = {
    # OpenAI-compatible
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    "content_filter": FinishReason.content_filter,
    "tool_calls": FinishReason.tool_call,
    "function_call": FinishReason.tool_call,
    # Anthropic
    "end_turn": FinishReason.stop,
    "stop_sequence": FinishReason.stop,
    "max_tokens": FinishReason.length,
    "tool_use": FinishReason.tool_call,
    "refusal": FinishReason.content_filter,
    # Bedrock Converse
    "content_filtered": FinishReason.content_filter,
    "guardrail_intervened": FinishReason.content_filter,
}


def normalize_finish_reason(raw: str | None) -> FinishReason:
    if raw is None:
        return FinishReason.other
    return _FINISH_REASON_MAP.get(raw.lower(), FinishReason.other)


class ChatResponse(BaseModel):
    """What every provider returns, normalized. `provider` and `model` tell you
    which target actually served the request - critical once fallback is in play.
    `finish_reason` tells you WHY it stopped - `length` means it was truncated."""

    content: str
    model: str
    provider: str
    usage: Usage
    latency_ms: float
    finish_reason: FinishReason | None = None
    # Populated only when the request set response_schema and validation passed:
    # the response content parsed into a dict, so callers don't re-parse.
    parsed: dict[str, Any] | None = None


class StreamChunk(BaseModel):
    """One item in a streamed response.

    Two shapes flow through the same type:
      - a text piece:   delta="Hello", finished=False
      - the final event: finished=True, with provider/model/usage/latency/finish_reason

    Keeping both in one type lets a provider's stream be a single async iterator
    the gateway can relay without special-casing, and the SSE layer maps each to
    a `delta` or `done` event."""

    delta: str = ""
    finished: bool = False
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    finish_reason: FinishReason | None = None