"""Structured-output tests.

The validator is a pure function, tested directly. The gateway behavior - that a
schema violation fails OVER to the next provider and a valid response attaches
`parsed` - is tested with fake providers and no network.
"""

from __future__ import annotations

import pytest

from core.gateway import AllProvidersFailedError, LLMGateway, Target
from core.structured import StructuredOutputError, validate_json
from core.types import ChatRequest, ChatResponse, Message, Role, Usage

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
    "required": ["label", "score"],
    "additionalProperties": False,
}


# ---- the pure validator ----------------------------------------------------

def test_valid_json_matching_schema_parses():
    out = validate_json('{"label": "positive", "score": 0.9}', SCHEMA)
    assert out == {"label": "positive", "score": 0.9}


def test_non_json_raises():
    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        validate_json("this is prose, not json", SCHEMA)


def test_json_violating_schema_raises():
    with pytest.raises(StructuredOutputError, match="did not match schema"):
        validate_json('{"label": "positive"}', SCHEMA)  # missing required "score"


def test_wrong_type_raises():
    with pytest.raises(StructuredOutputError, match="did not match schema"):
        validate_json('{"label": "x", "score": "high"}', SCHEMA)  # score not a number


# ---- gateway integration ---------------------------------------------------

class CannedProvider:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self._content = content
        self.calls = 0

    async def complete(self, request, *, model):
        self.calls += 1
        return ChatResponse(
            content=self._content, model=model, provider=self.name,
            usage=Usage(input_tokens=1, output_tokens=1), latency_ms=1.0,
        )


def _req():
    return ChatRequest(
        messages=[Message(role=Role.user, content="classify")],
        response_schema=SCHEMA,
    )


@pytest.mark.asyncio
async def test_valid_structured_output_attaches_parsed():
    p = CannedProvider("groq", '{"label": "positive", "score": 0.8}')
    gw = LLMGateway([Target(p, "m")])

    resp = await gw.complete(_req())

    assert resp.parsed == {"label": "positive", "score": 0.8}
    assert resp.provider == "groq"


@pytest.mark.asyncio
async def test_schema_violation_fails_over_to_next_provider():
    bad = CannedProvider("groq", '{"label": "positive"}')       # missing score
    good = CannedProvider("gemini", '{"label": "positive", "score": 0.7}')
    gw = LLMGateway([Target(bad, "m1"), Target(good, "m2")])

    resp = await gw.complete(_req())

    assert resp.provider == "gemini"      # bad one was skipped
    assert resp.parsed == {"label": "positive", "score": 0.7}
    assert bad.calls == 1 and good.calls == 1


@pytest.mark.asyncio
async def test_all_providers_violating_schema_raises_aggregate():
    a = CannedProvider("a", "not json at all")
    b = CannedProvider("b", '{"wrong": "shape"}')
    gw = LLMGateway([Target(a, "m1"), Target(b, "m2")])

    with pytest.raises(AllProvidersFailedError) as exc:
        await gw.complete(_req())

    assert len(exc.value.errors) == 2


@pytest.mark.asyncio
async def test_no_schema_means_no_validation_or_parsed():
    p = CannedProvider("groq", "just some prose")
    gw = LLMGateway([Target(p, "m")])

    resp = await gw.complete(ChatRequest(messages=[Message(role=Role.user, content="hi")]))

    assert resp.parsed is None
    assert resp.content == "just some prose"