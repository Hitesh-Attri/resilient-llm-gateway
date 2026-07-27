"""Structured-output validation.

This is the conformance guarantee. Rather than trusting each provider's native
'strict' mode (which varies and some compat layers ignore), we validate the
returned content against the caller's JSON Schema ourselves. A violation is a
failed attempt the gateway can fail over on - so "the model didn't honor the
schema" is handled by the same resilience as any other provider failure.

Kept as a pure function (string + schema -> dict, or raise) so it's fully
testable without any provider or network.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema


class StructuredOutputError(Exception):
    """The model's output was not valid JSON, or didn't match the schema."""


def validate_json(content: str, schema: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"response was not valid JSON: {e}") from e

    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        # e.message is concise; the full error is huge. Keep it readable.
        raise StructuredOutputError(f"response did not match schema: {e.message}") from e

    return data