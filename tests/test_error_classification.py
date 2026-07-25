"""Classification tests.

These encode a philosophy that got corrected mid-build: for a multi-provider
gateway the default is FAIL OVER, and fail-fast is the narrow exception.

The lesson came from two real failures:
  - Anthropic returns 400 for "credit balance too low" (an account problem).
  - Gemini returns 404 for a retired model ID (a per-rung problem).
Both were briefly misclassified as fail-fast, which stopped the chain at rung 1
and never tried the healthy providers behind it. The model ID, the credits, and
the key all belong to the RUNG, not to the request - so none of them predicts
whether the next provider will succeed.
"""

from __future__ import annotations

import pytest

from core.provider import is_transient, should_failover

CREDIT_MSG = "Your credit balance is too low to access the Anthropic API."
MODEL_GONE = "This model models/gemini-2.5-flash is no longer available to new users."


@pytest.mark.parametrize(
    "status,message",
    [
        (500, "internal server error"),
        (502, "bad gateway"),
        (503, "service unavailable"),
        (429, "rate limit exceeded"),
        (429, "insufficient_quota"),          # OpenAI: no credits, arrives as 429
        (408, "request timeout"),
        (401, "invalid api key"),             # bad key here; another provider may work
        (403, "forbidden"),
        (400, CREDIT_MSG),                    # Anthropic: no credits, arrives as 400
        (404, MODEL_GONE),                    # Gemini: retired model - per rung
        (404, "model not found"),
        (400, "context length exceeded: 300000 > 200000"),  # fits a bigger-context rung
        (400, "temperature is not supported by this model"),  # param differs per model
    ],
)
def test_target_specific_failures_fail_over(status, message):
    # Anything specific to THIS provider/model must try the next rung.
    assert should_failover(status, message) is True


@pytest.mark.parametrize("status", [405, 406, 414, 415])
def test_intrinsically_broken_requests_fail_fast(status):
    # These break identically on every provider, so don't waste the chain.
    assert should_failover(status, "whatever") is False


def test_default_is_fail_over_for_unlisted_4xx():
    # A 400 we've never seen defaults to failover - availability over strictness.
    assert should_failover(400, "some new error string we didn't anticipate") is True


# ---- transience: should the SAME provider be retried? ----------------------

@pytest.mark.parametrize(
    "status,message",
    [
        (500, "boom"),
        (503, "unavailable"),
        (429, "rate limit exceeded"),   # real rate limit -> transient
        (408, "timeout"),
    ],
)
def test_temporary_conditions_are_transient(status, message):
    assert is_transient(status, message) is True


@pytest.mark.parametrize(
    "status,message",
    [
        (429, "insufficient_quota"),    # no money -> not transient, don't retry
        (429, "You exceeded your current quota, check billing"),
        (400, CREDIT_MSG),
        (404, MODEL_GONE),
        (401, "invalid api key"),
    ],
)
def test_permanent_conditions_are_not_transient(status, message):
    assert is_transient(status, message) is False