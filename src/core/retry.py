"""Retry policy and backoff math.

Kept separate from the gateway and free of any sleeping or I/O so the delay
calculation is a pure function you can unit-test exhaustively.

Backoff strategy is exponential with FULL JITTER (the AWS "Exponential Backoff
and Jitter" recommendation):

    delay = uniform(0, min(max_delay, base_delay * 2**n))

Why jitter at all: if 50 requests all hit a rate limit at the same instant and
every client backs off by exactly 2s, they retry in the same instant too and
stampede the provider again. Randomizing each delay spreads the retries out.
Full jitter (uniform from zero) spreads them the most.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3     # total tries per target: 1 initial + (max_attempts-1) retries
    base_delay: float = 0.5   # seconds; the first backoff ceiling
    max_delay: float = 8.0    # cap on any single backoff, so 2**n can't explode

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")


def full_jitter_delay(retry_index: int, policy: RetryPolicy, rand=random.random) -> float:
    """Backoff before a retry.

    retry_index is 0-based: 0 for the first retry, 1 for the second, etc., so the
    exponential ceiling grows base, 2*base, 4*base, ... capped at max_delay.
    `rand` is injectable (returns [0, 1)) so tests can make the delay deterministic.
    """
    ceiling = min(policy.max_delay, policy.base_delay * (2 ** retry_index))
    return rand() * ceiling