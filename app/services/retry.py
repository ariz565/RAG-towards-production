"""Professional-grade retry policy for network calls (LLM / embeddings).

Implements the techniques a real retry policy needs — not "retry on any error":

1. **Retry only transient failures.** HTTP 408/425/429/500/502/503/504, plus
   network timeouts and connection errors. **Never** retry deterministic 4xx
   (400/401/403/404/409/422): retrying won't fix a bad request, bad key, or
   missing resource — it just wastes quota and latency.
2. **Exponential backoff** with a hard **max-delay cap** (`min(cap, base·2^n)`).
3. **Full jitter** (AWS "Exponential Backoff and Jitter") — sleep a random
   amount in `[0, computed_delay]` to avoid synchronized retry storms.
4. **Respect `Retry-After`** (seconds or HTTP-date) when the server sends it.
5. **Bounded attempts** and an optional **overall deadline**.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Transient HTTP statuses worth retrying.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# Exception type-name hints for transient network errors without a status code.
_TRANSIENT_NAME_HINTS = (
    "timeout", "connection", "temporarily", "unavailable",
    "ratelimit", "overloaded", "serviceunavailable",
)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2            # total attempts = max_retries + 1
    base_delay: float = 0.5         # seconds
    max_delay: float = 20.0         # cap per-sleep
    jitter: bool = True             # full jitter
    respect_retry_after: bool = True
    overall_deadline: float | None = None  # seconds across all attempts (optional)
    retryable_status: frozenset = RETRYABLE_STATUS


# ── Error classification ─────────────────────────────────────────────


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if not value:
        return None
    try:
        return max(0.0, float(value))  # delta-seconds form
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def classify(exc: Exception, policy: RetryPolicy) -> tuple[bool, float | None]:
    """Return (is_retryable, retry_after_seconds | None)."""
    status = _status_code(exc)
    if status is not None:
        if status in policy.retryable_status:
            return True, (_retry_after_seconds(exc) if policy.respect_retry_after else None)
        if 400 <= status < 500:
            return False, None        # deterministic client error → do not retry
        if status >= 500:
            return True, (_retry_after_seconds(exc) if policy.respect_retry_after else None)

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True, None

    name = type(exc).__name__.lower()
    if any(hint in name for hint in _TRANSIENT_NAME_HINTS):
        return True, None

    return False, None                # unknown → fail fast (safer than blind retry)


# ── Backoff ──────────────────────────────────────────────────────────


def compute_delay(attempt: int, policy: RetryPolicy, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, policy.max_delay)
    delay = min(policy.max_delay, policy.base_delay * (2 ** attempt))
    if policy.jitter:
        delay = random.uniform(0, delay)  # full jitter
    return delay


# ── Driver ───────────────────────────────────────────────────────────


async def retry_async(factory, *, policy: RetryPolicy, label: str = "call"):
    """Run ``await factory()`` under the retry policy.

    ``factory`` is a zero-arg callable returning a fresh coroutine each attempt.
    """
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return await factory()
        except Exception as e:
            last_error = e
            retryable, retry_after = classify(e, policy)
            if not retryable or attempt >= policy.max_retries:
                raise
            delay = compute_delay(attempt, policy, retry_after)
            if (
                policy.overall_deadline is not None
                and (time.monotonic() - started) + delay > policy.overall_deadline
            ):
                logger.warning(f"{label}: retry budget exhausted; giving up.")
                raise
            logger.warning(
                f"{label}: transient error (attempt {attempt + 1}/{policy.max_retries + 1}): "
                f"{type(e).__name__}: {e}; retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)
    raise last_error  # pragma: no cover
