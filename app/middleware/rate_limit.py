from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    tokens: float
    last_refill_at: float


class InMemoryTokenBucketLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}

    def _key(self, request: Request) -> str:
        ip = request.client.host if request.client else "unknown"
        return f"{ip}:{request.url.path}"

    def allow(self, request: Request, *, capacity: int, refill_per_second: float) -> bool:
        key = self._key(request)
        now = time.time()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(capacity), last_refill_at=now)
            self._buckets[key] = bucket

        elapsed = max(0.0, now - bucket.last_refill_at)
        bucket.tokens = min(float(capacity), bucket.tokens + elapsed * refill_per_second)
        bucket.last_refill_at = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


limiter = InMemoryTokenBucketLimiter()


def enforce_rate_limit(request: Request) -> None:
    # Default: 60 req/min per ip+path.
    capacity = 60
    refill = 60 / 60.0

    if request.url.path == "/ai/chatbot":
        capacity = 15
        refill = 15 / 60.0

    if not limiter.allow(request, capacity=capacity, refill_per_second=refill):
        raise HTTPException(status_code=429, detail="Too many requests")

