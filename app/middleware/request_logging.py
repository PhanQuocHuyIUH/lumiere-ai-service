from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response

from app.utils.trace import get_trace_id


async def log_request(request: Request, call_next: Callable[[Request], Response]) -> Response:
    started = time.perf_counter()
    trace_id = get_trace_id(request)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Keep logs minimal; infra can parse these fields.
    request.app.logger.info(
        "request path=%s method=%s status=%s elapsed_ms=%.1f trace_id=%s",
        request.url.path,
        request.method,
        response.status_code,
        elapsed_ms,
        trace_id,
    )

    # Propagate trace id for debugging (safe to add even if backend doesn't use it yet).
    response.headers.setdefault("X-Request-Id", trace_id)
    return response

