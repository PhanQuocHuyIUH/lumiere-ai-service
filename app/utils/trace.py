from __future__ import annotations

import uuid

from fastapi import Request


TRACE_HEADER = "X-Request-Id"


def get_trace_id(request: Request) -> str:
    incoming = request.headers.get(TRACE_HEADER)
    if incoming and incoming.strip():
        return incoming.strip()
    return str(uuid.uuid4())

