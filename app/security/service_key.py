from __future__ import annotations

from fastapi import Header, HTTPException

from app.settings import settings


SERVICE_KEY_HEADER = "X-AI-Service-Key"


def require_service_key(
    x_ai_service_key: str | None = Header(default=None, alias=SERVICE_KEY_HEADER),
) -> None:
    configured = settings.ai_service_key
    if not configured or not configured.strip():
        raise HTTPException(status_code=401, detail="AI service key not configured")

    if x_ai_service_key is None or not x_ai_service_key.strip():
        raise HTTPException(status_code=401, detail="Unauthorized")

    if x_ai_service_key != configured:
        raise HTTPException(status_code=403, detail="Forbidden")

