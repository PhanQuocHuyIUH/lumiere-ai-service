from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.services.redis_client import get_redis

logger = logging.getLogger(__name__)

_JOB_TTL_SECONDS = 86400  # 24 h


async def create_job(job_id: str) -> None:
    r = get_redis()
    now = datetime.now(timezone.utc).isoformat()
    await r.hset(
        f"ai:job:{job_id}",
        mapping={"job_id": job_id, "status": "PENDING", "message": "", "started_at": now, "completed_at": ""},
    )
    await r.expire(f"ai:job:{job_id}", _JOB_TTL_SECONDS)


async def update_job(job_id: str, status: str, message: str = "") -> None:
    r = get_redis()
    mapping: dict[str, str] = {"status": status, "message": message}
    if status in ("COMPLETED", "FAILED"):
        mapping["completed_at"] = datetime.now(timezone.utc).isoformat()
    await r.hset(f"ai:job:{job_id}", mapping=mapping)


async def get_job(job_id: str) -> dict[str, str] | None:
    r = get_redis()
    data = await r.hgetall(f"ai:job:{job_id}")
    return dict(data) if data else None
