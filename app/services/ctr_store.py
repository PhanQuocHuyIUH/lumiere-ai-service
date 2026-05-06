from __future__ import annotations

import logging

from app.services.redis_client import get_redis

logger = logging.getLogger(__name__)

_IMP_KEY = "ai:ctr:imp"
_CLK_KEY = "ai:ctr:clk"


async def record_events(events: list[dict]) -> int:
    """Increment impression and click counters from a list of feedback events."""
    processed = 0
    try:
        r = get_redis()
        pipe = r.pipeline()
        for event in events:
            if event.get("type") != "RECOMMENDATION_CLICK":
                continue
            shown: list[int] = event.get("shown") or []
            clicked: list[int] = event.get("clicked") or []
            for item_id in shown:
                pipe.hincrby(_IMP_KEY, str(item_id), 1)
            for item_id in clicked:
                pipe.hincrby(_CLK_KEY, str(item_id), 1)
            processed += 1
        await pipe.execute()
    except Exception as exc:
        logger.warning("CTR store write failed: %s", exc)
    return processed


async def get_ctr_map(item_ids: list[int]) -> dict[int, float]:
    """Return CTR (clicks/impressions) per item id; items with no data get 0.0."""
    if not item_ids:
        return {}
    try:
        r = get_redis()
        keys = [str(i) for i in item_ids]
        impressions = await r.hmget(_IMP_KEY, keys)
        clicks = await r.hmget(_CLK_KEY, keys)
        result: dict[int, float] = {}
        for idx, item_id in enumerate(item_ids):
            imp = int(impressions[idx] or 0)
            clk = int(clicks[idx] or 0)
            result[item_id] = clk / imp if imp > 0 else 0.0
        return result
    except Exception as exc:
        logger.warning("CTR store read failed: %s", exc)
        return {i: 0.0 for i in item_ids}
