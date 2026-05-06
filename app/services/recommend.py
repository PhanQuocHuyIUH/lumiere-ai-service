from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.schemas.ai import RecommendItem, RecommendRequest, RecommendResponse
from app.services.ctr_store import get_ctr_map
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MODEL_VERSION = "recommend_centroid_v1"

# Items in the same category as any cart item receive this score multiplier.
_SAME_CATEGORY_PENALTY = 0.5


async def recommend_items(req: RecommendRequest) -> RecommendResponse:
    if not req.current_items:
        logger.debug("Recommend request with empty cart")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    try:
        store = get_vector_store()
    except RuntimeError as exc:
        logger.warning("VectorStore unavailable for recommendation (%s)", exc)
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    # ── Step 1: Fetch vectors for all cart items ───────────────────────────────
    cart_docs = await store.get_by_ids(req.current_items)
    if not cart_docs:
        logger.warning("No cart documents found for items: %s", req.current_items)
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    # ── Step 2: Compute Centroid Vector (element-wise mean) ────────────────────
    vectors = [d["vector"] for d in cart_docs if d.get("vector")]
    if not vectors:
        logger.warning("No vectors found in cart documents")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    centroid: list[float] = np.mean(vectors, axis=0).tolist()
    logger.info("Computing recommendations: cart_items=%d, top_k=%d", len(req.current_items), req.top_k)

    # ── Step 3: Cosine Similarity search using centroid ────────────────────────
    # Fetch extra results to allow for post-processing filtering
    fetch_k = req.top_k * 3
    raw_results = await store.search_vector(
        query_vector=centroid,
        top_k=fetch_k,
        exclude_ids=req.current_items,
    )

    if not raw_results:
        logger.warning("No results from vector search")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    # ── Step 4: Blend vector score with CTR ───────────────────────────────────
    candidate_ids = [int(r["id"]) for r in raw_results]
    ctr_map = await get_ctr_map(candidate_ids)

    cart_categories: set[str] = {
        d["payload"].get("category", "") for d in cart_docs if d.get("payload")
    }

    scored: list[dict[str, Any]] = []
    for r in raw_results:
        item_id = int(r["id"])
        item_category: str = r["payload"].get("category", "")
        vector_score: float = float(r["score"])
        ctr: float = ctr_map.get(item_id, 0.0)

        # Score = 0.6 × VectorScore + 0.4 × CTR
        blended_score: float = 0.6 * vector_score + 0.4 * ctr

        if item_category and item_category in cart_categories:
            blended_score *= _SAME_CATEGORY_PENALTY
            reason = "similar"
        else:
            reason = "combo+popularity"

        scored.append({"id": item_id, "score": blended_score, "reason": reason, "payload": r["payload"]})

    # Re-rank by penalised score
    scored.sort(key=lambda x: x["score"], reverse=True)

    items = [
        RecommendItem(
            menu_item_id=r["id"],
            score=round(r["score"], 4),
            reason=r["reason"],
        )
        for r in scored[: req.top_k]
    ]

    logger.info("Recommendation results: returned=%d items", len(items))
    return RecommendResponse(success=True, source="model", items=items, model_version=MODEL_VERSION)
