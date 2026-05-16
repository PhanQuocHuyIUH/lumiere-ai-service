from __future__ import annotations

import logging
from typing import Any

from app.schemas.ai import RecommendItem, RecommendRequest, RecommendResponse
from app.services.ctr_store import get_ctr_map
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

MODEL_VERSION = "recommend_per_item_rrf_v2"

# Items in the same category as any cart item receive this score multiplier.
_SAME_CATEGORY_PENALTY = 0.5

# Reciprocal Rank Fusion constant (Cormack et al. 2009). Matches vector_store.
_RRF_K = 60

# Weight split for the final blended score.
_VECTOR_WEIGHT = 0.6
_CTR_WEIGHT = 0.4


async def recommend_items(req: RecommendRequest) -> RecommendResponse:
    if not req.current_items:
        logger.debug("Recommend request with empty cart")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    try:
        store = get_vector_store()
    except RuntimeError as exc:
        logger.warning("VectorStore unavailable for recommendation (%s)", exc)
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    # ── Step 1: Fetch each cart item's own vector ──────────────────────────────
    cart_docs = await store.get_by_ids(req.current_items)
    if not cart_docs:
        logger.warning("No cart documents found for items: %s", req.current_items)
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    cart_vectors = [(int(d["id"]), d["vector"]) for d in cart_docs if d.get("vector")]
    if not cart_vectors:
        logger.warning("No vectors found in cart documents")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    cart_categories: set[str] = {
        d["payload"].get("category", "") for d in cart_docs if d.get("payload")
    }

    # ── Step 2: Retrieve candidates per cart item and fuse with RRF ────────────
    # Per-item retrieval avoids the "centroid pulled between two unrelated tastes"
    # problem: each cart item votes for its own neighborhood; the fusion step
    # rewards items that appear near *several* cart items (true cross-sell).
    # A single batched Qdrant call keeps latency O(1) round-trip regardless of
    # cart size — a 15-item group order would otherwise cost 15 sequential RTTs.
    fetch_per_item = max(req.top_k * 3, 15)
    exclude_ids = list(req.current_items)
    query_vectors = [vec for _, vec in cart_vectors]

    batch_results = await store.search_vectors_batch(
        query_vectors=query_vectors,
        top_k=fetch_per_item,
        exclude_ids=exclude_ids,
    )

    candidates: dict[int, dict[str, Any]] = {}
    for hits in batch_results:
        for rank, hit in enumerate(hits):
            cand_id = int(hit["id"])
            if cand_id not in candidates:
                candidates[cand_id] = {
                    "rrf": 0.0,
                    "payload": hit["payload"],
                    "hit_count": 0,
                }
            candidates[cand_id]["rrf"] += 1.0 / (_RRF_K + rank + 1)
            candidates[cand_id]["hit_count"] += 1

    if not candidates:
        logger.warning("No results from per-item vector search")
        return RecommendResponse(success=True, source="fallback", items=[], model_version=MODEL_VERSION)

    # Normalize RRF score to [0, 1] for stable blending with CTR
    max_rrf = max(c["rrf"] for c in candidates.values())
    if max_rrf <= 0:
        max_rrf = 1.0

    # ── Step 3: Pull Bayesian-smoothed CTR for each candidate ──────────────────
    candidate_ids = list(candidates.keys())
    ctr_map = await get_ctr_map(candidate_ids)

    logger.info(
        "Computing recommendations: cart_items=%d, candidates=%d, top_k=%d",
        len(cart_vectors), len(candidates), req.top_k,
    )

    # ── Step 4: Blend RRF and CTR, apply category penalty ─────────────────────
    scored: list[dict[str, Any]] = []
    for cand_id, info in candidates.items():
        vector_score = info["rrf"] / max_rrf
        ctr = ctr_map.get(cand_id, 0.0)
        blended = _VECTOR_WEIGHT * vector_score + _CTR_WEIGHT * ctr

        category = info["payload"].get("category", "")
        if category and category in cart_categories:
            blended *= _SAME_CATEGORY_PENALTY
            reason = "similar"
        else:
            reason = "combo+popularity"

        scored.append({"id": cand_id, "score": blended, "reason": reason})

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
