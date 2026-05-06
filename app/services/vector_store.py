from __future__ import annotations

import logging
import re
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant for Reciprocal Rank Fusion


class VectorStore:
    """Async wrapper quanh ``AsyncQdrantClient`` cho collection menu items.

    Yêu cầu ``qdrant-client >= 1.11`` (đã lock trong pyproject.toml).
    """

    def __init__(self, url: str, api_key: str | None, collection: str, vector_dim: int) -> None:
        self._url = url
        self._api_key = api_key or None
        self._collection = collection
        self._dim = vector_dim
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)
        await self._ensure_collection()
        logger.info("VectorStore ready: collection=%s  dim=%d", self._collection, self._dim)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    async def _ensure_collection(self) -> None:
        assert self._client
        if not await self._client.collection_exists(self._collection):
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                ),
            )
            logger.info("Created collection '%s' with HNSW (m=16, ef_construct=200)", self._collection)

    @property
    def _c(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("VectorStore not connected — call connect() first")
        return self._client

    # ── Write ──────────────────────────────────────────────────────────────────

    async def upsert(self, item_id: int, vector: list[float], payload: dict[str, Any]) -> None:
        await self._c.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=item_id, vector=vector, payload=payload)],
        )

    async def delete(self, item_id: int) -> None:
        await self._c.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[item_id]),
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    async def get_by_ids(self, item_ids: list[int]) -> list[dict[str, Any]]:
        if not item_ids:
            return []
        results = await self._c.retrieve(
            collection_name=self._collection,
            ids=item_ids,
            with_vectors=True,
            with_payload=True,
        )
        return [{"id": int(p.id), "vector": p.vector, "payload": p.payload or {}} for p in results]

    async def search_vector(
        self,
        query_vector: list[float],
        top_k: int,
        exclude_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        exclude_set = set(exclude_ids or [])
        response = await self._c.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k + len(exclude_set),
            with_payload=True,
        )
        output = []
        for r in response.points:
            r_id = int(r.id)
            if r_id not in exclude_set:
                output.append({"id": r_id, "score": float(r.score), "payload": r.payload or {}})
        return output[:top_k]

    async def hybrid_search_rrf(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        exclude_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid Search: dense vector + BM25 keyword, fused with Reciprocal Rank Fusion."""
        fetch_n = max(top_k * 3, 20)

        dense_response = await self._c.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=fetch_n,
            with_payload=True,
        )
        dense_hits = dense_response.points  # list[ScoredPoint]

        all_docs = await self._scroll_all()
        payload_map: dict[int, dict] = {int(d["id"]): d["payload"] for d in all_docs}

        for r in dense_hits:
            if int(r.id) not in payload_map:
                payload_map[int(r.id)] = r.payload or {}

        doc_ids = list(payload_map.keys())
        scores: dict[int, float] = {}

        # Dense rank contribution (RRF)
        for rank, r in enumerate(dense_hits):
            scores[int(r.id)] = scores.get(int(r.id), 0.0) + 1.0 / (_RRF_K + rank + 1)

        # BM25 rank contribution
        tokenized = [_tokenize(payload_map[did].get("full_text") or "") for did in doc_ids]
        if tokenized and any(t for t in tokenized if t != ["_pad_"]):
            bm25 = BM25Okapi(tokenized)
            bm25_scores_arr = bm25.get_scores(_tokenize(query_text))
            bm25_ranked = sorted(
                zip(doc_ids, bm25_scores_arr), key=lambda x: x[1], reverse=True
            )[:fetch_n]
            for rank, (did, _) in enumerate(bm25_ranked):
                scores[did] = scores.get(did, 0.0) + 1.0 / (_RRF_K + rank + 1)

        exclude_set = set(exclude_ids or [])
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"id": did, "score": score, "payload": payload_map.get(did, {})}
            for did, score in fused
            if did not in exclude_set
        ][:top_k]

    async def count(self) -> int:
        """Return total number of points in the collection."""
        result = await self._c.count(collection_name=self._collection)
        return result.count

    async def _scroll_all(self) -> list[dict[str, Any]]:
        """Lấy toàn bộ documents trong collection (dùng cho BM25 index)."""
        points, _ = await self._c.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return [{"id": int(p.id), "payload": p.payload or {}} for p in points]


def _tokenize(text: str) -> list[str]:
    tokens = re.split(r"\s+", text.lower().strip())
    return tokens if tokens and tokens != [""] else ["_pad_"]


# ── Singleton ──────────────────────────────────────────────────────────────────

_store: VectorStore | None = None


def init_vector_store(url: str, api_key: str, collection: str, dim: int) -> VectorStore:
    global _store
    _store = VectorStore(url=url, api_key=api_key, collection=collection, vector_dim=dim)
    return _store


def get_vector_store() -> VectorStore:
    if _store is None:
        raise RuntimeError("VectorStore not initialized — init_vector_store() must be called first")
    return _store
