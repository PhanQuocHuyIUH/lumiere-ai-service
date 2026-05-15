from __future__ import annotations

import logging

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)

_COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"
_MAX_BATCH = 96  # Cohere v2 limit per request
_dim: int = 0   # auto-detected on first successful embed call


def get_dimension() -> int:
    if _dim == 0:
        raise RuntimeError("Embedding dimension not yet detected — embed_one() must be called first")
    return _dim


async def embed_texts(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Batch embed via Cohere v2 API. Splits automatically if len > 96."""
    if not texts:
        return []

    results: list[list[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = texts[i : i + _MAX_BATCH]
        results.extend(await _embed_batch(batch, input_type))
    return results


async def _embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
    global _dim
    payload = {
        "texts": texts,
        "model": settings.embed_model,
        "input_type": input_type,
        "embedding_types": ["float"],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _COHERE_EMBED_URL,
            json=payload,
            headers={"Authorization": f"Bearer {settings.cohere_api_key}"},
        )
        if not resp.is_success:
            logger.error("Cohere embedding error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        vectors: list[list[float]] = resp.json()["embeddings"]["float"]

    if _dim == 0 and vectors:
        _dim = len(vectors[0])
        logger.info("Embedding dimension auto-detected: %d", _dim)
    return vectors


async def embed_one(text: str, input_type: str = "search_document") -> list[float]:
    results = await embed_texts([text], input_type=input_type)
    return results[0]
