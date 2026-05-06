from __future__ import annotations

import asyncio
import logging

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_CONCURRENCY = 5  # max concurrent embedding requests
_dim: int = 0  # auto-detected on first successful embed call


def get_dimension() -> int:
    if _dim == 0:
        raise RuntimeError("Embedding dimension not yet detected — embed_one() must be called first")
    return _dim


async def embed_one(text: str) -> list[float]:
    """Single embed via Gemini embedContent API. Auto-detects dimension on first call."""
    global _dim
    model = settings.embed_model
    url = f"{_BASE_URL}/models/{model}:embedContent"
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, params={"key": settings.llm_api_key})
        if not resp.is_success:
            logger.error("Embedding API error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        vector: list[float] = resp.json()["embedding"]["values"]
    if _dim == 0:
        _dim = len(vector)
        logger.info("Embedding dimension auto-detected: %d", _dim)
    return vector


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts with bounded concurrency."""
    if not texts:
        return []
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(t: str) -> list[float]:
        async with sem:
            return await embed_one(t)

    return await asyncio.gather(*[_bounded(t) for t in texts])
