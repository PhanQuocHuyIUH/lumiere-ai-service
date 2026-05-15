from __future__ import annotations

import datetime as dt
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.ai import router as ai_router
from app.errors import AiErrorBody
from app.middleware.rate_limit import enforce_rate_limit
from app.middleware.request_logging import log_request
from app.security.service_key import require_service_key
from app.settings import settings
from app.utils.trace import get_trace_id

logger = logging.getLogger("lumiere-ai-service")

# Module-level readiness flags (set during lifespan startup)
_vector_store_ready: bool = False
_embedder_ready: bool = False


def _error_code_for_http(status_code: int) -> str:
    return {
        400: "AI_VALIDATION_ERROR",
        401: "AI_UNAUTHORIZED",
        403: "AI_FORBIDDEN",
        404: "AI_NOT_FOUND",
        409: "AI_CONFLICT",
        429: "AI_RATE_LIMITED",
        503: "AI_UNAVAILABLE",
    }.get(status_code, "AI_HTTP_ERROR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store_ready, _embedder_ready

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("Starting Lumiere AI Service...")

    # ── Verify Gemini embedding API (auto-detects dimension) ──────────────────
    try:
        from app.services.embedder import embed_one, get_dimension
        key_check = str(settings.cohere_api_key)
        logger.info(f"RENDER DEBUG - Chiều dài API Key: {len(key_check)} ký tự. Bắt đầu bằng: {key_check[:4]}")
        
        await embed_one("ping")
        await embed_one("ping")  # sets _dim internally on first call
        dim = get_dimension()
        _embedder_ready = True
        logger.info("Embedder ready: model=%s  dim=%d", settings.embed_model, dim)
    except Exception as exc:
        _embedder_ready = False
        dim = 0  # vector store won't be created if dim=0
        logger.warning("Embedder API check failed (%s) — sync-menu and recommend may be unavailable", exc)

    # ── Connect to Qdrant vector store ─────────────────────────────────────────
    try:
        if dim == 0:
            raise RuntimeError("Embedding dimension unknown — skipping vector store init")
        from app.services.vector_store import init_vector_store
        store = init_vector_store(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            dim=dim,
        )
        await store.connect()
        _vector_store_ready = True
    except Exception as exc:
        _vector_store_ready = False
        logger.warning("VectorStore connection failed (%s) — sync-menu and recommend will be unavailable", exc)

    # ── Auto-seed vector store on first launch ─────────────────────────────────
    if _vector_store_ready and _embedder_ready:
        try:
            from app.services.vector_store import get_vector_store
            n = await get_vector_store().count()
            if n == 0:
                logger.info("Vector store is empty — triggering auto-seed from backend")
                from app.services.sync import sync_all_menu_items_from_backend
                import asyncio
                asyncio.create_task(sync_all_menu_items_from_backend())
            else:
                logger.info("Vector store has %d items — skipping auto-seed", n)
        except Exception as exc:
            logger.warning("Auto-seed check failed (%s) — run POST /ai/sync-menu/full manually", exc)

    # ── Connect to Redis ───────────────────────────────────────────────────────
    _redis_ready = False
    try:
        from app.services.redis_client import init_redis
        await init_redis(settings.redis_url)
        _redis_ready = True
    except Exception as exc:
        logger.warning("Redis connection failed (%s) — feedback and retrain unavailable", exc)

    # ── Ensure model directory exists ──────────────────────────────────────────
    try:
        os.makedirs(settings.model_dir, exist_ok=True)
        logger.info("Model directory ready: %s", settings.model_dir)
    except Exception as exc:
        logger.warning("Failed to create model directory: %s", exc)

    logger.info(
        "Service started: embedder=%s  vector_store=%s  redis=%s",
        "UP" if _embedder_ready else "DOWN",
        "UP" if _vector_store_ready else "DOWN",
        "UP" if _redis_ready else "DOWN",
    )

    yield

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    try:
        from app.services.redis_client import close_redis
        await close_redis()
    except Exception:
        pass
    try:
        from app.services.vector_store import get_vector_store
        await get_vector_store().close()
    except Exception:
        pass
    logger.info("Lumiere AI Service stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Lumiere AI Service", version="v2", lifespan=lifespan)
    # Attach module logger to the FastAPI app so middleware can use `request.app.logger`.
    app.logger = logger

    @app.middleware("http")
    async def _request_logging_mw(request: Request, call_next):
        return await log_request(request, call_next)

    @app.middleware("http")
    async def _rate_limit_mw(request: Request, call_next):
        enforce_rate_limit(request)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        body = AiErrorBody(
            success=False,
            code=_error_code_for_http(exc.status_code),
            message=str(exc.detail) if exc.detail else "Request failed",
            trace_id=get_trace_id(request),
        )
        return JSONResponse(status_code=exc.status_code, content=body.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = AiErrorBody(
            success=False,
            code="AI_VALIDATION_ERROR",
            message="Invalid request payload",
            trace_id=get_trace_id(request),
        )
        return JSONResponse(status_code=400, content=body.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        body = AiErrorBody(
            success=False,
            code="AI_INTERNAL_ERROR",
            message="Internal error",
            trace_id=get_trace_id(request),
        )
        return JSONResponse(status_code=500, content=body.to_dict())

    @app.get("/ai/health", dependencies=[Depends(require_service_key)])
    async def health() -> dict:
        now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        llm_ready = bool(settings.llm_api_key and settings.llm_model)
        return {
            "status": "UP",
            "version": "v2",
            "components": {
                "embedder": _embedder_ready,
                "vector_store": _vector_store_ready,
                "llm": llm_ready,
            },
            "endpoints": {
                "sync_menu": _embedder_ready and _vector_store_ready,
                "recommend": _vector_store_ready,
                "chatbot": llm_ready,
                "forecast": True,
                "combo_generate": True,
                "kitchen_batching": True,
            },
            "timestamp": now,
        }

    app.include_router(ai_router)
    return app


app = create_app()
