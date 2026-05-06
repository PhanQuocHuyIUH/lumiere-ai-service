from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.security.service_key import require_service_key

logger = logging.getLogger(__name__)
from app.schemas.ai import (
    ChatbotRequest,
    ChatbotResponse,
    ComboGenerateRequest,
    ComboGenerateResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForecastRequest,
    ForecastResponse,
    JobStatusResponse,
    KitchenBatchingRequest,
    KitchenBatchingResponse,
    RecommendRequest,
    RecommendResponse,
    RetrainResponse,
    SyncMenuRequest,
    SyncMenuResponse,
)
from app.services.batching import suggest_batches
from app.services.chatbot import chatbot_reply
from app.services.feedback import process_feedback
from app.services.forecast import forecast_metric
from app.services.recommend import recommend_items
from app.services.sync import delete_menu_item, sync_menu_item, sync_all_menu_items_from_backend

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(require_service_key)])


# ── Health Check ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=None)
async def health_check() -> dict:
    """Health check endpoint for AI Service."""
    return {
        "status": "UP",
        "service": "lumiere-ai-service",
        "timestamp": asyncio.get_event_loop().time()
    }


# ── Module 1 ───────────────────────────────────────────────────────────────────

@router.post("/sync-menu", response_model=SyncMenuResponse)
async def sync_menu(req: SyncMenuRequest) -> SyncMenuResponse:
    return await sync_menu_item(req)


@router.delete("/sync-menu/{item_id}", response_model=SyncMenuResponse)
async def delete_sync_menu(item_id: int) -> SyncMenuResponse:
    return await delete_menu_item(item_id)


@router.post("/sync-menu/full", response_model=None)
async def sync_menu_full(background_tasks: BackgroundTasks) -> dict:
    """Trigger a full sync from the main backend export into the vector store.

    This runs as a background task and returns immediately with a started message.
    """
    background_tasks.add_task(sync_all_menu_items_from_backend)
    return {"success": True, "message": "Full sync started in background"}


# ── Module 2 ───────────────────────────────────────────────────────────────────

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        return await asyncio.wait_for(recommend_items(req), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error("Recommend request timed out after 5s")
        return RecommendResponse(success=False, source="timeout", items=[], model_version=None)
    except Exception as exc:
        logger.error("Recommend request failed: %s", exc)
        return RecommendResponse(success=False, source="error", items=[], model_version=None)


@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(req: ChatbotRequest) -> ChatbotResponse:
    try:
        return await asyncio.wait_for(chatbot_reply(req), timeout=20.0)
    except asyncio.TimeoutError:
        logger.error("Chatbot request timed out after 20s: message=%s", req.message[:100])
        return ChatbotResponse(
            success=True,
            reply_text="Xin lỗi, hệ thống đang tải. Bạn có thể thử lại sau.",
            suggested_actions=[]
        )
    except Exception as exc:
        logger.error("Chatbot request failed: %s", exc)
        return ChatbotResponse(
            success=True,
            reply_text="Xin lỗi, hệ thống đang bận. Bạn có thể thử lại sau.",
            suggested_actions=[]
        )


# ── Module 3 ───────────────────────────────────────────────────────────────────

@router.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    return await forecast_metric(req)


@router.post("/combo-generate", response_model=ComboGenerateResponse)
async def combo_generate(req: ComboGenerateRequest) -> ComboGenerateResponse:
    from app.services.combo import generate_combos
    return await generate_combos(req)


@router.post("/kitchen-batching", response_model=KitchenBatchingResponse)
async def kitchen_batching(req: KitchenBatchingRequest) -> KitchenBatchingResponse:
    return await suggest_batches(req)


# ── Phase 2: Recommendation Feedback ──────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    return await process_feedback(req)


# ── Phase 3: Batch Retraining ──────────────────────────────────────────────────

@router.post("/retrain", response_model=RetrainResponse)
async def retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    from app.services.job_store import create_job
    from app.services.retrain import run_retrain

    job_id = str(uuid.uuid4())
    await create_job(job_id)
    background_tasks.add_task(run_retrain, job_id)
    return RetrainResponse(success=True, job_id=job_id, status="PENDING")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    from app.services.job_store import get_job

    data = await get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(
        success=True,
        job_id=data.get("job_id", job_id),
        status=data.get("status", "UNKNOWN"),
        message=data.get("message") or None,
    )
