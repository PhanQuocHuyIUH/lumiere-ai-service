from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Module 1: Foundation & Data Sync ──────────────────────────────────────────

class SyncMenuRequest(BaseModel):
    menu_item_id: int
    name: str
    description: str | None = None
    price: float
    category: str | None = None
    tags: list[str] = Field(default_factory=list)


class SyncMenuResponse(BaseModel):
    success: bool = True
    menu_item_id: int
    vector_id: str


class SyncMenuBulkResponse(BaseModel):
    success: bool = True
    synced_count: int = 0


# ── Module 2: Customer-Facing ──────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    current_items: list[int]
    top_k: int = Field(ge=1, le=50)


class RecommendItem(BaseModel):
    menu_item_id: int
    score: float
    reason: str


class RecommendResponse(BaseModel):
    success: bool = True
    source: Literal["model", "fallback", "timeout", "error"]
    items: list[RecommendItem]
    model_version: str | None = None


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatbotRequest(BaseModel):
    session_id: str
    message: str
    current_cart_item_ids: list[int] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class ChatbotResponse(BaseModel):
    success: bool = True
    reply_text: str
    suggested_actions: list[str] = Field(default_factory=list)


# ── Module 3: Back-Office ──────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    metric: Literal["orders", "revenue"]
    # Capped at 30 days: recursive prediction with 7-day lag features compounds
    # error rapidly past ~2-4 weeks; longer horizons produce flat-line noise.
    horizon_days: int = Field(ge=1, le=30)


class ForecastPrediction(BaseModel):
    day: int
    value: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    success: bool = True
    metric: Literal["orders", "revenue"]
    predictions: list[ForecastPrediction]


class ComboGenerateRequest(BaseModel):
    analyze_days: int = Field(default=30, ge=1, le=365)
    min_support: float = Field(default=0.05, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class DraftCombo(BaseModel):
    combo_items: list[int]
    confidence_score: float
    lift_score: float


class ComboGenerateResponse(BaseModel):
    success: bool = True
    draft_combos: list[DraftCombo]


class KitchenTaskIn(BaseModel):
    task_id: int
    menu_item_id: int
    quantity: int
    created_at: str
    cook_time_seconds: int = Field(default=600)


class KitchenBatchingRequest(BaseModel):
    active_tasks: list[KitchenTaskIn]


class KitchenBatchSuggestion(BaseModel):
    menu_item_id: int
    task_ids: list[int]
    estimated_saving_minutes: float
    reason: str


class KitchenBatchingResponse(BaseModel):
    success: bool = True
    suggestions: list[KitchenBatchSuggestion]


# ── Phase 2: Recommendation Feedback ──────────────────────────────────────────

class FeedbackEvent(BaseModel):
    type: str
    shown: list[int] = Field(default_factory=list)
    clicked: list[int] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    events: list[FeedbackEvent]


class FeedbackResponse(BaseModel):
    success: bool = True
    processed: int


# ── Phase 3: Batch Retraining ──────────────────────────────────────────────────

class RetrainResponse(BaseModel):
    success: bool = True
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    success: bool = True
    job_id: str
    status: str
    message: str | None = None
