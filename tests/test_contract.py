from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.main import create_app

_KEY = "test-key"
_HEADERS = {"X-AI-Service-Key": _KEY}


def _app():
    app = create_app()
    from app import settings as _s
    _s.settings.ai_service_key = _KEY  # type: ignore[misc]
    return app


# ── Auth guard ─────────────────────────────────────────────────────────────────

def test_health_requires_key() -> None:
    client = TestClient(_app())
    r = client.get("/ai/health")
    assert r.status_code in (401, 403)
    body = r.json()
    assert body["success"] is False
    assert "trace_id" in body


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_ok() -> None:
    client = TestClient(_app())
    r = client.get("/ai/health", headers=_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "UP"
    assert "components" in body
    assert "endpoints" in body


# ── POST /ai/sync-menu ─────────────────────────────────────────────────────────

def test_sync_menu_schema_validation() -> None:
    """sync-menu must reject a missing required field."""
    client = TestClient(_app())
    r = client.post("/ai/sync-menu", headers=_HEADERS, json={"name": "only-name"})
    assert r.status_code == 400


# ── POST /ai/recommend ─────────────────────────────────────────────────────────

def test_recommend_returns_success() -> None:
    client = TestClient(_app())
    payload = {"current_items": [1, 5], "top_k": 3}
    r = client.post("/ai/recommend", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "items" in body


def test_recommend_rejects_invalid_top_k() -> None:
    client = TestClient(_app())
    payload = {"current_items": [1], "top_k": 0}
    r = client.post("/ai/recommend", headers=_HEADERS, json=payload)
    assert r.status_code == 400


# ── POST /ai/chatbot ───────────────────────────────────────────────────────────

def test_chatbot_returns_success() -> None:
    client = TestClient(_app())
    payload = {"session_id": "s1", "message": "cho mình 2 risotto", "current_cart_item_ids": []}
    r = client.post("/ai/chatbot", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "reply_text" in body
    assert "suggested_actions" in body


def test_chatbot_returns_no_menu_reply_when_retrieval_is_empty(monkeypatch) -> None:
    from app.schemas.ai import ChatbotRequest
    import app.services.chatbot as chatbot_module

    class _EmptyStore:
        async def hybrid_search_rrf(self, **kwargs):
            return []

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LLM client should not be constructed when no menu items are retrieved")

    async def _fake_embed_one(message: str):
        return [0.0]

    monkeypatch.setattr(chatbot_module, "get_vector_store", lambda: _EmptyStore())
    monkeypatch.setattr(chatbot_module, "embed_one", _fake_embed_one)
    monkeypatch.setattr(chatbot_module, "OpenAICompatibleClient", _FailingClient)

    result = asyncio.run(
        chatbot_module.chatbot_reply(
            ChatbotRequest(session_id="s-empty", message="cho tôi xem menu", current_cart_item_ids=[])
        )
    )

    assert result.success is True
    assert result.reply_text.startswith("Chưa có thông tin menu")
    assert result.suggested_actions == []


# ── POST /ai/forecast ──────────────────────────────────────────────────────────

def test_forecast_returns_predictions() -> None:
    client = TestClient(_app())
    payload = {"metric": "orders", "horizon_days": 3}
    r = client.post("/ai/forecast", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    preds = body["predictions"]
    assert len(preds) == 3
    assert preds[0]["day"] == 1
    assert "lower_bound" in preds[0]
    assert "upper_bound" in preds[0]


def test_forecast_rejects_zero_horizon() -> None:
    client = TestClient(_app())
    r = client.post("/ai/forecast", headers=_HEADERS, json={"metric": "orders", "horizon_days": 0})
    assert r.status_code == 400


# ── POST /ai/combo-generate ────────────────────────────────────────────────────

def test_combo_generate_returns_success() -> None:
    client = TestClient(_app())
    payload = {"analyze_days": 30, "min_support": 0.05, "min_confidence": 0.6}
    r = client.post("/ai/combo-generate", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "draft_combos" in body


# ── POST /ai/kitchen-batching ──────────────────────────────────────────────────

def test_kitchen_batching_returns_success() -> None:
    client = TestClient(_app())
    payload = {
        "active_tasks": [
            {"task_id": 501, "menu_item_id": 2, "quantity": 1,
             "created_at": "2026-04-17T08:20:00Z", "cook_time_seconds": 900},
            {"task_id": 502, "menu_item_id": 2, "quantity": 1,
             "created_at": "2026-04-17T08:22:00Z", "cook_time_seconds": 900},
        ]
    }
    r = client.post("/ai/kitchen-batching", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "suggestions" in body


def test_kitchen_batching_no_batchable_tasks() -> None:
    """Single task per item → no suggestions."""
    client = TestClient(_app())
    payload = {
        "active_tasks": [
            {"task_id": 1, "menu_item_id": 10, "quantity": 1,
             "created_at": "2026-04-17T08:00:00Z", "cook_time_seconds": 600},
        ]
    }
    r = client.post("/ai/kitchen-batching", headers=_HEADERS, json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["suggestions"] == []
