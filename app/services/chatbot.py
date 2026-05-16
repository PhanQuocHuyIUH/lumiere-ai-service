from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.clients.llm import OpenAICompatibleClient
from app.schemas.ai import ChatbotRequest, ChatbotResponse
from app.services.embedder import embed_one
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ── Function Calling tool definitions ─────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Thêm một món ăn vào giỏ hàng của khách",
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_item_id": {
                        "type": "integer",
                        "description": "ID của món ăn từ danh sách menu",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Số lượng muốn đặt (1-99)",
                        "minimum": 1,
                        "maximum": 99,
                    },
                },
                "required": ["menu_item_id", "quantity"],
            },
        },
    }
]

_SYSTEM_TEMPLATE = """Bạn là trợ lý đặt món của nhà hàng Âu cao cấp Lumière.

[CHÂN LÝ HIỆN TẠI - MỚI NHẤT]
{context_block}

HƯỚNG DẪN HOẠT ĐỘNG:
1. NGUỒN DỮ LIỆU DUY NHẤT: Trả lời và tư vấn CHỈ dựa trên danh sách [CHÂN LÝ HIỆN TẠI] ở trên. Danh sách này cập nhật theo thời gian thực và CÓ GIÁ TRỊ CAO HƠN mọi lịch sử trò chuyện. Nếu lịch sử nói hết món nhưng danh sách hiện tại có, hãy vui vẻ báo khách là có món.
2. NẾU KHÔNG CÓ MÓN: Lịch sự báo nhà hàng không phục vụ. Tuyệt đối không tự suy diễn hoặc gợi ý các món không nằm trong danh sách trên.
3. LINH HOẠT TỪ VỰNG: Nếu khách hỏi từ khóa chung chung (ví dụ: "cơm", "bò"), hãy tự động mapping tới các món có nguyên liệu tương đương trong danh sách (như "Risotto", "Beefsteak", "Lasagna Bò").
4. GỌI HÀM ĐẶT MÓN: Khi khách xác nhận (ok, ừ, vâng) cho một món bạn vừa gợi ý ở lượt trước, hãy gọi `add_to_cart` với ID tương ứng. Không hiển thị ID món ăn ra text.
"""

_NO_MENU_REPLY = "Chưa có thông tin menu ạ. Quý khách vui lòng cho tôi biết món mình muốn tìm hoặc mô tả nhu cầu cụ thể hơn."

_ACK_TOKENS: set[str] = {
    "ok", "okay", "okela", "oke", "okê", "okie", "k",
    "ừ", "ừm", "uh", "uhm", "ờ", "ờm",
    "vâng", "dạ", "dạ vâng", "dạ được",
    "có", "đúng", "phải", "yep", "yes", "y",
    "đồng ý", "ổn", "được", "được ạ", "ok ạ",
    "rồi", "rồi ạ",
}


def _is_ack_message(msg: str) -> bool:
    """Return True if msg is a short acknowledgment that shouldn't drive a fresh RAG search."""
    cleaned = msg.strip().lower().rstrip(".!?,;:")
    return cleaned in _ACK_TOKENS


def _build_context_block(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Hiện không có thông tin menu. Hãy tư vấn dựa trên mô tả của khách."
    lines = ["Danh sách món ăn phù hợp:"]
    for it in items:
        p = it.get("payload", {})
        tags = ", ".join(p.get("tags") or [])
        lines.append(
            f"  ID: {it['id']} | {p.get('name', '?')} | "
            f"Giá: {p.get('price', 0):,.0f}đ | {p.get('description', '')} | Đặc tính: {tags}"
        )
    return "\n".join(lines)


def _build_system_prompt(context_block: str) -> str:
    return _SYSTEM_TEMPLATE.format(context_block=context_block)


def _parse_suggested_actions(tool_calls: list[dict]) -> list[str]:
    actions: list[str] = []
    for tc in tool_calls:
        if tc.get("name") == "add_to_cart":
            args = tc.get("arguments", {})
            mid = args.get("menu_item_id")
            qty = int(args.get("quantity") or 1)
            if mid is not None:
                actions.append(f"ADD_ITEM:{mid}:{qty}")
    return actions


# ── Main entry point ───────────────────────────────────────────────────────────

async def chatbot_reply(req: ChatbotRequest) -> ChatbotResponse:
    # ── Step 1: RAG — Hybrid Search (vector + BM25 + RRF) ─────────────────────
    # Skip retrieval for pure acknowledgments ("ok", "ừ", ...): a fresh vector
    # search on these tokens returns irrelevant items and tempts the LLM to
    # add_to_cart something the customer never asked for. Let the LLM rely on
    # conversation history instead.
    is_ack = _is_ack_message(req.message)

    retrieved: list[dict] = []
    if is_ack:
        logger.info("ACK message detected, skipping RAG search: %s", req.message[:50])
    else:
        try:
            store = get_vector_store()
            query_vector = await embed_one(req.message, input_type="search_query")
            retrieved = await store.hybrid_search_rrf(
                query_text=req.message,
                query_vector=query_vector,
                top_k=5,
            )
            logger.info("RAG search returned %d results for message: %s", len(retrieved), req.message[:50])
        except Exception as exc:
            logger.warning("RAG retrieval failed (%s) — proceeding without context", exc, exc_info=True)

    if not retrieved and not is_ack:
        return ChatbotResponse(
            success=True,
            reply_text=_NO_MENU_REPLY,
            suggested_actions=[],
        )

    if is_ack:
        context_block = (
            "Khách vừa trả lời xác nhận ngắn (ví dụ: 'ok', 'ừ', 'vâng'). "
            "KHÔNG có danh sách món mới. Hãy dựa vào lịch sử hội thoại: "
            "nếu lượt trước bạn đã đề xuất ĐẶT một món cụ thể, gọi add_to_cart với món đó; "
            "ngược lại, hỏi lại khách muốn đặt món nào."
        )
    else:
        context_block = _build_context_block(retrieved)

    # ── Step 2: Function Calling via LLM ──────────────────────────────────────
    client = OpenAICompatibleClient()
    if not client.is_configured():
        logger.error("LLM client not configured: base_url=%s, model=%s, provider=%s",
                    client.base_url or "NONE",
                    client.model or "NONE",
                    client.provider or "NONE")
        return ChatbotResponse(
            success=True,
            reply_text="Xin lỗi, hệ thống chưa được cấu hình. Liên hệ quản trị viên.",
            suggested_actions=[],
        )

    history = req.conversation_history or []
    history_dicts = [{"role": t.role, "content": t.content} for t in history]

    try:
        logger.info("Calling LLM chat_tools: message=%s history_turns=%d",
                    req.message[:100], len(history))
        result = await client.chat_tools(
            system_prompt=_build_system_prompt(context_block),
            user_prompt=req.message,
            tools=_TOOLS,
            timeout_s=15.0,
            history=history_dicts,
        )
        logger.info("LLM chat_tools succeeded: tool_calls=%d", len(result.tool_calls))
    except asyncio.TimeoutError:
        logger.error("LLM chat_tools TIMEOUT (>15s): message=%s, session=%s",
                    req.message[:100], req.session_id)
        return ChatbotResponse(
            success=True,
            reply_text="Xin lỗi, hệ thống đang bận. Bạn có thể thử lại sau.",
            suggested_actions=[],
        )
    except Exception as exc:
        logger.error("LLM call failed: type=%s  message=%s  session=%s  error=%s",
                    type(exc).__name__, req.message[:50], req.session_id, str(exc)[:200],
                    exc_info=True)
        return ChatbotResponse(
            success=True,
            reply_text="Xin lỗi, hệ thống đang bận. Bạn có thể thử lại sau.",
            suggested_actions=[],
        )

    # ── Step 3: Map tool calls → suggested_actions ────────────────────────────
    suggested_actions = _parse_suggested_actions(result.tool_calls)
    reply_text = result.reply_text or "Dạ em ghi nhận ạ!"

    logger.info("Chatbot response ready: session=%s  actions=%d", req.session_id, len(suggested_actions))
    return ChatbotResponse(
        success=True,
        reply_text=reply_text,
        suggested_actions=suggested_actions,
    )
