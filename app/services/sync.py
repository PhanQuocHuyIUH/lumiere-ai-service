from __future__ import annotations

import logging

from app.schemas.ai import SyncMenuRequest, SyncMenuResponse
from app.services.embedder import embed_one
from app.services.vector_store import get_vector_store
from app.clients.backend import BackendExportClient, crawl_all

from typing import Any

logger = logging.getLogger(__name__)


async def sync_menu_item(req: SyncMenuRequest) -> SyncMenuResponse:
    # Metadata Chunking: concatenate all semantic fields into one rich string
    # so the embedding captures the full context of the menu item.
    tags_str = ", ".join(req.tags) if req.tags else ""
    full_text = (
        f"Tên: {req.name}, "
        f"Giá: {req.price}, "
        f"Mô tả: {req.description or ''}, "
        f"Danh mục: {req.category or ''}, "
        f"Đặc tính: {tags_str}"
    )

    vector = await embed_one(full_text)

    payload = {
        "menu_item_id": req.menu_item_id,
        "name": req.name,
        "price": float(req.price),
        "description": req.description or "",
        "category": req.category or "",
        "tags": req.tags,
        "full_text": full_text,
    }

    await get_vector_store().upsert(req.menu_item_id, vector, payload)

    return SyncMenuResponse(
        success=True,
        menu_item_id=req.menu_item_id,
        vector_id=f"vec_{req.menu_item_id}",
    )


async def delete_menu_item(item_id: int) -> SyncMenuResponse:
    try:
        await get_vector_store().delete(item_id)
    except RuntimeError as exc:
        logger.warning("VectorStore unavailable during delete (%s)", exc)
    except Exception as exc:
        logger.warning("Failed to delete vector for item %d: %s", item_id, exc)
    return SyncMenuResponse(success=True, menu_item_id=item_id, vector_id=f"vec_{item_id}")


async def sync_all_menu_items_from_backend() -> dict[str, Any]:
    """Crawl the main backend export endpoint and upsert all menu items into the vector store.

    Returns a dict with synced_count.
    """
    client = BackendExportClient()
    if not client.is_configured():
        logger.warning("Backend export client not configured — cannot sync from backend")
        return {"success": False, "synced_count": 0}

    try:
        raw = await crawl_all(client, "/internal/ai/export/menu-items", max_pages=500, page_size=200, timeout_s=5.0)
    except Exception as exc:
        logger.warning("Backend export failed for menu-items (%s)", exc)
        return {"success": False, "synced_count": 0}

    if not raw:
        logger.info("No menu items returned from backend export")
        return {"success": True, "synced_count": 0}

    synced = 0
    for item in raw:
        # Tolerant mapping of common field names from backend
        raw_mid = item.get("menuItemId") or item.get("menu_item_id") or item.get("id") or item.get("menuId")
        if raw_mid is None:
            continue
        try:
            mid = int(raw_mid)
        except (TypeError, ValueError):
            continue

        req_obj = {
            "menu_item_id": mid,
            "name": item.get("name") or item.get("title") or "",
            "price": float(item.get("price") or 0),
            "description": item.get("description") or item.get("desc") or "",
            "category": item.get("category") or None,
            "tags": item.get("tags") or [],
        }

        try:
            req = SyncMenuRequest.parse_obj(req_obj)
            await sync_menu_item(req)
            synced += 1
        except Exception as exc:
            logger.warning("Failed to sync menu item %s: %s", mid, exc)

    logger.info("Synced %d menu items from backend", synced)
    return {"success": True, "synced_count": synced}
