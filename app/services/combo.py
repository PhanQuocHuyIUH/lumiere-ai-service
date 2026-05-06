from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from app.clients.backend import BackendExportClient, crawl_all
from app.schemas.ai import ComboGenerateRequest, ComboGenerateResponse, DraftCombo

logger = logging.getLogger(__name__)


def _load_saved_rules() -> list[dict] | None:
    from app.settings import settings
    path = os.path.join(settings.model_dir, "combo_rules.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load saved combo rules: %s", exc)
    return None


async def generate_combos(req: ComboGenerateRequest) -> ComboGenerateResponse:
    saved_rules = _load_saved_rules()
    if saved_rules is not None:
        filtered = [
            r for r in saved_rules
            if r.get("confidence", 0.0) >= req.min_confidence
        ][:20]
        logger.debug("Using %d pre-trained combo rules", len(filtered))
        return ComboGenerateResponse(
            success=True,
            draft_combos=[
                DraftCombo(
                    combo_items=r["combo_items"],
                    confidence_score=r["confidence"],
                    lift_score=r["lift"],
                )
                for r in filtered
            ],
        )

    client = BackendExportClient()

    # ── Step 1: Pull order-items for the analysis window ──────────────────────
    from_date = (datetime.now(timezone.utc) - timedelta(days=req.analyze_days)).strftime("%Y-%m-%d")
    try:
        raw = await crawl_all(
            client,
            "/internal/ai/export/order-items",
            extra_params={"fromDate": from_date},
            max_pages=100,
            page_size=200,
            timeout_s=5.0,
        )
    except Exception as exc:
        logger.warning("Backend export failed for combo (%s) — returning empty", exc)
        return ComboGenerateResponse(success=True, draft_combos=[])

    if not raw:
        return ComboGenerateResponse(success=True, draft_combos=[])

    # ── Step 2: Build transaction sets (order_id → set of menu_item_ids) ──────
    orders: dict[str, set[int]] = {}
    for item in raw:
        order_id = str(item.get("orderId") or item.get("order_id") or "")
        raw_mid = item.get("menuItemId") or item.get("menu_item_id")
        if not order_id or raw_mid is None:
            continue
        try:
            mid = int(raw_mid)
        except (TypeError, ValueError):
            continue
        orders.setdefault(order_id, set()).add(mid)

    # Keep only multi-item orders — single-item orders cannot produce rules
    transactions = [frozenset(items) for items in orders.values() if len(items) >= 2]
    if len(transactions) < 5:
        logger.info("Insufficient transactions (%d) for FP-Growth", len(transactions))
        return ComboGenerateResponse(success=True, draft_combos=[])

    all_items = sorted({mid for tx in transactions for mid in tx})

    # ── Step 3: One-hot encode → FP-Growth (HNSW-style tree scan, 2 DB passes) ─
    records = [{mid: (mid in tx) for mid in all_items} for tx in transactions]
    df = pd.DataFrame(records, columns=all_items)

    try:
        freq_itemsets = fpgrowth(df, min_support=req.min_support, use_colnames=True)
    except Exception as exc:
        logger.warning("FP-Growth failed (%s)", exc)
        return ComboGenerateResponse(success=True, draft_combos=[])

    if freq_itemsets.empty:
        return ComboGenerateResponse(success=True, draft_combos=[])

    # ── Step 4: Association rules — filter by confidence AND Lift > 1 ─────────
    try:
        rules = association_rules(
            freq_itemsets, metric="confidence", min_threshold=req.min_confidence
        )
    except Exception as exc:
        logger.warning("association_rules failed (%s)", exc)
        return ComboGenerateResponse(success=True, draft_combos=[])

    # Lift > 1 proves the items genuinely co-purchase; eliminates noise from best-sellers
    rules = rules[rules["lift"] > 1.0].sort_values("lift", ascending=False)

    # ── Step 5: Deduplicate and build response ─────────────────────────────────
    seen: set[frozenset] = set()
    draft_combos: list[DraftCombo] = []

    for _, row in rules.iterrows():
        combo_set = frozenset(row["antecedents"]) | frozenset(row["consequents"])
        if combo_set in seen:
            continue
        seen.add(combo_set)

        draft_combos.append(
            DraftCombo(
                combo_items=sorted(int(x) for x in combo_set),
                confidence_score=round(float(row["confidence"]), 4),
                lift_score=round(float(row["lift"]), 4),
            )
        )
        if len(draft_combos) >= 20:
            break

    logger.info("Combo generation: %d rules from %d transactions", len(draft_combos), len(transactions))
    return ComboGenerateResponse(success=True, draft_combos=draft_combos)
