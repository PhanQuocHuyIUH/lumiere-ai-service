from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from app.clients.backend import BackendExportClient, crawl_all
from app.schemas.ai import ComboGenerateRequest, ComboGenerateResponse, DraftCombo
from app.services.model_cache import load_cached

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
# Min support floor: anything below this is too noisy for combo recommendation
# (a "combo" supported by <2% of visits is statistical noise for a restaurant).
_MIN_SUPPORT_FLOOR = 0.02
# Bill-level grouping: orders for the same tableId within VISIT_WINDOW_SECONDS
# are treated as one "visit" (1 group of customers eating together) — handles
# the common case where customers add items later in the meal.
_VISIT_WINDOW_SECONDS = 4 * 3600  # 4 hours


def _read_pickle(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_saved_rules() -> list[dict] | None:
    from app.settings import settings
    path = os.path.join(settings.model_dir, "combo_rules.pkl")
    return load_cached(path, _read_pickle)


def build_visit_transactions(raw: list[dict]) -> list[frozenset[int]]:
    """Group order items into bill-level transactions.

    A "visit" = all items ordered at the same tableId within a 4-hour window.
    Falls back to grouping by orderId when tableId or createdAt is missing.
    Returns frozensets of menu_item_id with size >= 2 (rule mining needs pairs).
    """
    visits: dict[tuple, set[int]] = {}
    fallback: dict[str, set[int]] = {}

    for item in raw:
        raw_mid = item.get("menuItemId") or item.get("menu_item_id")
        if raw_mid is None:
            continue
        try:
            mid = int(raw_mid)
        except (TypeError, ValueError):
            continue

        table_id = item.get("tableId") or item.get("table_id")
        created_at = item.get("createdAt") or item.get("created_at")

        if table_id and created_at:
            try:
                ts = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                bucket = int(ts.timestamp()) // _VISIT_WINDOW_SECONDS
                key = (int(table_id), bucket)
                visits.setdefault(key, set()).add(mid)
                continue
            except (TypeError, ValueError):
                pass

        # Fallback to orderId grouping when bill-level fields are missing
        order_id = str(item.get("orderId") or item.get("order_id") or "")
        if order_id:
            fallback.setdefault(order_id, set()).add(mid)

    all_sets = list(visits.values()) + list(fallback.values())
    return [frozenset(items) for items in all_sets if len(items) >= 2]


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

    # ── Step 2: Bill-level transactions (tableId + 4h bucket) ────────────────
    transactions = build_visit_transactions(raw)
    if len(transactions) < 5:
        logger.info("Insufficient transactions (%d) for FP-Growth", len(transactions))
        return ComboGenerateResponse(success=True, draft_combos=[])

    all_items = sorted({mid for tx in transactions for mid in tx})

    # ── Step 3: One-hot encode → FP-Growth ────────────────────────────────────
    records = [{mid: (mid in tx) for mid in all_items} for tx in transactions]
    df = pd.DataFrame(records, columns=all_items)

    # Effective min_support = max(request, floor). Floor prevents noise rules.
    support = max(float(req.min_support), _MIN_SUPPORT_FLOOR)
    n_tx = len(transactions)
    support_threshold_count = max(1, int(support * n_tx))
    logger.info(
        "Combo generation: transactions=%d unique_items=%d min_support=%.4f (count>=%d, floor=%.4f)",
        n_tx, len(all_items), support, support_threshold_count, _MIN_SUPPORT_FLOOR,
    )

    try:
        freq_itemsets = fpgrowth(df, min_support=support, use_colnames=True)
        if freq_itemsets.empty or not any(len(x) >= 2 for x in freq_itemsets["itemsets"]):
            logger.info("FP-Growth found no pair itemsets at support=%.4f", support)
            return ComboGenerateResponse(success=True, draft_combos=[])
    except Exception as exc:
        logger.warning("FP-Growth failed (%s)", exc)
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
