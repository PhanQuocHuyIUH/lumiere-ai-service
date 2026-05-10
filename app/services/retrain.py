from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone

import joblib
import lightgbm as lgb
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from app.clients.backend import BackendExportClient, crawl_all
from app.services.forecast import _FEATURE_COLS, _LGBM_PARAMS, _aggregate_daily, _build_features, _train
from app.services.job_store import update_job
from app.settings import settings

logger = logging.getLogger(__name__)


async def run_retrain(job_id: str) -> None:
    """Background worker: crawl all history, retrain models, save to disk."""
    await update_job(job_id, "PROCESSING")
    logger.info("Retrain job %s started", job_id)

    try:
        client = BackendExportClient()
        from_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

        order_items = await _crawl_safe(client, "/internal/ai/export/order-items", from_date, "order items")
        orders = await _crawl_safe(client, "/internal/ai/export/orders", from_date, "orders")
        payments = await _crawl_safe(client, "/internal/ai/export/payments", from_date, "payments")

        model_dir = settings.model_dir
        os.makedirs(model_dir, exist_ok=True)
        saved: list[str] = []

        if order_items:
            rules = _train_combo_rules(order_items)
            if rules is not None:
                path = os.path.join(model_dir, "combo_rules.pkl")
                with open(path, "wb") as f:
                    pickle.dump(rules, f)
                saved.append("combo_rules")
                logger.info("Saved %d combo rules → %s", len(rules), path)

        if orders:
            trained = _train_lgbm_model(orders, "orders")
            if trained is not None:
                model, residual_std = trained
                path = os.path.join(model_dir, "forecast_orders.joblib")
                joblib.dump({"model": model, "residual_std": residual_std}, path)
                saved.append("forecast_orders")
                logger.info("Saved orders forecast model → %s", path)

        if payments:
            trained = _train_lgbm_model(payments, "revenue")
            if trained is not None:
                model, residual_std = trained
                path = os.path.join(model_dir, "forecast_revenue.joblib")
                joblib.dump({"model": model, "residual_std": residual_std}, path)
                saved.append("forecast_revenue")
                logger.info("Saved revenue forecast model → %s", path)

        msg = f"Đã cập nhật: {', '.join(saved)}" if saved else "Không đủ dữ liệu để huấn luyện"
        await update_job(job_id, "COMPLETED", msg)
        logger.info("Retrain job %s completed: %s", job_id, msg)

    except Exception as exc:
        logger.error("Retrain job %s failed: %s", job_id, exc, exc_info=True)
        await update_job(job_id, "FAILED", str(exc))


async def _crawl_safe(client: BackendExportClient, path: str, from_date: str, label: str) -> list[dict]:
    try:
        records = await crawl_all(
            client, path,
            extra_params={"fromDate": from_date},
            max_pages=200, page_size=500, timeout_s=10.0,
        )
        logger.info("Retrain: crawled %d %s", len(records), label)
        return records
    except Exception as exc:
        logger.warning("Retrain: failed to crawl %s: %s", label, exc)
        return []


def _train_combo_rules(order_items: list[dict]) -> list[dict] | None:
    orders: dict[str, set[int]] = {}
    for item in order_items:
        order_id = str(item.get("orderId") or item.get("order_id") or "")
        raw_mid = item.get("menuItemId") or item.get("menu_item_id")
        if not order_id or raw_mid is None:
            continue
        try:
            orders.setdefault(order_id, set()).add(int(raw_mid))
        except (TypeError, ValueError):
            continue

    transactions = [frozenset(items) for items in orders.values() if len(items) >= 2]
    if len(transactions) < 5:
        logger.info("Retrain combo: only %d transactions — skipping", len(transactions))
        return None

    all_items = sorted({mid for tx in transactions for mid in tx})
    records = [{mid: (mid in tx) for mid in all_items} for tx in transactions]
    df = pd.DataFrame(records, columns=all_items)

    try:
        freq_itemsets = fpgrowth(df, min_support=0.03, use_colnames=True)
        if freq_itemsets.empty:
            return None
        rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
        rules = rules[rules["lift"] > 1.0].sort_values("lift", ascending=False)

        result: list[dict] = []
        seen: set[frozenset] = set()
        for _, row in rules.iterrows():
            combo_set = frozenset(row["antecedents"]) | frozenset(row["consequents"])
            if combo_set in seen:
                continue
            seen.add(combo_set)
            result.append({
                "combo_items": sorted(int(x) for x in combo_set),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4),
            })
            if len(result) >= 50:
                break
        return result or None
    except Exception as exc:
        logger.warning("Retrain combo FP-Growth failed: %s", exc)
        return None


def _train_lgbm_model(records: list[dict], metric: str) -> tuple[lgb.LGBMRegressor, float] | None:
    try:
        daily = _aggregate_daily(records, metric)
        if len(daily) < 14:
            logger.info("Retrain %s: only %d days — skipping", metric, len(daily))
            return None
        df = _build_features(daily)
        if len(df) < 7:
            return None
        model, residual_std = _train(df)
        return model, residual_std
    except Exception as exc:
        logger.warning("Retrain LightGBM (%s) failed: %s", metric, exc)
        return None
