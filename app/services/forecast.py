from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from app.clients.backend import BackendExportClient, crawl_all
from app.schemas.ai import ForecastPrediction, ForecastRequest, ForecastResponse

logger = logging.getLogger(__name__)

# Feature column order must be identical at train and predict time
_FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7",
    "roll_mean_7", "roll_std_7",
    "day_of_week", "is_weekend", "month",
]

_LGBM_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


# ── Public entry point ─────────────────────────────────────────────────────────

def _load_saved_model(metric: str) -> tuple[lgb.LGBMRegressor, float | None] | None:
    from app.settings import settings
    path = os.path.join(settings.model_dir, f"forecast_{metric}.joblib")
    if os.path.exists(path):
        try:
            loaded = joblib.load(path)
            # Backward compatibility: old artifact is bare LGBMRegressor.
            if isinstance(loaded, lgb.LGBMRegressor):
                return loaded, None
            # New artifact format: {"model": LGBMRegressor, "residual_std": float}
            if isinstance(loaded, dict):
                model = loaded.get("model")
                residual_std = loaded.get("residual_std")
                if isinstance(model, lgb.LGBMRegressor):
                    return model, float(residual_std) if residual_std is not None else None
            logger.warning("Unsupported saved forecast model artifact format for %s", metric)
        except Exception as exc:
            logger.warning("Failed to load saved forecast model for %s: %s", metric, exc)
    return None


def _resolve_residual_std(
    residual_std: float | None,
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
) -> float:
    if residual_std is not None and residual_std > 0:
        return float(residual_std)
    try:
        X = df[_FEATURE_COLS]
        y = df["value"].values
        y_hat = model.predict(X)
        return max(0.0, float(np.std(y - y_hat)))
    except Exception as exc:
        logger.warning("Failed to estimate residual std from saved model: %s", exc)
        return 0.0


async def forecast_metric(req: ForecastRequest) -> ForecastResponse:
    client = BackendExportClient()

    # Always pull enough recent history to seed the rolling prediction window
    history_days = max(90, req.horizon_days + 30)
    from_date = (datetime.now(timezone.utc) - timedelta(days=history_days)).strftime("%Y-%m-%d")

    path = "/internal/ai/export/orders" if req.metric == "orders" else "/internal/ai/export/payments"
    try:
        records = await crawl_all(
            client,
            path,
            extra_params={"fromDate": from_date},
            max_pages=100,
            page_size=200,
            timeout_s=5.0,
        )
    except Exception as exc:
        logger.warning("Backend export failed for forecast (%s) — fallback", exc)
        return _flat_fallback(req)

    daily = _aggregate_daily(records, req.metric)
    if len(daily) < 14:
        logger.info("Insufficient history (%d days) — fallback forecast", len(daily))
        return _flat_fallback(req, daily)

    df = _build_features(daily)
    if len(df) < 7:
        return _flat_fallback(req, daily)

    # Try pre-trained model first; fall back to on-demand training
    saved_artifact = _load_saved_model(req.metric)
    if saved_artifact is not None:
        logger.debug("Using pre-trained model for %s forecast", req.metric)
        saved_model, saved_residual_std = saved_artifact
        residual_std = _resolve_residual_std(saved_residual_std, saved_model, df)
        predictions = _predict_future(saved_model, df, req.horizon_days, residual_std)
        return ForecastResponse(success=True, metric=req.metric, predictions=predictions)

    try:
        model, residual_std = _train(df)
    except Exception as exc:
        logger.warning("LightGBM training failed (%s) — fallback", exc)
        return _flat_fallback(req, daily)

    predictions = _predict_future(model, df, req.horizon_days, residual_std)
    return ForecastResponse(success=True, metric=req.metric, predictions=predictions)


# ── Data preparation ───────────────────────────────────────────────────────────

def _aggregate_daily(records: list[dict], metric: str) -> pd.Series:
    rows = []
    for r in records:
        raw_date = (
            r.get("createdAt") or r.get("createdDate") or r.get("created_at") or ""
        )
        if not raw_date:
            continue
        try:
            dt = pd.to_datetime(raw_date).normalize()
        except Exception:
            continue

        if metric == "orders":
            value = 1.0
        else:
            amount = r.get("amount") or r.get("totalAmount") or r.get("total") or 0
            try:
                value = float(amount)
            except (TypeError, ValueError):
                value = 0.0

        rows.append({"date": dt, "value": value})

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    return df.groupby("date")["value"].sum()


def _build_features(daily: pd.Series) -> pd.DataFrame:
    df = daily.to_frame(name="value").sort_index()
    df.index = pd.to_datetime(df.index)

    # Fill gaps with zero (closed days / missing data)
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range, fill_value=0.0)

    # Lag features — shift(k) ensures no data leakage from future
    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["value"].shift(lag)

    # Rolling statistics computed from lagged data
    lagged = df["value"].shift(1)
    df["roll_mean_7"] = lagged.rolling(7).mean()
    df["roll_std_7"] = lagged.rolling(7).std().fillna(0.0)

    # Calendar categorical features
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month

    return df.dropna()


# ── Model training ─────────────────────────────────────────────────────────────

def _train(df: pd.DataFrame) -> tuple[lgb.LGBMRegressor, float]:
    X = df[_FEATURE_COLS].values
    y = df["value"].values

    model = lgb.LGBMRegressor(**_LGBM_PARAMS)
    model.fit(X, y)

    # Residual std on training set → used for prediction intervals
    y_hat = model.predict(X)
    residual_std = float(np.std(y - y_hat))
    return model, residual_std


# ── Recursive prediction ───────────────────────────────────────────────────────

def _predict_future(
    model: lgb.LGBMRegressor,
    history_df: pd.DataFrame,
    horizon_days: int,
    residual_std: float,
) -> list[ForecastPrediction]:
    # Seed rolling window with the last 7 known actual values
    window: list[float] = list(history_df["value"].values[-7:])
    last_date = history_df.index[-1]
    margin = residual_std * 1.5

    predictions: list[ForecastPrediction] = []
    for day in range(1, horizon_days + 1):
        next_date = last_date + timedelta(days=day)
        lags = list(reversed(window[-7:]))  # lag_1 = most recent, lag_7 = oldest

        roll_mean = float(np.mean(window[-7:]))
        roll_std = float(np.std(window[-7:]))

        x = [
            *lags,
            roll_mean,
            roll_std,
            next_date.dayofweek,
            int(next_date.dayofweek >= 5),
            next_date.month,
        ]

        x_df = pd.DataFrame([x], columns=_FEATURE_COLS)
        raw_value = float(model.predict(x_df)[0])
        value = max(0.0, raw_value)

        predictions.append(
            ForecastPrediction(
                day=day,
                value=round(value, 2),
                lower_bound=round(max(0.0, value - margin), 2),
                upper_bound=round(value + margin, 2),
            )
        )
        window.append(value)

    return predictions


# ── Fallback (insufficient data or errors) ────────────────────────────────────

def _flat_fallback(req: ForecastRequest, daily: pd.Series | None = None) -> ForecastResponse:
    if daily is not None and len(daily) >= 3:
        base = float(daily.iloc[-min(7, len(daily)):].mean())
    else:
        base = 50.0 if req.metric == "orders" else 5_000_000.0

    predictions = [
        ForecastPrediction(
            day=d,
            value=round(base, 2),
            lower_bound=round(base * 0.8, 2),
            upper_bound=round(base * 1.2, 2),
        )
        for d in range(1, req.horizon_days + 1)
    ]
    return ForecastResponse(success=True, metric=req.metric, predictions=predictions)
