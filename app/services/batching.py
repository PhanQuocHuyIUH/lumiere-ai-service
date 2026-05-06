from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from ortools.sat.python import cp_model

from app.schemas.ai import (
    KitchenBatchSuggestion,
    KitchenBatchingRequest,
    KitchenBatchingResponse,
    KitchenTaskIn,
)

logger = logging.getLogger(__name__)

# ── Objective function weights ─────────────────────────────────────────────────
# Score = α × T_wait  +  β × N_items  −  γ × P_penalty
# T_wait:    wait time (minutes) of the oldest task in the candidate batch
# N_items:   number of tasks to be merged into one cook cycle
# P_penalty: items exceeding station capacity (prevents overloading one station)
_ALPHA = 2           # points per minute of wait
_BETA = 5            # points per extra task batched
_GAMMA = 15          # penalty per item over capacity
_STATION_CAPACITY = 5    # max tasks per batch before penalty kicks in
_MAX_WAIT_CAP_MIN = 60   # cap T_wait to avoid extreme dominance
_SOLVER_TIMEOUT_S = 0.5  # OR-Tools time budget


async def suggest_batches(req: KitchenBatchingRequest) -> KitchenBatchingResponse:
    if not req.active_tasks:
        return KitchenBatchingResponse(success=True, suggestions=[])

    # ── Group tasks by menu_item_id ────────────────────────────────────────────
    groups: dict[int, list[KitchenTaskIn]] = defaultdict(list)
    for task in req.active_tasks:
        groups[task.menu_item_id].append(task)

    batchable = {mid: tasks for mid, tasks in groups.items() if len(tasks) >= 2}
    if not batchable:
        return KitchenBatchingResponse(success=True, suggestions=[])

    # ── Build and solve CP model ───────────────────────────────────────────────
    now_ts = int(datetime.now(timezone.utc).timestamp())
    selected = _solve(batchable, now_ts)

    # ── Build output suggestions ───────────────────────────────────────────────
    suggestions: list[KitchenBatchSuggestion] = []
    for mid, tasks in selected:
        avg_cook_s = sum(t.cook_time_seconds for t in tasks) / len(tasks)
        saving_min = round(min(15.0, (len(tasks) - 1) * avg_cook_s / 60), 1)

        suggestions.append(
            KitchenBatchSuggestion(
                menu_item_id=mid,
                task_ids=[t.task_id for t in tasks],
                estimated_saving_minutes=saving_min,
                reason="Same item high queue overlap",
            )
        )

    return KitchenBatchingResponse(success=True, suggestions=suggestions)


# ── CP-SAT solver ──────────────────────────────────────────────────────────────

def _solve(
    batchable: dict[int, list[KitchenTaskIn]], now_ts: int
) -> list[tuple[int, list[KitchenTaskIn]]]:
    """
    Maximise  Σ (α·T_wait + β·N_items − γ·P_penalty) · x_i
    subject to  Σ x_i  ≤  STATION_CAPACITY   (concurrent batch limit)
    where x_i ∈ {0, 1} for each batchable group i.
    """
    model = cp_model.CpModel()

    entries: list[tuple[int, list[KitchenTaskIn], object, int]] = []
    for mid, tasks in batchable.items():
        score = _score(tasks, now_ts)
        if score <= 0:
            continue
        var = model.NewBoolVar(f"b_{mid}")
        entries.append((mid, tasks, var, score))

    if not entries:
        return []

    # Capacity constraint: limit simultaneous recommendations
    model.Add(sum(e[2] for e in entries) <= _STATION_CAPACITY)

    # Objective
    model.Maximize(sum(e[3] * e[2] for e in entries))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _SOLVER_TIMEOUT_S
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("OR-Tools returned status %s — no batching suggestions", status)
        return []

    return [(mid, tasks) for mid, tasks, var, _ in entries if solver.Value(var) == 1]


def _score(tasks: list[KitchenTaskIn], now_ts: int) -> int:
    n = len(tasks)
    total_qty = sum(t.quantity for t in tasks)

    oldest_ts = min(_parse_ts(t.created_at) for t in tasks)
    wait_min = min(int((now_ts - oldest_ts) / 60), _MAX_WAIT_CAP_MIN)
    wait_min = max(0, wait_min)

    p_penalty = max(0, total_qty - _STATION_CAPACITY)

    return int(_ALPHA * wait_min + _BETA * n - _GAMMA * p_penalty)


def _parse_ts(ts: str) -> int:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())
