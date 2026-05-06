from __future__ import annotations

from app.schemas.ai import FeedbackRequest, FeedbackResponse
from app.services.ctr_store import record_events


async def process_feedback(req: FeedbackRequest) -> FeedbackResponse:
    events = [e.model_dump() for e in req.events]
    processed = await record_events(events)
    return FeedbackResponse(success=True, processed=processed)
