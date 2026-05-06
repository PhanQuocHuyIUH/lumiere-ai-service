from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    code: str
    message: str
    trace_id: str

