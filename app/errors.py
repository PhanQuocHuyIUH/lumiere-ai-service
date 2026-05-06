from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AiErrorBody:
    success: bool
    code: str
    message: str
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "code": self.code,
            "message": self.message,
            "trace_id": self.trace_id,
        }

