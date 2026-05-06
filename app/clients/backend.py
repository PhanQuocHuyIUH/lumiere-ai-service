from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.security.service_key import SERVICE_KEY_HEADER
from app.settings import settings


@dataclass(frozen=True)
class ExportPage:
    data: list[dict[str, Any]]
    page: int
    size: int
    total_elements: int
    total_pages: int
    has_next: bool


class BackendExportClient:
    def __init__(self) -> None:
        self.base_url = settings.backend_base_url.rstrip("/")
        self.service_key = settings.ai_service_key

    def is_configured(self) -> bool:
        return bool(self.base_url and self.service_key and self.service_key.strip())

    async def get_export_page(
        self,
        path: str,
        *,
        page: int = 0,
        size: int = 200,
        extra_params: dict[str, Any] | None = None,
        timeout_s: float = 1.2,
    ) -> ExportPage:
        if not self.is_configured():
            raise RuntimeError("Backend export client not configured")

        url = f"{self.base_url}{path}"
        params: dict[str, Any] = {"page": page, "size": size}
        if extra_params:
            params.update({k: v for k, v in extra_params.items() if v is not None})

        headers = {SERVICE_KEY_HEADER: self.service_key}
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            body = resp.json()

        # Expected shape: { success: true, message: "...", data: { data: [...], page:..., has_next:... } }
        payload = body.get("data") if isinstance(body, dict) else None
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected backend export response")

        data = payload.get("data")
        if not isinstance(data, list):
            data = []

        return ExportPage(
            data=[x for x in data if isinstance(x, dict)],
            page=int(payload.get("page") or 0),
            size=int(payload.get("size") or size),
            total_elements=int(payload.get("total_elements") or 0),
            total_pages=int(payload.get("total_pages") or 0),
            has_next=bool(payload.get("has_next")),
        )


async def crawl_all(
    client: BackendExportClient,
    path: str,
    *,
    extra_params: dict[str, Any] | None = None,
    max_pages: int = 20,
    page_size: int = 200,
    timeout_s: float = 1.2,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 0
    while page < max_pages:
        p = await client.get_export_page(
            path,
            page=page,
            size=page_size,
            extra_params=extra_params,
            timeout_s=timeout_s,
        )
        out.extend(p.data)
        if not p.has_next:
            break
        page += 1
    return out

