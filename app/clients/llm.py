from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmResult:
    raw_text: str
    json_obj: dict[str, Any] | None


@dataclass(frozen=True)
class LlmToolsResult:
    reply_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class OpenAICompatibleClient:
    """LLM client hỗ trợ hai chế độ:

    - **Google Native** (default khi base_url chứa ``generativelanguage.googleapis.com``):
      Gọi trực tiếp ``/v1beta/models/{model}:generateContent`` bằng API key header
      ``x-goog-api-key``.  Function Calling và JSON mode dùng định dạng native của Gemini.
    - **OpenAI-compatible**: gọi ``{base_url}/chat/completions`` với Bearer token.

    Lưu ý: không còn per-request model-list discovery.  Model được dùng thẳng từ
    ``LLM_MODEL`` trong ``.env``.  Nếu muốn đổi model, sửa biến đó rồi restart service.
    """

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model.strip()
        self.provider = settings.llm_provider.strip().lower()

    # ── Routing helpers ────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def _is_google_native(self) -> bool:
        """True khi base_url trỏ tới Google AI Studio hoặc provider được khai báo rõ."""
        host = urlparse(self.base_url).netloc.lower()
        return (
            self.provider in {"google", "google_generative", "gemini", "gemini_native"}
            or "generativelanguage.googleapis.com" in host
        )

    def _google_model_id(self) -> str:
        """Trả về model id sạch (không prefix ``models/``)."""
        return self.model.removeprefix("models/")

    def _google_generate_url(self) -> str:
        """URL endpoint generateContent của Google AI Studio (Native API)."""
        return (
            f"https://generativelanguage.googleapis.com"
            f"/v1beta/models/{self._google_model_id()}:generateContent"
        )

    def _chat_completions_url(self) -> str:
        """URL endpoint cho OpenAI-compatible providers.

        Handles three cases:
        - base_url ends with /v1   (e.g. https://api.groq.com/openai/v1)     → append /chat/completions
        - base_url ends with /openai (e.g. Gemini OpenAI-compat endpoint)     → append /chat/completions
        - base_url is root          (e.g. https://api.openai.com)              → append /v1/chat/completions
        """
        path = urlparse(self.base_url).path.lower().rstrip("/")
        if path.endswith("/v1") or path.endswith("/openai"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    # ── Auth headers ──────────────────────────────────────────────────────────

    def _google_native_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    def _openai_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    # ── Response parsers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_openai_message(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        message = data.get("choices", [{}])[0].get("message", {})
        reply_text: str = message.get("content") or ""

        parsed_calls: list[dict[str, Any]] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            if name and isinstance(args, dict):
                parsed_calls.append({"name": name, "arguments": args})

        return reply_text, parsed_calls

    @staticmethod
    def _parse_google_native_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        candidates = data.get("candidates") or []
        if not candidates:
            return "", []

        content = candidates[0].get("content", {}) or {}
        parts = content.get("parts") or []
        reply_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for part in parts:
            if "text" in part and part["text"]:
                reply_chunks.append(str(part["text"]))
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = function_call.get("name", "")
                args = function_call.get("args", {})
                if name and isinstance(args, dict):
                    tool_calls.append({"name": name, "arguments": args})

        return "".join(reply_chunks), tool_calls

    @staticmethod
    def _build_google_native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        function_declarations: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function", {})
            if fn.get("name"):
                function_declarations.append(
                    {
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
        return [{"functionDeclarations": function_declarations}] if function_declarations else []

    # ── HTTP ──────────────────────────────────────────────────────────────────

    @staticmethod
    async def _post_json_with_retry(
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_s: float,
        retries: int = 0,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(url, headers=headers, json=payload)

            try:
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code

                # Extract quota info from response headers
                quota_info = {
                    k: resp.headers[k]
                    for k in resp.headers
                    if "quota" in k.lower() or "ratelimit" in k.lower()
                }

                # Try to get error details from response body
                error_detail = ""
                try:
                    error_data = resp.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except Exception:
                    error_detail = resp.text[:200] if resp.text else ""

                if status == 429:
                    logger.warning(
                        "LLM API rate limited (429) - attempt %d/%d | quota_headers=%s | error=%s",
                        attempt + 1, retries + 1, quota_info, error_detail,
                    )
                elif status == 503:
                    logger.warning(
                        "LLM API unavailable (503) - attempt %d/%d | error=%s",
                        attempt + 1, retries + 1, error_detail,
                    )
                else:
                    logger.error(
                        "LLM API error %d - attempt %d/%d | error=%s | quota_info=%s",
                        status, attempt + 1, retries + 1, error_detail, quota_info,
                    )

                if status in {429, 503} and attempt < retries:
                    wait_time = 0.25 * (attempt + 1)
                    await asyncio.sleep(wait_time)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without a response")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def chat_json(self, *, system_prompt: str, user_prompt: str, timeout_s: float) -> LlmResult:
        """Gọi LLM, yêu cầu trả về JSON object."""
        if not self.is_configured():
            raise RuntimeError("LLM is not configured")

        if self._is_google_native():
            url = f"{self._google_generate_url()}?{urlencode({'key': self.api_key})}"
            payload: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            }
            headers = self._google_native_headers()
            retries_count = 3
        else:
            url = self._chat_completions_url()
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            headers = self._openai_headers()
            retries_count = 0

        logger.info(
            "Calling LLM chat_json: model=%s  provider=%s  native=%s  retries=%d  timeout=%.1fs",
            self._google_model_id() if self._is_google_native() else self.model,
            self.provider,
            self._is_google_native(),
            retries_count,
            timeout_s,
        )

        data = await self._post_json_with_retry(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            retries=retries_count - 1 if retries_count > 0 else 0,
        )

        if self._is_google_native():
            content, _ = self._parse_google_native_response(data)
        else:
            content, _ = self._parse_openai_message(data)

        if not isinstance(content, str):
            content = str(content)

        try:
            obj = json.loads(content)
        except Exception:
            obj = None

        return LlmResult(raw_text=content, json_obj=obj if isinstance(obj, dict) else None)

    async def chat_tools(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        timeout_s: float,
    ) -> LlmToolsResult:
        """Gọi LLM với Function Calling.

        Hỗ trợ Google AI Studio (Native Gemini format) và OpenAI-compatible APIs.
        """
        if not self.is_configured():
            raise RuntimeError("LLM is not configured")

        if self._is_google_native():
            url = f"{self._google_generate_url()}?{urlencode({'key': self.api_key})}"
            payload: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0},
                "tools": self._build_google_native_tools(tools),
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            }
            headers = self._google_native_headers()
            retries_count = 3
        else:
            url = self._chat_completions_url()
            payload = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "tools": tools,
                "tool_choice": "auto",
            }
            headers = self._openai_headers()
            retries_count = 0

        logger.info(
            "Calling LLM chat_tools: model=%s  provider=%s  native=%s  retries=%d  timeout=%.1fs",
            self._google_model_id() if self._is_google_native() else self.model,
            self.provider,
            self._is_google_native(),
            retries_count,
            timeout_s,
        )

        data = await self._post_json_with_retry(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            retries=retries_count - 1 if retries_count > 0 else 0,
        )

        if self._is_google_native():
            reply_text, parsed_calls = self._parse_google_native_response(data)
        else:
            reply_text, parsed_calls = self._parse_openai_message(data)

        return LlmToolsResult(reply_text=reply_text, tool_calls=parsed_calls)
