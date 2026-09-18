"""Common service helpers for API v1."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from fastapi.responses import JSONResponse

try:
    from ...observability import new_run_id
except ImportError:  # pragma: no cover
    from observability import new_run_id


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def ok(data: Any, *, request_id: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ok",
        "request_id": request_id or new_run_id(),
        "data": data,
    }
    if meta is not None:
        payload["meta"] = meta
    return payload


def accepted(data: Any, *, request_id: str | None = None) -> dict[str, Any]:
    return {
        "status": "accepted",
        "request_id": request_id or new_run_id(),
        "data": data,
    }


def error_envelope(
    code: str,
    message: str,
    *,
    status_code: int = 500,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    rid = request_id or new_run_id()
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "request_id": rid,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": details or {},
            },
        },
        headers={"X-Request-ID": rid},
    )


def map_exception(exc: Exception, *, request_id: str | None = None) -> JSONResponse:
    message = str(exc) or type(exc).__name__
    lowered = message.lower()
    if isinstance(exc, ValueError):
        if (
            "knowledge node not found" in lowered
            or "trashed knowledge node not found" in lowered
            or "knowledge scope not found" in lowered
        ):
            return error_envelope("knowledge_node_not_found", message, status_code=404, request_id=request_id)
        if "topic" in lowered and "not found" in lowered:
            return error_envelope("topic_not_found", message, status_code=404, request_id=request_id)
        if "document" in lowered and "not found" in lowered:
            return error_envelope("document_not_found", message, status_code=404, request_id=request_id)
        return error_envelope("validation_error", message, status_code=400, request_id=request_id)
    if "rate" in lowered and "limit" in lowered:
        return error_envelope("upstream_rate_limited", message, status_code=429, retryable=True, request_id=request_id)
    if "auth" in lowered or "api key" in lowered:
        return error_envelope("upstream_auth_error", message, status_code=401, request_id=request_id)
    if "unavailable" in lowered or "connection" in lowered or "timeout" in lowered:
        return error_envelope("upstream_unavailable", message, status_code=503, retryable=True, request_id=request_id)
    return error_envelope("internal_error", message, status_code=500, request_id=request_id)


def parse_tool_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return {"raw": text}
        value = json.loads(text[start : end + 1])
    return value if isinstance(value, dict) else {"value": value}


def invoke_tool(tool: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return parse_tool_json(tool.invoke(payload))


def invoke_catching(func: Callable[[], Any], *, request_id: str | None = None) -> dict[str, Any] | JSONResponse:
    try:
        return ok(func(), request_id=request_id)
    except Exception as exc:
        return map_exception(exc, request_id=request_id)
