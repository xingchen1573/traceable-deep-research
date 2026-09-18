"""Response helpers and error handling utilities."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi.responses import JSONResponse
from fastapi import HTTPException
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

try:
    from ..observability import emit_event, new_run_id
except ImportError:  # pragma: no cover
    from observability import emit_event, new_run_id

from .models import ResumeRequest


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def response_json(payload: dict[str, Any], *, request_id: str, status_code: int = 200) -> JSONResponse:
    """Create a standardized JSON response with request ID header."""
    content = dict(payload)
    content["request_id"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )


def error_response(
    exc: Exception,
    *,
    default_message: str,
    session_id: str,
    event_type: str,
    request_id: str,
    component: str = "server",
) -> JSONResponse:
    """Create a standardized error response with OpenAI error mapping."""
    status_code = 500
    error_code = "internal_error"
    retryable = False
    message = default_message
    extra: dict[str, Any] = {}

    if isinstance(exc, APITimeoutError):
        status_code = 504
        error_code = "upstream_timeout"
        retryable = True
        message = "The upstream model request timed out."
    elif isinstance(exc, APIConnectionError):
        status_code = 502
        error_code = "upstream_connection_error"
        retryable = True
        message = "The upstream model endpoint could not be reached."
    elif isinstance(exc, RateLimitError):
        status_code = 429
        error_code = "upstream_rate_limited"
        retryable = True
        message = "The upstream model provider rate-limited this request."
    elif isinstance(exc, AuthenticationError):
        status_code = 401
        error_code = "upstream_auth_error"
        message = "The upstream model provider rejected the API credentials."
    elif isinstance(exc, PermissionDeniedError):
        status_code = 403
        error_code = "upstream_permission_denied"
        message = "The upstream model provider denied access to this request."
    elif isinstance(exc, BadRequestError):
        status_code = 400
        error_code = "model_request_invalid"
        message = "The upstream model provider rejected the request payload."
    elif isinstance(exc, InternalServerError):
        status_code = 503
        error_code = "upstream_unavailable"
        retryable = True
        message = "The upstream model provider is temporarily unavailable."
    elif isinstance(exc, APIStatusError):
        upstream_status = getattr(exc, "status_code", None)
        extra["upstream_status"] = upstream_status
        status_code = 502 if upstream_status and upstream_status >= 500 else 400
        error_code = "upstream_api_error" if status_code == 502 else "model_request_invalid"
        retryable = status_code >= 500
        message = "The upstream model provider returned an API error."

    emit_event(
        event_type,
        component,
        message,
        session_id=session_id,
        run_id=request_id,
        level="error",
        data={
            "error_code": error_code,
            "status_code": status_code,
            "retryable": retryable,
            "error": str(exc),
            **extra,
        },
    )
    return response_json(
        {
            "status": "error",
            "error": {
                "code": error_code,
                "message": message,
                "retryable": retryable,
                "type": type(exc).__name__,
                **extra,
            },
        },
        request_id=request_id,
        status_code=status_code,
    )


def json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from text, handling various formats."""
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def serialize_interrupt(interrupt: Any) -> Any:
    """Serialize an interrupt for API response."""
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def decision_payload(request: ResumeRequest) -> dict[str, Any]:
    """Convert a ResumeRequest decision to internal format."""
    decision = request.decision.strip().lower()
    if decision == "approve":
        return {"type": "approve"}
    if decision == "reject":
        return {"type": "reject", "message": request.message or "Rejected by API reviewer."}
    if decision == "edit":
        if not request.edited_action:
            raise HTTPException(status_code=400, detail="edited_action is required for decision=edit")
        return {"type": "edit", "edited_action": request.edited_action}
    raise HTTPException(status_code=400, detail="decision must be approve, edit, or reject")


def pending_interrupt_view(thread_id: str, pending: dict[str, Any]) -> dict[str, Any]:
    """Format a pending interrupt for API response."""
    return {
        "status": "interrupted",
        "thread_id": thread_id,
        "run_id": pending.get("run_id"),
        "interrupts": pending.get("interrupts") or [],
        "resume_supported": True,
        "pending_interrupt": {
            "created_at": pending.get("created_at"),
            "interrupt_count": len(pending.get("interrupts") or []),
        },
    }
