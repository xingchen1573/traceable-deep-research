"""Agent chat API v1 compatibility routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import ChatRequest, ResumeRequest
from ..services.common import ok
from .chat import chat, chat_pending, chat_resume

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
def chat_v1(request: ChatRequest) -> Any:
    return _wrap_legacy_chat_response(chat(request))


@router.get("/chat/pending/{thread_id}")
def chat_pending_v1(thread_id: str) -> Any:
    return _wrap_legacy_chat_response(chat_pending(thread_id))


@router.post("/chat/resume")
def chat_resume_v1(request: ResumeRequest) -> Any:
    return _wrap_legacy_chat_response(chat_resume(request))


def _wrap_legacy_chat_response(response: Any) -> Any:
    if not isinstance(response, JSONResponse):
        if isinstance(response, dict) and response.get("status") in {"ok", "accepted", "error"}:
            return response
        return ok(response)
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        return response
    request_id = payload.get("request_id")
    if payload.get("status") == "error":
        return response
    data = dict(payload)
    data.pop("request_id", None)
    state = data.pop("status", None)
    if state:
        data["state"] = state
    return ok(data, request_id=request_id)

