"""Feishu adapter API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..services import feishu_service, feishu_ws_service
from ..services.common import map_exception, ok

router = APIRouter(prefix="/api/v1/feishu", tags=["feishu"])


@router.post("/webhook")
async def feishu_webhook(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
        result = feishu_service.handle_event(payload)
        if "challenge" in result:
            return result
        return ok(result)
    except Exception as exc:
        return map_exception(exc)


@router.get("/replies")
def list_replies(limit: int = 50):
    try:
        return ok({"replies": feishu_service.list_replies(limit=limit)})
    except Exception as exc:
        return map_exception(exc)


@router.get("/config")
def get_config():
    try:
        data = feishu_service.config_status()
        data["long_connection"] = feishu_ws_service.status()
        return ok(data)
    except Exception as exc:
        return map_exception(exc)
