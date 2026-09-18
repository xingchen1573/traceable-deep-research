"""Conversation history API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..services.common import invoke_catching, ok
from ..services import conversation_service

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get("")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List conversation history ordered by most recently updated."""
    conversations = conversation_service.list_conversations(limit=limit, offset=offset)
    total = conversation_service.count_conversations()
    return ok({
        "conversations": conversations,
        "total": total,
    })


@router.get("/{thread_id}")
def get_conversation(thread_id: str) -> dict[str, Any]:
    """Get a single conversation metadata."""
    conv = conversation_service.get_conversation(thread_id)
    if conv is None:
        return ok(None)
    return ok(conv)


@router.get("/{thread_id}/messages")
def get_messages(
    thread_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Get all messages for a conversation."""
    messages = conversation_service.get_messages(thread_id, limit=limit, offset=offset)
    conv = conversation_service.get_conversation(thread_id)
    return ok({
        "thread_id": thread_id,
        "conversation": conv,
        "messages": messages,
    })


@router.delete("/{thread_id}")
def delete_conversation(thread_id: str) -> dict[str, Any]:
    """Delete a conversation and all its messages."""
    found = conversation_service.delete_conversation(thread_id)
    return ok({"thread_id": thread_id, "deleted": found})
