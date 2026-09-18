"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = None
    channel_id: str | None = None
    platform: str | None = None
    external_id: str | None = None
    user_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str
    message: str | None = None
    edited_action: dict[str, Any] | None = None


class DreamReviewRequest(BaseModel):
    decision: str
    note: str | None = None
