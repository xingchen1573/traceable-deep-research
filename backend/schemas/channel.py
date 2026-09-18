"""Channel runtime API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChannelCreateRequest(BaseModel):
    external_id: str = Field(..., min_length=1)
    platform: str = "local"
    title: str | None = None
    memory_scope: Literal["channel"] = "channel"


class ChannelUpdateRequest(BaseModel):
    title: str | None = None
    mode: Literal["main", "kb_qa"] | None = None
    active_kb: str | None = None
    topic_binding: str | None = None
    memory_scope: Literal["channel"] | None = None


class ChannelPushConfigRequest(BaseModel):
    enabled: bool | None = None
    frequency: Literal["daily", "weekly_3x", "weekly_1x"] | None = None
    content_strategy: Literal["rotation", "new_only", "random"] | None = None
    push_time: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    timezone: str | None = Field(default=None, min_length=1, max_length=64)


class ChannelScopeBindingRequest(BaseModel):
    scope_type: Literal["knowledge_base", "theme"]
    scope_id: str = Field(..., min_length=1)
