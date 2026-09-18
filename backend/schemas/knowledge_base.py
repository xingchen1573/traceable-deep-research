"""Knowledge-base API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ActiveKnowledgeBaseRequest(BaseModel):
    topic: str
    thread_id: str | None = None


class KnowledgeDocumentCreateRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    sources: list[str] | None = None
    keywords: list[str] | None = None


class KnowledgeMetadataUpdateRequest(BaseModel):
    title: str | None = None
    version: str | None = None
    source_mode: str | None = None
    description: str | None = None
    status: str | None = None
    notes: str | None = None
    extra: dict[str, Any] | None = None


class KnowledgeQueryHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[KnowledgeQueryHistoryMessage] = Field(default_factory=list)
    backend: str = "auto"
    mode: str = "mix"
    include_citations: bool = True


class KnowledgeQaHistoryRequest(BaseModel):
    id: str | None = None
    topic: str = ""
    topicTitle: str = ""
    backend: str = "auto"
    mode: str = "mix"
    title: str = "未命名对话"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    createdAt: str | None = None
    updatedAt: str | None = None
