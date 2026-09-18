"""Schemas for the manifest-backed personal knowledge hierarchy API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


KnowledgeTreeAction = Literal[
    "create_knowledge_base",
    "create_theme",
    "archive_research",
    "update_research_status",
    "rename_node",
    "move_research",
    "trash_node",
    "trash_legacy",
    "restore_node",
]


class KnowledgeTreePreviewRequest(BaseModel):
    action: KnowledgeTreeAction
    payload: dict[str, Any] = Field(default_factory=dict)


class KnowledgeScopeHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class KnowledgeScopeQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    history: list[KnowledgeScopeHistoryMessage] = Field(default_factory=list)
    include_history: bool = False
    max_candidates: int = Field(default=12, ge=1, le=12)
