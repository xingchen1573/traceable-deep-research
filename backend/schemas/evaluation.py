"""Evaluation API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class KnowledgeQaEvaluationJobRequest(BaseModel):
    topic: str
    backend: str = "auto"
    mode: str = "mix"


class KnowledgeQaCaseGenerationRequest(BaseModel):
    topic: str


class KnowledgeScopeEvaluationJobRequest(BaseModel):
    scope_type: Literal["knowledge_base", "theme", "research"]
    scope_id: str
    mode: str = "mix"
    include_history: bool = False


class KnowledgeScopeEvaluationCaseRequest(BaseModel):
    case: dict[str, Any]


class KnowledgeQaCaseVersionSwitchRequest(BaseModel):
    topic: str
    version: int


class ResearchProbeEvaluationJobRequest(BaseModel):
    case_ids: list[str] | None = None
    strategy: str = "hv_analysis"
    depth: str = "normal"
    case_interval_seconds: float = 30.0


class SingleUrlBriefEvaluationJobRequest(BaseModel):
    case_ids: list[str] | None = None


class EvaluationCaseUpsertRequest(BaseModel):
    eval_type: str
    topic: str | None = None
    case: dict[str, Any]
