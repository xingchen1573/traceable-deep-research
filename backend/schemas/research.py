"""Research API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchJobCreateRequest(BaseModel):
    user_request: str = Field(..., min_length=1)
    source_mode: Literal["web", "repo", "mixed"] = "web"
    research_strategy: Literal["standard", "hv_analysis"] = "standard"
    depth: Literal["normal", "deep"] = "normal"
    topic: str | None = None
    channel_id: str | None = None
    requested_by: str | None = None
    delivery_target: Literal["web", "feishu_doc"] = "web"
    output_mode: Literal["single_document", "document_set"] = "document_set"
    required_titles: list[str] = Field(default_factory=list)
    # Legacy transport retained for stored jobs and explicit compatibility callers. New
    # interactive entry points use confirmed_plan + plan_token so the executed structure is
    # cryptographically bound to the draft the user reviewed.
    planned_sections: list[dict[str, Any]] = Field(default_factory=list)
    confirmed_plan: dict[str, Any] | None = None
    plan_id: str | None = None
    plan_token: str | None = None
    plan_policy: Literal["confirmed_required", "system_generated", "legacy_unplanned"] = "confirmed_required"
    # Disambiguation qualifier terms clarify_concept_tool's recon step produced (决策16: 静默随
    # plan 进 job，不展示给用户). Separate from planned_sections because it's a plan-level list
    # of bare strings, not a section -- doesn't fit planned_sections' list[dict] shape. Only
    # meaningful together with planned_sections: when both are set, the job skips re-running
    # _research_reconnaissance and reuses these directly instead (see _run_question_loop_research).
    disambiguation_terms: list[str] = Field(default_factory=list)
    max_rounds: int = 2
    auto_save: bool = False
    pipeline_mode: Literal["question_loop", "single_url"] = "question_loop"
    url: str | None = None


class ResearchPlanningClarifyRequest(BaseModel):
    user_request: str = Field(..., min_length=1, max_length=8000)


class ResearchPlanningDraftRequest(BaseModel):
    user_request: str = Field(..., min_length=1, max_length=8000)
    topic: str = Field(default="", max_length=240)
    recon_token: str = Field(..., min_length=1, max_length=50000)
    strategy: Literal["auto", "standard", "hv_analysis"] = "auto"
    adjustment: dict[str, Any] | None = None


class ResearchApproveRequest(BaseModel):
    topic: str
    documents: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] | None = None
    keywords: list[str] | None = None
    acknowledge_matrix_gaps: bool = False


class ResearchArchivePreviewRequest(BaseModel):
    knowledge_base_id: str | None = None
    theme_id: str | None = None
    new_knowledge_base: dict[str, Any] | None = None
    new_theme: dict[str, Any] | None = None
    title: str | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[Any] | None = None
    tags: list[str] | None = None
    acknowledge_matrix_gaps: bool = False
    acknowledge_quality_warnings: bool = False


class ResearchArchiveConfirmRequest(BaseModel):
    preview_id: str = Field(..., min_length=1)


class ResearchFollowupRequest(BaseModel):
    issue_id: str | None = None
    issue_type: str | None = None
    evidence_id: str | None = None
    subject: str | None = None
    comparison_axis: str | None = None
    query: str | None = None
    reason: str | None = None


class ResearchSectionRevisionRequest(BaseModel):
    """Manual, user-directed revision targeting one or more report sections.

    Distinct from ResearchFollowupRequest (which targets a single evidence issue and may
    trigger new retrieval): this only rewrites the targeted sections' prose using material
    already in the job's evidence_map, and leaves every other section untouched.
    """

    section_ids: list[str] = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
