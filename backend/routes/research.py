"""Research API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..schemas.research import (
    ResearchArchiveConfirmRequest,
    ResearchArchivePreviewRequest,
    ResearchApproveRequest,
    ResearchFollowupRequest,
    ResearchJobCreateRequest,
    ResearchPlanningClarifyRequest,
    ResearchPlanningDraftRequest,
    ResearchSectionRevisionRequest,
)
from ..services import research_planning_service, research_service
from ..services.common import accepted, map_exception, ok

router = APIRouter(prefix="/api/v1/research", tags=["research"])


@router.post("/planning/clarify")
def clarify_research_plan(request: ResearchPlanningClarifyRequest):
    """Run the same preflight concept clarification used by the conversational Agent."""
    try:
        return ok(research_planning_service.clarify_concept(request.user_request))
    except Exception as exc:
        return map_exception(exc)


@router.post("/planning/draft")
def draft_research_plan(request: ResearchPlanningDraftRequest):
    """Generate or revise a plan without creating a research job."""
    try:
        return ok(
            research_planning_service.draft_research_plan(
                user_request=request.user_request,
                topic=request.topic,
                recon_token_value=request.recon_token,
                strategy=request.strategy,
                adjustment=request.adjustment,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.post("/jobs")
def create_research_job(request: ResearchJobCreateRequest):
    try:
        try:
            payload = request.model_dump()
        except AttributeError:
            payload = request.dict()
        job = research_service.create_research_job(payload)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.get("/jobs/{job_id}")
def get_research_job(job_id: str):
    try:
        return ok(research_service.get_research_job(job_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/jobs/{job_id}/artifacts")
def get_research_artifacts(job_id: str):
    try:
        return ok(research_service.research_artifacts(job_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/jobs/{job_id}/evidence-reader")
def get_research_evidence_reader(job_id: str) -> HTMLResponse:
    """Serve the single_url_brief 证据标注阅读版 HTML (VISION.md 第2节) as a directly-openable
    page -- not JSON-wrapped, so a plain link (e.g. shared in the frontend job view) opens the
    rendered brief in a browser tab once the backend is deployed on a real server."""
    path = research_service.RESEARCH_ARTIFACT_DIR / research_service._safe_artifact_name(job_id) / "brief.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evidence reader html not found for this job")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.post("/jobs/{job_id}/approve")
def approve_research(job_id: str, request: ResearchApproveRequest):
    try:
        try:
            payload = request.model_dump()
        except AttributeError:
            payload = request.dict()
        return ok(research_service.approve_research(job_id, payload))
    except Exception as exc:
        return map_exception(exc)


@router.post("/jobs/{job_id}/archive-preview")
def preview_research_archive(job_id: str, request: ResearchArchivePreviewRequest):
    try:
        try:
            payload = request.model_dump()
        except AttributeError:
            payload = request.dict()
        return ok(research_service.preview_research_archive(job_id, payload))
    except Exception as exc:
        return map_exception(exc)


@router.post("/jobs/{job_id}/archive-confirm")
def confirm_research_archive(job_id: str, request: ResearchArchiveConfirmRequest):
    try:
        return ok(research_service.confirm_research_archive(job_id, request.preview_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/jobs/{job_id}/followup")
def followup_research(job_id: str, request: ResearchFollowupRequest):
    try:
        try:
            payload = request.model_dump()
        except AttributeError:
            payload = request.dict()
        return ok(research_service.followup_research(job_id, payload))
    except Exception as exc:
        return map_exception(exc)


@router.post("/jobs/{job_id}/revise-sections")
def revise_research_sections(job_id: str, request: ResearchSectionRevisionRequest):
    try:
        try:
            payload = request.model_dump()
        except AttributeError:
            payload = request.dict()
        return ok(research_service.revise_research_sections(job_id, payload))
    except Exception as exc:
        return map_exception(exc)
