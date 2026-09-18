"""Evaluation API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..schemas.evaluation import (
    EvaluationCaseUpsertRequest,
    KnowledgeQaCaseGenerationRequest,
    KnowledgeQaCaseVersionSwitchRequest,
    KnowledgeQaEvaluationJobRequest,
    KnowledgeScopeEvaluationCaseRequest,
    KnowledgeScopeEvaluationJobRequest,
    ResearchProbeEvaluationJobRequest,
    SingleUrlBriefEvaluationJobRequest,
)
from ..services import evaluation_service
from ..services.common import accepted, map_exception, ok

router = APIRouter(prefix="/api/v1/evaluations", tags=["evaluations"])


@router.post("/knowledge-qa/jobs")
def create_knowledge_qa_evaluation_job(request: KnowledgeQaEvaluationJobRequest):
    try:
        job = evaluation_service.create_knowledge_qa_eval_job(request.topic, backend=request.backend, mode=request.mode)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.post("/knowledge-scopes/jobs")
def create_knowledge_scope_evaluation_job(request: KnowledgeScopeEvaluationJobRequest):
    try:
        job = evaluation_service.create_knowledge_scope_eval_job(
            request.scope_type,
            request.scope_id,
            mode=request.mode,
            include_history=request.include_history,
        )
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.get("/knowledge-scopes/{scope_type}/{scope_id}/cases")
def list_knowledge_scope_evaluation_cases(scope_type: str, scope_id: str):
    try:
        return ok(evaluation_service.list_knowledge_scope_eval_cases(scope_type, scope_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/knowledge-scopes/{scope_type}/{scope_id}/cases")
def upsert_knowledge_scope_evaluation_case(
    scope_type: str, scope_id: str, request: KnowledgeScopeEvaluationCaseRequest
):
    try:
        return ok(evaluation_service.upsert_knowledge_scope_eval_case(scope_type, scope_id, request.case))
    except Exception as exc:
        return map_exception(exc)


@router.delete("/knowledge-scopes/{scope_type}/{scope_id}/cases/{case_id}")
def delete_knowledge_scope_evaluation_case(scope_type: str, scope_id: str, case_id: str):
    try:
        evaluation_service.delete_knowledge_scope_eval_case(scope_type, scope_id, case_id)
        return ok({"deleted": case_id})
    except Exception as exc:
        return map_exception(exc)


@router.post("/knowledge-qa/case-generation/jobs")
def create_knowledge_qa_case_generation_job(request: KnowledgeQaCaseGenerationRequest):
    try:
        job = evaluation_service.create_knowledge_qa_case_generation_job(request.topic)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.post("/knowledge-qa/case-generation/jobs/{job_id}/approve")
def approve_knowledge_qa_case_generation_job(job_id: str):
    try:
        return ok(evaluation_service.approve_knowledge_qa_case_generation(job_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/knowledge-qa/case-versions/activate")
def activate_knowledge_qa_case_version(request: KnowledgeQaCaseVersionSwitchRequest):
    try:
        return ok(evaluation_service.switch_knowledge_qa_case_version(request.topic, request.version))
    except Exception as exc:
        return map_exception(exc)


@router.post("/research-probe/jobs")
def create_research_probe_evaluation_job(request: ResearchProbeEvaluationJobRequest):
    try:
        job = evaluation_service.create_research_probe_eval_job(
            request.case_ids,
            strategy=request.strategy,
            depth=request.depth,
            case_interval_seconds=request.case_interval_seconds,
        )
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.post("/single-url-brief/jobs")
def create_single_url_brief_evaluation_job(request: SingleUrlBriefEvaluationJobRequest):
    try:
        job = evaluation_service.create_single_url_brief_eval_job(request.case_ids)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.get("/jobs/{job_id}")
def get_evaluation_job(job_id: str):
    try:
        return ok(evaluation_service.get_eval_job(job_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/cases")
def list_evaluation_cases(eval_type: str, topic: str | None = None):
    try:
        return ok(evaluation_service.list_eval_cases(eval_type, topic=topic))
    except Exception as exc:
        return map_exception(exc)


@router.post("/cases")
def upsert_evaluation_case(request: EvaluationCaseUpsertRequest):
    try:
        case = evaluation_service.upsert_eval_case(request.eval_type, request.case, topic=request.topic)
        return ok(case)
    except Exception as exc:
        return map_exception(exc)


@router.delete("/cases/{case_id}")
def delete_evaluation_case(case_id: str, eval_type: str, topic: str | None = None):
    try:
        evaluation_service.delete_eval_case(eval_type, case_id, topic=topic)
        return ok({"deleted": case_id})
    except Exception as exc:
        return map_exception(exc)


@router.get("/reports")
def list_reports(topic: str | None = None, limit: int = Query(default=50, ge=1, le=200)):
    try:
        return ok(evaluation_service.list_reports(topic=topic, limit=limit))
    except Exception as exc:
        return map_exception(exc)


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    try:
        return ok(evaluation_service.get_report(report_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/reports/{report_id}/compare")
def compare_reports(report_id: str, baseline_report_id: str):
    try:
        return ok(evaluation_service.compare_reports(report_id, baseline_report_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/research-probe/reports")
def list_research_probe_reports(limit: int = Query(default=50, ge=1, le=200)):
    try:
        return ok(evaluation_service.list_research_probe_reports(limit=limit))
    except Exception as exc:
        return map_exception(exc)


@router.get("/research-probe/reports/{report_id}")
def get_research_probe_report(report_id: str):
    try:
        return ok(evaluation_service.get_research_probe_report(report_id))
    except Exception as exc:
        return map_exception(exc)
