"""Job API v1 routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from ..schemas.job import JobArchiveRequest, JobCancelRequest
from ..services.common import map_exception, ok
from ..services.job_service import job_service

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("")
def list_jobs(
    job_type: str | None = None,
    research_type: Literal["deep_research", "single_url"] | None = None,
    status: str | None = None,
    statuses: str | None = None,
    channel_id: str | None = None,
    archived: bool | None = None,
    query: str | None = None,
    cursor: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
):
    jobs, next_cursor, total = job_service.list_page(
        job_type=job_type,
        research_type=research_type,
        status=status,
        statuses=[value.strip() for value in statuses.split(',') if value.strip()] if statuses else None,
        channel_id=channel_id,
        archived=archived,
        query=query,
        cursor=cursor,
        offset=offset,
        limit=limit,
    )
    return ok(
        {
            "jobs": jobs,
            "next_cursor": next_cursor,
            "total": total,
        }
    )


@router.get("/stats")
def get_job_stats(archived: bool | None = False):
    return ok(job_service.stats(archived=archived))


@router.get("/{job_id}")
def get_job(job_id: str):
    try:
        return ok(job_service.get(job_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: JobCancelRequest):
    try:
        job = job_service.cancel(job_id, request.reason)
        return ok({"job_id": job["job_id"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.post("/{job_id}/archive")
def archive_job(job_id: str, request: JobArchiveRequest):
    try:
        job = job_service.set_archived(job_id, request.archived)
        return ok({"job_id": job["job_id"], "archived": job["archived"]})
    except Exception as exc:
        return map_exception(exc)
