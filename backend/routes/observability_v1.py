"""Observability API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..services import observability_service
from ..services.common import map_exception, ok
from .observability import observability_stream

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])


@router.get("/events")
def events(
    session_id: str | None = None,
    thread_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    request_id: str | None = None,
    event_type: str | None = None,
    level: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    cursor: str | None = None,
    days: int = Query(default=7, ge=1, le=90),
):
    try:
        return ok(
            observability_service.events(
                session_id=session_id,
                thread_id=thread_id,
                run_id=run_id,
                job_id=job_id,
                request_id=request_id,
                event_type=event_type,
                level=level,
                limit=limit,
                cursor=cursor,
                days=days,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.get("/stream")
async def stream(
    session_id: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    replay: int = Query(default=20, ge=0, le=200),
):
    return await observability_stream(session_id=session_id, run_id=run_id, job_id=job_id, replay=replay)


@router.get("/runs")
def runs_list(
    thread_id: str | None = None,
    job_id: str | None = None,
    days: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        return ok(
            observability_service.runs(
                thread_id=thread_id,
                job_id=job_id,
                days=days,
                limit=limit,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.get("/runs/{run_id}")
def run_detail(run_id: str, days: int = Query(default=30, ge=1, le=90)):
    try:
        return ok(observability_service.run_detail(run_id, days=days))
    except Exception as exc:
        return map_exception(exc)


@router.get("/jobs/{job_id}/trace")
def job_trace(job_id: str, days: int = Query(default=30, ge=1, le=90)):
    try:
        return ok(observability_service.job_trace(job_id, days=days))
    except Exception as exc:
        return map_exception(exc)

