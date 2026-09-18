"""Test management API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas.test import TestRunCreateRequest
from ..services import test_service
from ..services.common import accepted, map_exception, ok

router = APIRouter(prefix="/api/v1/tests", tags=["tests"])


@router.post("/runs")
def create_test_run(request: TestRunCreateRequest):
    try:
        job = test_service.create_test_run(
            suite=request.suite,
            target=request.target,
            mode=request.mode,
            options=request.options,
        )
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.get("/runs")
def list_test_runs():
    try:
        return ok(test_service.list_test_runs())
    except Exception as exc:
        return map_exception(exc)


@router.get("/runs/{run_id}")
def get_test_run(run_id: str):
    try:
        return ok(test_service.get_test_run(run_id))
    except Exception as exc:
        return map_exception(exc)

