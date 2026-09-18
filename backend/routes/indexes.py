"""Index API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas.index import IndexJobCreateRequest
from ..services import index_service
from ..services.common import accepted, map_exception, ok

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["indexes"])


@router.post("/{topic}/index/jobs")
def create_index_job(topic: str, request: IndexJobCreateRequest):
    try:
        job = index_service.create_index_job(topic, force=request.force, mode=request.mode)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.delete("/{topic}/index")
def delete_index(topic: str):
    try:
        return ok(index_service.delete_index(topic))
    except Exception as exc:
        return map_exception(exc)

