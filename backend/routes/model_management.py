"""Model/provider API v1 routes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request

from ..schemas.model import (
    ModelCheckRequest,
    ModelConfigUpdateRequest,
    ResearchRoleKeyCheckRequest,
    ResearchRoleKeyCreateRequest,
)
from ..services import model_service
from ..services.common import map_exception, ok

router = APIRouter(prefix="/api/v1", tags=["models"])


@router.get("/models/config")
def get_model_config():
    try:
        return ok(model_service.model_config())
    except Exception as exc:
        return map_exception(exc)


@router.put("/models/config")
def update_model_config(body: ModelConfigUpdateRequest, request: Request):
    try:
        try:
            payload = body.model_dump()
        except AttributeError:  # pydantic v1 fallback
            payload = body.dict()
        result = model_service.update_model_config(payload)
        # Hot-update (3.2): only reset the agents whose cached model client actually
        # depends on whatever env vars just changed -- see
        # docs/modules/model-config-completion-v0.1.md 3.2. get_agent()/get_review_agent()/
        # get_kb_qa_agent() lazily rebuild from the new env on next use.
        for attr in model_service.affected_agent_attrs(result.get("updated_fields") or []):
            setattr(request.app.state, attr, None)
        return ok(result)
    except Exception as exc:
        return map_exception(exc)


@router.post("/models/check")
def check_models(request: ModelCheckRequest):
    try:
        return ok(model_service.check_models(request.targets, request.timeout_seconds))
    except Exception as exc:
        return map_exception(exc)


@router.post("/models/research-roles/{role}/keys")
def add_research_role_key(role: Literal["evidence", "writer"], body: ResearchRoleKeyCreateRequest):
    try:
        return ok(model_service.add_research_role_key(role, body.api_key))
    except Exception as exc:
        return map_exception(exc)


@router.delete("/models/research-roles/{role}/keys/{key_id}")
def delete_research_role_key(role: Literal["evidence", "writer"], key_id: str):
    try:
        return ok(model_service.delete_research_role_key(role, key_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/models/research-roles/{role}/keys/check")
def check_research_role_candidate(role: Literal["evidence", "writer"], body: ResearchRoleKeyCheckRequest):
    try:
        if not body.api_key:
            raise ValueError("api_key is required when checking a new key")
        return ok(
            model_service.check_research_role_candidate(
                role,
                body.api_key,
                timeout_seconds=body.timeout_seconds,
                model=body.model,
                base_url=body.base_url,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.post("/models/research-roles/{role}/keys/{key_id}/check")
def check_research_role_key(
    role: Literal["evidence", "writer"],
    key_id: str,
    body: ResearchRoleKeyCheckRequest,
):
    try:
        return ok(
            model_service.check_research_role_key(
                role,
                key_id,
                timeout_seconds=body.timeout_seconds,
                model=body.model,
                base_url=body.base_url,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.get("/providers")
def providers():
    try:
        return ok(model_service.providers())
    except Exception as exc:
        return map_exception(exc)
