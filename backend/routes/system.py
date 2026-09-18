"""System API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter

from ..services.common import ok
from ..services import system_service

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health")
def health() -> dict:
    return ok(system_service.health())


@router.get("/capabilities")
def capabilities() -> dict:
    return ok(system_service.capabilities())


@router.get("/config/runtime")
def runtime_config() -> dict:
    return ok(system_service.runtime_config())

