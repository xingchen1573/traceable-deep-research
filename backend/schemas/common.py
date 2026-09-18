"""Common API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class Envelope(BaseModel):
    status: Literal["ok", "accepted", "error"]
    request_id: str
    data: Any | None = None
    meta: dict[str, Any] | None = None
    error: ErrorBody | None = None

