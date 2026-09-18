"""Job API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class JobCancelRequest(BaseModel):
    reason: str | None = None


class JobArchiveRequest(BaseModel):
    archived: bool

