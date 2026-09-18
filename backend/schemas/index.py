"""Index API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class IndexJobCreateRequest(BaseModel):
    force: bool = False
    mode: str = "lightrag"

