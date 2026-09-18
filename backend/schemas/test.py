"""Test run API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TestRunCreateRequest(BaseModel):
    suite: str = "api_smoke"
    target: str | None = None
    mode: str = "smoke"
    options: dict[str, Any] | None = None

