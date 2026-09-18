"""Model/provider API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCheckRequest(BaseModel):
    targets: list[str] = Field(default_factory=lambda: ["chat_model", "embedding"])
    timeout_seconds: int = 30


class ChatModelConfigRequest(BaseModel):
    model: str = Field(..., min_length=1)
    base_url: str | None = None
    api_key: str | None = None


class RecallModelConfigRequest(BaseModel):
    follow_chat_model: bool = True
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class EmbeddingConfigRequest(BaseModel):
    model: str = Field(..., min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    dim: int = Field(..., gt=0)


class ResearchRoleModelConfigRequest(BaseModel):
    """Config for a high-frequency research role (evidence extraction / writing).

    Unlike the roles above, these two roles support multiple API keys for
    round-robin/rate-limit-avoidance pooling -- see
    backend/research/model_runtime.py::ResearchModelPool.
    """

    model: str = Field(..., min_length=1)
    base_url: str | None = None
    api_keys: list[str] | None = None


class ResearchRoleKeyCreateRequest(BaseModel):
    api_key: str = Field(..., min_length=1)


class ResearchRoleKeyCheckRequest(BaseModel):
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: int = Field(30, ge=1, le=120)


class ModelConfigUpdateRequest(BaseModel):
    chat_model: ChatModelConfigRequest
    recall_model: RecallModelConfigRequest = Field(default_factory=RecallModelConfigRequest)
    embedding: EmbeddingConfigRequest
    # These three are optional so existing callers that only manage chat/recall/embedding
    # keep working unchanged.
    dream_model: ChatModelConfigRequest | None = None
    research_evidence_model: ResearchRoleModelConfigRequest | None = None
    research_writer_model: ResearchRoleModelConfigRequest | None = None
