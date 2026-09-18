"""bind_tools-capable chat model for the per-question agentic retrieval loop.

See docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md
"bind_tools 封装的具体接口设计": `invoke_research_chat` (model_runtime.py) is a single-call
JSON in/out helper and does not support `bind_tools`, so `create_agent`'s ReAct loop needs a
real `BaseChatModel`. This wraps the same `ResearchModelPool` "evidence" role credential pool
`invoke_research_chat` already uses, but pushes retry down to a single underlying
`ChatOpenAI.invoke()` call so it is transparent to `create_agent`/the agent loop -- a failure on
round 3 of a multi-round session never discards rounds 1-2's already-committed progress, because
retry never wraps more than one model turn.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field

from .audit import invoke_with_model_trace
from .model_runtime import (
    ResearchModelPool,
    ResearchRole,
    research_model_pool,
    research_role_base_url,
    research_role_model_name,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class AgenticRetryChatModel(BaseChatModel):
    """Retrying, tool-binding chat model backed by `ResearchModelPool`'s "evidence" role.

    Any exception (not just rate limiting -- see the module docstring's linked decision doc,
    section "超时/异常重试 + 兜底去留") from one underlying model call is retried up to
    `max_attempts` times total, rotating to a different credential from the pool each attempt.
    A fixed backoff sleep separates attempts (see `backoff_seconds`) -- a failure under real
    concurrent load is usually the whole pool being saturated (all keys briefly out of capacity),
    not one bad key, so retrying instantly just re-hits the same congestion; a short pause gives
    other in-flight callers a chance to free a slot. Retry exhaustion raises the last exception
    rather than silently degrading to a deterministic fallback -- callers (the agentic tools)
    decide what "this round produced nothing" means, this class never hides the failure.
    """

    role: ResearchRole = "evidence"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    backoff_seconds: float = 1.5
    bound_tools: list[Any] = Field(default_factory=list)
    tool_choice: Optional[str] = None
    on_call: Optional[Callable[[str, Optional[Exception]], None]] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "agentic-retry-research-chat-model"

    def bind_tools(
        self,
        tools: Sequence[dict | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        return self.__class__(
            role=self.role,
            timeout_seconds=self.timeout_seconds,
            max_attempts=self.max_attempts,
            backoff_seconds=self.backoff_seconds,
            bound_tools=list(tools),
            tool_choice=tool_choice,
            on_call=self.on_call,
        )

    def _build_client(self, api_key: str) -> Runnable:
        client: Runnable = ChatOpenAI(
            model=research_role_model_name(self.role),
            api_key=api_key,
            base_url=research_role_base_url(self.role),
            temperature=0,
            timeout=max(0.1, self.timeout_seconds),
            # This class owns retry semantics end to end; the SDK's own hidden retry-on-timeout
            # (openai.DEFAULT_MAX_RETRIES) must stay off or a single attempt silently re-runs the
            # full timeout underneath us -- see model_runtime.py's `invoke_research_chat` for the
            # production incident that motivated always setting this explicitly.
            max_retries=0,
            disable_streaming=True,
        )
        if self.bound_tools:
            client = (
                client.bind_tools(self.bound_tools, tool_choice=self.tool_choice)
                if self.tool_choice
                else client.bind_tools(self.bound_tools)
            )
        return client

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        pool: ResearchModelPool = research_model_pool(self.role)
        last_exc: Exception | None = None
        parent_call_id: str | None = None
        attempts = max(1, self.max_attempts)
        for attempt in range(attempts):
            with pool.credential() as credential:
                try:
                    client = self._build_client(credential.api_key)
                    ai_message = invoke_with_model_trace(
                        lambda: client.invoke(messages, stop=stop, **kwargs),
                        messages=messages,
                        requested_provider="primary",
                        requested_model=research_role_model_name(self.role),
                        actual_provider="primary",
                        call_kind="retry" if attempt else "agentic_retrieval",
                        parent_call_id=parent_call_id,
                        attempt=attempt + 1,
                        parameters={
                            "temperature": 0,
                            "timeout_seconds": max(0.1, self.timeout_seconds),
                            "max_retries": 0,
                            "base_url": research_role_base_url(self.role),
                            "credential_index": credential.index,
                            "stop": stop,
                            "invoke_kwargs": kwargs,
                            "bound_tools": self.bound_tools,
                            "tool_choice": self.tool_choice,
                        },
                    )
                    pool.record_success(credential)
                    if self.on_call:
                        self.on_call("success", None)
                    return ChatResult(generations=[ChatGeneration(message=ai_message)])
                except Exception as exc:  # pragma: no cover - exact provider failures vary
                    last_exc = exc
                    parent_call_id = str(getattr(exc, "harness_call_id", "") or parent_call_id or "") or None
                    pool.record_failure(credential, exc)
                    if self.on_call:
                        self.on_call("failure", exc)
                    if attempt < attempts - 1 and self.backoff_seconds > 0:
                        time.sleep(self.backoff_seconds)
                    continue
        raise last_exc or RuntimeError("agentic retry chat model failed without an error")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.to_thread(self._generate, messages, stop, None, **kwargs)


def build_agentic_model(
    *,
    role: ResearchRole = "evidence",
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
    backoff_seconds: float | None = None,
    on_call: Callable[[str, Optional[Exception]], None] | None = None,
) -> AgenticRetryChatModel:
    resolved_timeout = timeout_seconds if timeout_seconds is not None else _env_float(
        "RESEARCH_AGENTIC_MODEL_TIMEOUT_SECONDS",
        _env_float("RESEARCH_EVIDENCE_TIMEOUT_SECONDS", 60.0),
    )
    # "2次重试（共3次尝试）" -- see the decision doc's "超时/异常重试" section.
    resolved_attempts = max_attempts if max_attempts is not None else _env_int("RESEARCH_AGENTIC_MODEL_MAX_ATTEMPTS", 3)
    resolved_backoff = (
        backoff_seconds if backoff_seconds is not None else _env_float("RESEARCH_AGENTIC_RETRY_BACKOFF_SECONDS", 1.5)
    )
    return AgenticRetryChatModel(
        role=role,
        timeout_seconds=resolved_timeout,
        max_attempts=resolved_attempts,
        backoff_seconds=resolved_backoff,
        on_call=on_call,
    )
