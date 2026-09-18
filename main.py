"""CLI entry for Deep Research.

Run from the repository root:

    python main.py

Environment variables:

    OPENAI_API_KEY
    OPENAI_EXTRA_API_KEYS (optional, comma-separated; defaults to reusing
        RESEARCH_MODEL_EXTRA_API_KEYS for main-agent rate-limit failover)
    MIMO_EXTRA_API_KEYS (optional, comma-separated; used only when
        OPENAI_BASE_URL points to api.xiaomimimo.com)
    OPENAI_BASE_URL
    MODEL_NAME
    RESEARCH_UNIFIED_MODEL_* / RESEARCH_PREFERRED_* (optional authoritative route)
    RECALL_API_KEY
    RECALL_BASE_URL
    RECALL_MODEL_NAME
    DREAM_API_KEY
    DREAM_BASE_URL
    DREAM_MODEL_NAME
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import PrivateAttr

try:
    from . import memory as memory_store
    from .middleware import (
        ContextCompactMiddleware,
        ContextSnipMiddleware,
        FrustrationMiddleware,
        LoopGuardMiddleware,
        MicroCompactMiddleware,
        MemoryWritePolicyMiddleware,
        ReactiveCompactMiddleware,
        RelevantMemoryRecallMiddleware,
        SkillSelectionMiddleware,
    )
    from .observability import emit_event, new_run_id, set_observation_context
    from .prompts import BASE_PROMPT, KB_QA_AGENT_PROMPT, REVIEW_AGENT_PROMPT
    from .tools import (
        approve_research_job_tool,
        confirm_knowledge_tree_change_tool,
        confirm_research_archive_tool,
        check_research_job_tool,
        clarify_concept_tool,
        configure_surfaced_registry,
        draft_research_plan_tool,
        get_latest_research_plan_tool,
        ensure_demo_files,
        ensure_knowledge_base,
        list_conversation_jobs,
        list_knowledge_bases,
        list_skills,
        list_memory_topics,
        load_memory_topic,
        load_skill,
        read_dream_report,
        read_memory_index,
        refresh_memory_index,
        read_file,
        delete_knowledge_base_index,
        inspect_knowledge_base,
        query_knowledge_base,
        research_coverage_check,
        revise_report_section_tool,
        save_knowledge,
        save_knowledge_set,
        save_single_url_brief_tool,
        set_active_knowledge_base,
        start_research_job_tool,
        start_single_url_brief_tool,
        summarize_knowledge_base,
        write_file,
        search_memory,
        search_web,
        set_current_thread_id,
        write_memory_topic,
    )
    from .workers import knowledge_manager_worker_tool, repo_worker_tool, web_worker_tool
except ImportError:  # pragma: no cover - script mode fallback
    import memory as memory_store
    from middleware import (
        ContextCompactMiddleware,
        ContextSnipMiddleware,
        FrustrationMiddleware,
        LoopGuardMiddleware,
        MicroCompactMiddleware,
        MemoryWritePolicyMiddleware,
        ReactiveCompactMiddleware,
        RelevantMemoryRecallMiddleware,
        SkillSelectionMiddleware,
    )
    from observability import emit_event, new_run_id, set_observation_context
    from prompts import BASE_PROMPT, KB_QA_AGENT_PROMPT, REVIEW_AGENT_PROMPT
    from tools import (
        approve_research_job_tool,
        confirm_knowledge_tree_change_tool,
        confirm_research_archive_tool,
        check_research_job_tool,
        clarify_concept_tool,
        configure_surfaced_registry,
        draft_research_plan_tool,
        get_latest_research_plan_tool,
        ensure_demo_files,
        ensure_knowledge_base,
        list_conversation_jobs,
        list_knowledge_bases,
        list_skills,
        list_memory_topics,
        load_memory_topic,
        load_skill,
        read_dream_report,
        read_memory_index,
        refresh_memory_index,
        read_file,
        delete_knowledge_base_index,
        inspect_knowledge_base,
        query_knowledge_base,
        research_coverage_check,
        revise_report_section_tool,
        save_knowledge,
        save_knowledge_set,
        save_single_url_brief_tool,
        set_active_knowledge_base,
        start_research_job_tool,
        start_single_url_brief_tool,
        summarize_knowledge_base,
        write_file,
        search_memory,
        search_web,
        set_current_thread_id,
        write_memory_topic,
    )
    from workers import knowledge_manager_worker_tool, repo_worker_tool, web_worker_tool


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SESSION_ID = "harness-demo-session"

# Message types to print in clean (default) mode
_AI_TYPES = {"ai", "assistant", ""}

# Icon prefixes for debug mode
_DEBUG_ICONS = {
    "ai": "[AI] ",
    "system": "[System] ",
    "human": "[Human] ",
    "tool": "[Tool] ",
    "assistant": "[AI] ",
    "": "[AI] ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research CLI")
    parser.add_argument(
        "--debug",
        "--verbose",
        action="store_true",
        default=False,
        help="Show all message types with icons (AI, System, Tool, Human)",
    )
    return parser.parse_args()


def _chat_api_keys() -> list[str]:
    """Resolve the API key pool for the main chat model (build_model()).

    Primary key is OPENAI_API_KEY. Extra keys come from OPENAI_EXTRA_API_KEYS;
    when that variable is unset, reuse RESEARCH_MODEL_EXTRA_API_KEYS. A research
    dispatch needs several consecutive main-agent decisions (clarify, plan, then
    present the plan), so keeping the main model on only the primary credential
    can exhaust that key's rolling TPM before the final response is produced.
    Recall deliberately uses only the primary key in build_recall_model(), so
    this fallback does not multiply the optional recall call across the pool.
    """
    primary = (os.getenv("OPENAI_API_KEY") or "").strip()
    extras_raw = os.getenv("OPENAI_EXTRA_API_KEYS")
    if extras_raw is None and "api.xiaomimimo.com" in (os.getenv("OPENAI_BASE_URL") or "").lower():
        extras_raw = os.getenv("MIMO_EXTRA_API_KEYS")
    if extras_raw is None:
        extras_raw = os.getenv("RESEARCH_MODEL_EXTRA_API_KEYS", "")
    extras = [item.strip() for item in extras_raw.split(",")]
    return list(dict.fromkeys(item for item in [primary, *extras] if item))


class RotatingChatOpenAI(ChatOpenAI):
    """ChatOpenAI with key failover and bounded managed-gateway retries.

    Delegates each call to a per-key ChatOpenAI instance (langchain-openai builds
    its SDK client once at construction time from the key it was given, so a
    single instance can't just swap self.api_key at call time). Reuses
    backend.research.model_runtime.ResearchModelPool for the same cooldown/retry
    bookkeeping the research evidence/writer pools already rely on, instead of
    duplicating it. Only non-streaming (_generate) and streaming (_stream) sync
    paths rotate; async paths aren't overridden because build_agent()/
    build_kb_qa_agent() only ever call .invoke()/.stream() today (see
    backend/dependencies.py, main.py's own CLI loop) -- if that changes,
    _agenerate/_astream need the same treatment.
    """

    _pool: Any = PrivateAttr()
    _delegates: dict[int, ChatOpenAI] = PrivateAttr(default_factory=dict)
    _delegate_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _fallback_pool: Any = PrivateAttr(default=None)
    _fallback_delegates: dict[int, ChatOpenAI] = PrivateAttr(default_factory=dict)
    _fallback_delegate_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _gateway_managed: bool = PrivateAttr(default=False)
    _gateway_retries: int = PrivateAttr(default=0)

    def __init__(
        self,
        *,
        api_keys: list[str],
        fallback_api_keys: list[str] | None = None,
        fallback_kwargs: dict[str, Any] | None = None,
        gateway_managed: bool = False,
        gateway_retries: int = 0,
        per_key_concurrency: int | None = None,
        key_acquire_timeout_seconds: float | None = None,
        rate_limit_cooldown_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_keys[0], **kwargs)
        try:
            from .backend.research.model_runtime import ResearchModelPool
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.research.model_runtime import ResearchModelPool
        self._pool = ResearchModelPool(
            api_keys=api_keys,
            per_key_concurrency=(
                per_key_concurrency
                if per_key_concurrency is not None
                else int(os.getenv("CHAT_MODEL_PER_KEY_CONCURRENCY", "2"))
            ),
            acquire_timeout_seconds=(
                key_acquire_timeout_seconds
                if key_acquire_timeout_seconds is not None
                else float(os.getenv("CHAT_MODEL_KEY_ACQUIRE_TIMEOUT_SECONDS", "30"))
            ),
            rate_limit_cooldown_seconds=(
                rate_limit_cooldown_seconds
                if rate_limit_cooldown_seconds is not None
                else float(os.getenv("CHAT_MODEL_RATE_LIMIT_COOLDOWN_SECONDS", "20"))
            ),
            gateway_managed=gateway_managed,
        )
        self._delegate_kwargs = kwargs
        self._delegates = {0: self}
        self._gateway_managed = gateway_managed
        self._gateway_retries = max(0, gateway_retries)
        if fallback_api_keys and fallback_kwargs:
            self._fallback_pool = ResearchModelPool(
                api_keys=fallback_api_keys,
                per_key_concurrency=int(os.getenv("MIMO_FALLBACK_PER_KEY_CONCURRENCY", "1")),
                acquire_timeout_seconds=float(os.getenv("MIMO_FALLBACK_KEY_ACQUIRE_TIMEOUT_SECONDS", "30")),
                rate_limit_cooldown_seconds=float(os.getenv("MIMO_FALLBACK_RATE_LIMIT_COOLDOWN_SECONDS", "20")),
            )
            self._fallback_delegate_kwargs = fallback_kwargs

    def _delegate_for(self, index: int, api_key: str) -> ChatOpenAI:
        delegate = self._delegates.get(index)
        if delegate is None:
            delegate = ChatOpenAI(api_key=api_key, **self._delegate_kwargs)
            self._delegates[index] = delegate
        return delegate

    def _fallback_delegate_for(self, index: int, api_key: str) -> ChatOpenAI:
        delegate = self._fallback_delegates.get(index)
        if delegate is None:
            delegate = ChatOpenAI(api_key=api_key, **self._fallback_delegate_kwargs)
            self._fallback_delegates[index] = delegate
        return delegate

    def _record_provider_fallback(self, outcome: str, reason: str = "") -> None:
        try:
            from .backend.research.model_runtime import record_provider_fallback
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.research.model_runtime import record_provider_fallback
        record_provider_fallback("main_agent", outcome, reason)

    def _can_use_provider_fallback(self, exc: Exception | None) -> bool:
        if self._fallback_pool is None or exc is None:
            return False
        try:
            from .backend.research.model_runtime import should_use_provider_fallback
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.research.model_runtime import should_use_provider_fallback
        return should_use_provider_fallback(exc)

    def _can_retry_managed_gateway(self, exc: Exception) -> bool:
        if not self._gateway_managed:
            return False
        try:
            from .backend.research.model_runtime import _is_retryable_gateway_error
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.research.model_runtime import _is_retryable_gateway_error
        return _is_retryable_gateway_error(exc)

    def _generate_with_fallback(self, messages: Any, stop: Any, run_manager: Any, **kwargs: Any) -> Any:
        self._record_provider_fallback("triggered", "main_agent_primary_exhausted")
        last_exc: Exception | None = None
        for _ in range(max(1, len(self._fallback_pool.api_keys))):
            with self._fallback_pool.credential() as credential:
                delegate = self._fallback_delegate_for(credential.index, credential.api_key)
                try:
                    result = ChatOpenAI._generate(delegate, messages, stop=stop, run_manager=run_manager, **kwargs)
                    self._fallback_pool.record_success(credential)
                    self._record_provider_fallback("success")
                    return result
                except Exception as exc:  # noqa: BLE001 - provider failover must classify the error
                    last_exc = exc
                    rate_limited = self._fallback_pool.record_failure(credential, exc)
                    if rate_limited or _is_transient_chat_error(exc):
                        continue
                    self._record_provider_fallback("failure")
                    raise
        self._record_provider_fallback("failure")
        raise last_exc  # type: ignore[misc]

    def _stream_with_fallback(self, messages: Any, stop: Any, run_manager: Any, **kwargs: Any) -> Any:
        self._record_provider_fallback("triggered", "main_agent_primary_exhausted")
        last_exc: Exception | None = None
        for _ in range(max(1, len(self._fallback_pool.api_keys))):
            with self._fallback_pool.credential() as credential:
                delegate = self._fallback_delegate_for(credential.index, credential.api_key)
                chunks = ChatOpenAI._stream(delegate, messages, stop=stop, run_manager=run_manager, **kwargs)
                yielded_any = False
                try:
                    for chunk in chunks:
                        yielded_any = True
                        yield chunk
                    self._fallback_pool.record_success(credential)
                    self._record_provider_fallback("success")
                    return
                except Exception as exc:  # noqa: BLE001 - see _generate_with_fallback
                    last_exc = exc
                    rate_limited = self._fallback_pool.record_failure(credential, exc)
                    if yielded_any:
                        self._record_provider_fallback("failure")
                        raise
                    if rate_limited or _is_transient_chat_error(exc):
                        continue
                    self._record_provider_fallback("failure")
                    raise
        self._record_provider_fallback("failure")
        if last_exc:
            raise last_exc

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        transient_attempts = 0
        gateway_retries_used = 0
        max_transient_attempts = max(1, min(
            len(self._pool.api_keys),
            int(os.getenv("CHAT_MODEL_TRANSIENT_KEY_ATTEMPTS", "2")),
        ))
        attempt_budget = max(1, len(self._pool.api_keys)) + self._gateway_retries
        for _ in range(attempt_budget):
            with self._pool.credential() as credential:
                delegate = self._delegate_for(credential.index, credential.api_key)
                try:
                    result = ChatOpenAI._generate(delegate, messages, stop=stop, run_manager=run_manager, **kwargs)
                    self._pool.record_success(credential)
                    return result
                except Exception as exc:  # noqa: BLE001 - must inspect provider error text before deciding to retry
                    last_exc = exc
                    rate_limited = self._pool.record_failure(credential, exc)
                    emit_event(
                        "model.key.failed",
                        "RotatingChatOpenAI",
                        "Chat model key attempt failed.",
                        level="warning",
                        data={
                            "key_index": credential.index,
                            "error_type": type(exc).__name__,
                            "rate_limited": rate_limited,
                            "yielded_any": False,
                        },
                    )
                    if rate_limited:
                        continue
                    if self._can_retry_managed_gateway(exc) and gateway_retries_used < self._gateway_retries:
                        gateway_retries_used += 1
                        time.sleep(min(1.0, 0.25 * (2 ** (gateway_retries_used - 1))))
                        continue
                    if _is_transient_chat_error(exc):
                        transient_attempts += 1
                        if transient_attempts < max_transient_attempts:
                            continue
                        break
                    raise
        if self._can_use_provider_fallback(last_exc):
            return self._generate_with_fallback(messages, stop, run_manager, **kwargs)
        raise last_exc  # type: ignore[misc]

    def _stream(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        transient_attempts = 0
        gateway_retries_used = 0
        max_transient_attempts = max(1, min(
            len(self._pool.api_keys),
            int(os.getenv("CHAT_MODEL_TRANSIENT_KEY_ATTEMPTS", "2")),
        ))
        attempt_budget = max(1, len(self._pool.api_keys)) + self._gateway_retries
        for _ in range(attempt_budget):
            with self._pool.credential() as credential:
                delegate = self._delegate_for(credential.index, credential.api_key)
                chunks = ChatOpenAI._stream(delegate, messages, stop=stop, run_manager=run_manager, **kwargs)
                yielded_any = False
                try:
                    for chunk in chunks:
                        yielded_any = True
                        yield chunk
                    self._pool.record_success(credential)
                    return
                except Exception as exc:  # noqa: BLE001 - see _generate
                    last_exc = exc
                    rate_limited = self._pool.record_failure(credential, exc)
                    emit_event(
                        "model.key.failed",
                        "RotatingChatOpenAI",
                        "Streaming chat model key attempt failed.",
                        level="warning",
                        data={
                            "key_index": credential.index,
                            "error_type": type(exc).__name__,
                            "rate_limited": rate_limited,
                            "yielded_any": yielded_any,
                        },
                    )
                    # Once we've handed the caller partial output, a retry would
                    # duplicate it -- only retry a stream that failed before its
                    # first chunk.
                    if yielded_any:
                        raise
                    if rate_limited:
                        continue
                    if self._can_retry_managed_gateway(exc) and gateway_retries_used < self._gateway_retries:
                        gateway_retries_used += 1
                        time.sleep(min(1.0, 0.25 * (2 ** (gateway_retries_used - 1))))
                        continue
                    if _is_transient_chat_error(exc):
                        transient_attempts += 1
                        if transient_attempts < max_transient_attempts:
                            continue
                        break
                    raise
        if self._can_use_provider_fallback(last_exc):
            yield from self._stream_with_fallback(messages, stop, run_manager, **kwargs)
            return
        if last_exc:
            raise last_exc


def build_model(streaming: bool = False) -> ChatOpenAI:
    load_env_files()
    keys = _chat_api_keys()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    base_kwargs: dict[str, Any] = dict(
        model=model_name,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0.2,
        streaming=streaming,
        timeout=float(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("CHAT_MODEL_MAX_RETRIES", "0")),
        max_tokens=int(os.getenv("CHAT_MODEL_MAX_TOKENS", "2048")),
    )
    disable_thinking = os.getenv("CHAT_MODEL_DISABLE_THINKING", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    if disable_thinking and "deepseek" in model_name.lower():
        # The main agent routes tools and converses; long-form research uses the
        # dedicated research writer/evidence paths. Provider-default reasoning
        # here can consume a large TPM allowance before a tool is even called.
        base_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    try:
        from .backend.research.model_runtime import (
            configured_preferred_provider_keys,
            configured_mimo_fallback_keys,
            mimo_fallback_base_url,
            mimo_fallback_enabled,
            mimo_fallback_extra_body,
            mimo_fallback_model_name,
            preferred_provider_base_url,
            preferred_provider_enabled,
            preferred_provider_extra_body,
            preferred_provider_gateway_managed,
            preferred_provider_gateway_retries,
            preferred_provider_key_acquire_timeout_seconds,
            preferred_provider_model_name,
            preferred_provider_per_key_concurrency,
            preferred_provider_rate_limit_cooldown_seconds,
        )
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.research.model_runtime import (
            configured_preferred_provider_keys,
            configured_mimo_fallback_keys,
            mimo_fallback_base_url,
            mimo_fallback_enabled,
            mimo_fallback_extra_body,
            mimo_fallback_model_name,
            preferred_provider_base_url,
            preferred_provider_enabled,
            preferred_provider_extra_body,
            preferred_provider_gateway_managed,
            preferred_provider_gateway_retries,
            preferred_provider_key_acquire_timeout_seconds,
            preferred_provider_model_name,
            preferred_provider_per_key_concurrency,
            preferred_provider_rate_limit_cooldown_seconds,
        )

    # The system unified/preferred provider is the authoritative user-facing model route when
    # configured. MODEL_NAME may intentionally mirror that route's model name for UI display,
    # while OPENAI_BASE_URL still points at a legacy provider; combining those two independent
    # settings sends a valid GLM model name to the wrong gateway and returns HTTP 404. Keep the
    # model, base URL and key pool atomic by selecting all three from the same provider channel.
    gateway_managed = False
    preferred_pool_kwargs: dict[str, Any] = {}
    if preferred_provider_enabled():
        keys = configured_preferred_provider_keys()
        model_name = preferred_provider_model_name()
        gateway_managed = preferred_provider_gateway_managed()
        preferred_pool_kwargs = {
            "per_key_concurrency": preferred_provider_per_key_concurrency(),
            "key_acquire_timeout_seconds": preferred_provider_key_acquire_timeout_seconds(),
            "rate_limit_cooldown_seconds": preferred_provider_rate_limit_cooldown_seconds(),
        }
        base_kwargs["model"] = model_name
        base_kwargs["base_url"] = preferred_provider_base_url()
        base_kwargs["default_headers"] = {
            "User-Agent": os.getenv("RESEARCH_PREFERRED_USER_AGENT", "curl/8.0").strip()
            or "curl/8.0"
        }
        preferred_extra_body = preferred_provider_extra_body()
        if preferred_extra_body:
            base_kwargs["extra_body"] = preferred_extra_body
        else:
            base_kwargs.pop("extra_body", None)
    fallback_keys = []
    fallback_kwargs: dict[str, Any] | None = None
    if mimo_fallback_enabled() and "api.xiaomimimo.com" not in str(base_kwargs.get("base_url") or "").lower():
        fallback_keys = configured_mimo_fallback_keys()
        fallback_kwargs = {
            **base_kwargs,
            "model": mimo_fallback_model_name(),
            "base_url": mimo_fallback_base_url(),
        }
        fallback_extra_body = mimo_fallback_extra_body()
        if fallback_extra_body:
            fallback_kwargs["extra_body"] = fallback_extra_body
        else:
            fallback_kwargs.pop("extra_body", None)
    if len(keys) > 1 or fallback_keys or gateway_managed:
        return RotatingChatOpenAI(
            api_keys=keys,
            fallback_api_keys=fallback_keys,
            fallback_kwargs=fallback_kwargs,
            gateway_managed=gateway_managed,
            gateway_retries=preferred_provider_gateway_retries() if gateway_managed else 0,
            **preferred_pool_kwargs,
            **base_kwargs,
        )
    return ChatOpenAI(api_key=keys[0] if keys else None, **base_kwargs)


def _is_transient_chat_error(exc: Exception) -> bool:
    message = str(exc).lower()
    error_type = type(exc).__name__.lower()
    return any(marker in f"{error_type} {message}" for marker in (
        "timeout", "timed out", "connection", "connecterror", "readerror",
        "remoteprotocolerror", "server disconnected", "temporarily unavailable",
    ))


def build_recall_model() -> ChatOpenAI:
    """Build the cheap model used only for long-term topic selection."""
    load_env_files()
    dedicated_key = (os.getenv("RECALL_API_KEY") or "").strip()
    # Recall is optional and already has a deterministic fallback in the
    # middleware. Without a dedicated credential it uses only the primary chat
    # key, rather than multiplying one user turn across every chat/research key.
    chat_keys = _chat_api_keys()
    keys = [dedicated_key] if dedicated_key else chat_keys[:1]
    model_name = os.getenv("RECALL_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini")
    kwargs: dict[str, Any] = dict(
        model=model_name,
        base_url=os.getenv("RECALL_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        streaming=False,
        timeout=float(os.getenv("RECALL_MODEL_TIMEOUT_SECONDS", "20")),
        max_retries=int(os.getenv("RECALL_MODEL_MAX_RETRIES") or os.getenv("CHAT_MODEL_MAX_RETRIES", "0")),
        max_tokens=int(os.getenv("RECALL_MODEL_MAX_TOKENS", "256")),
    )
    disable_thinking = os.getenv("RECALL_MODEL_DISABLE_THINKING", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }
    if disable_thinking and "deepseek" in model_name.lower():
        # Topic selection is a tiny JSON classification task. Letting DeepSeek's
        # provider-default reasoning run here has repeatedly consumed the whole
        # completion/TPM budget and turned one short Feishu message into minutes
        # of waiting before the real agent call even starts.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if len(keys) > 1:
        return RotatingChatOpenAI(api_keys=keys, **kwargs)
    return ChatOpenAI(api_key=keys[0] if keys else None, **kwargs)


def build_dream_model() -> ChatOpenAI:
    """Build the model used only for review-only AutoDream consolidation."""
    load_env_files()
    return ChatOpenAI(
        model=os.getenv("DREAM_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        api_key=os.getenv("DREAM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("DREAM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        streaming=False,
        timeout=float(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("CHAT_MODEL_MAX_RETRIES", "1")),
    )


def build_agent() -> Any:
    load_env_files()
    memory_store.ensure_memory_files()
    ensure_demo_files()
    ensure_knowledge_base()
    try:
        from .tools import skill_summaries
    except ImportError:  # pragma: no cover - script mode fallback
        from tools import skill_summaries

    model = build_model(streaming=True)
    recall_model = build_recall_model()
    surfaced_registry = memory_store.SurfacedMemoryRegistry()
    configure_surfaced_registry(surfaced_registry)
    set_current_thread_id(SESSION_ID)
    system_prompt = BASE_PROMPT.format(
        memory_index=memory_store.read_memory_index(),
        skill_summaries=skill_summaries(),
    )

    middleware = [
        FrustrationMiddleware(),
        MemoryWritePolicyMiddleware(),
        RelevantMemoryRecallMiddleware(
            recall_model=recall_model,
            surfaced_registry=surfaced_registry,
            max_topics=3,
        ),
        ContextSnipMiddleware(max_chars=18000),
        MicroCompactMiddleware(keep_recent=8),
        SkillSelectionMiddleware(),
        ContextCompactMiddleware(
            max_recent=8,
            trigger_at=16,
            model=recall_model,  # use cheap model for extraction
        ),
        ReactiveCompactMiddleware(keep_recent=8),
        LoopGuardMiddleware(max_repeats=2, max_failures=3),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "write_file writes/appends inside demo_workspace and requires review.",
                },
                "save_knowledge": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "save_knowledge writes a research document into knowledge_base and requires review.",
                },
                "save_knowledge_set": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "save_knowledge_set writes a document set into knowledge_base and requires review.",
                },
                "delete_knowledge_base_index": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "delete_knowledge_base_index deletes a local LightRAG index directory but keeps Markdown knowledge-base documents. knowledge_manager_worker_tool (workers.py) can only *propose* a deletion in its structured JSON return -- it has no delete_knowledge_base_index tool of its own -- so this gate must stay registered here and never be bypassed.",
                },
                "approve_research_job_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "approve_research_job_tool atomically archives a research job into the hierarchical personal knowledge system and requires review. This gate must never be bypassed.",
                },
                "save_single_url_brief_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "save_single_url_brief_tool commits a single_url_brief job's brief into knowledge_base (via the same internal knowledge-writing path as save_knowledge_set) and requires review. This gate must never be bypassed, even though the single_url generation step (start_single_url_brief_tool) itself is not gated.",
                },
                "confirm_knowledge_tree_change_tool": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Executes a previously generated knowledge-tree preview. The preview must be shown to the user before this confirmation step.",
                },
                "confirm_research_archive_tool": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Atomically archives a completed research job into the hierarchical personal knowledge system after the user reviews its location and file-change preview.",
                },
            },
            description_prefix="Deep Research HITL approval required",
        ),
    ]

    return create_agent(
        model=model,
        tools=[
            search_memory,
            load_memory_topic,
            write_memory_topic,
            list_skills,
            load_skill,
            read_file,
            write_file,
            query_knowledge_base,
            delete_knowledge_base_index,
            research_coverage_check,
            save_knowledge,
            save_knowledge_set,
            clarify_concept_tool,
            draft_research_plan_tool,
            get_latest_research_plan_tool,
            start_research_job_tool,
            check_research_job_tool,
            approve_research_job_tool,
            revise_report_section_tool,
            start_single_url_brief_tool,
            save_single_url_brief_tool,
            confirm_knowledge_tree_change_tool,
            confirm_research_archive_tool,
            repo_worker_tool,
            web_worker_tool,
            knowledge_manager_worker_tool,
        ],
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


def build_review_agent() -> Any:
    load_env_files()
    memory_store.ensure_memory_files()
    ensure_demo_files()
    review_model = build_dream_model()
    return create_agent(
        model=review_model,
        tools=[
            read_dream_report,
            read_memory_index,
            list_memory_topics,
            load_memory_topic,
            write_memory_topic,
            refresh_memory_index,
        ],
        system_prompt=REVIEW_AGENT_PROMPT,
    )


def build_kb_qa_agent() -> Any:
    """Build the dedicated, stateful knowledge-base QA agent.

    See docs/modules/kb-qa-channel-bypass-v0.1.md 4.2. Unlike the worker agents in workers.py
    (repo_worker_tool/knowledge_manager_worker_tool, all stateless/one-shot),
    this agent is persistent and carries multi-turn conversation memory via its own
    checkpointer, the same shape as build_agent() itself -- it is driven by
    channel_router_service.py::_route_qa() using the existing per-channel
    channel_service.thread_id_for(channel_id, "kb_qa") thread_id, not by this module's own
    SESSION_ID. Its tools are all read-only (query/list/inspect/summarize + switching the
    active topic), so unlike build_agent() it needs no HumanInTheLoopMiddleware.
    """
    load_env_files()
    memory_store.ensure_memory_files()
    ensure_demo_files()
    ensure_knowledge_base()
    return create_agent(
        model=build_model(),
        tools=[
            query_knowledge_base,
            set_active_knowledge_base,
            list_knowledge_bases,
            inspect_knowledge_base,
            summarize_knowledge_base,
        ],
        system_prompt=KB_QA_AGENT_PROMPT,
        checkpointer=InMemorySaver(),
    )


def load_env_files() -> None:
    """Load local .env files without printing or exposing secret values."""
    candidates = [
        ROOT / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "backend" / "backend" / ".env",
    ]
    try:
        from dotenv import load_dotenv

        for path in candidates:
            if path.exists():
                load_dotenv(path, override=False)
        _sanitize_proxy_env()
        return
    except ModuleNotFoundError:
        pass

    # Minimal fallback parser so the demo still works if python-dotenv is absent.
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _sanitize_proxy_env()


def _sanitize_proxy_env() -> None:
    """Clear only known-bad local proxy values that break model connectivity.

    We do not remove valid user-configured proxies. This targets the broken
    placeholder style we observed during local MVP testing.
    """
    bad_hosts = {
        "127.0.0.1:9",
        "localhost:9",
        "http://127.0.0.1:9",
        "http://localhost:9",
        "https://127.0.0.1:9",
        "https://localhost:9",
    }
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = (os.environ.get(key) or "").strip()
        if value in bad_hosts:
            os.environ.pop(key, None)


def _make_console_utf8_tolerant() -> None:
    """Reconfigure stdout/stderr to UTF-8 so CLI turns don't crash on emoji/CJK-extension
    characters the model chooses to output.

    On Windows, the console's default codepage is frequently GBK/936 (not UTF-8), and plain
    `print()` of characters outside that codepage (e.g. "✅") raises UnicodeEncodeError and
    kills the turn -- this was only discovered once the CLI entrypoint itself became reachable
    (see workers.py's lazy invoke_research_chat import fixing the pre-existing main.py<->backend
    circular import). `errors="replace"` degrades gracefully (prints "?") instead of crashing;
    it never changes what gets sent to the model, only what gets printed locally. No-op on
    platforms where stdout is already UTF-8 (Linux/macOS).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# Job statuses worth surfacing to a CLI user who wasn't watching when they landed. Keyed by
# job_id -> last status we already printed a notice for, so a job sitting in "running" for a
# while doesn't get re-announced every loop iteration, and a status genuinely changing (e.g.
# waiting_approval -> succeeded after a later approval from another surface) prints again.
_NOTIFY_JOB_STATUSES = {"waiting_approval", "succeeded", "failed", "cancelled"}
_seen_job_notifications: dict[str, str] = {}


def check_unread_job_notifications() -> None:
    """Print a short notice for any research job (tied to this CLI session) that reached a
    notify-worthy status since we last checked. This is "see it next time you show up", not a
    real push -- see docs/modules/agent-dispatcher-role-v0.1.md 3.6 for why CLI gets this
    instead of a live progress bridge (that's Web's/Feishu's mechanism, not CLI's).
    """
    try:
        jobs = list_conversation_jobs(thread_id=SESSION_ID, limit=10)
    except Exception:
        return
    for job in jobs:
        job_id = str(job.get("job_id") or "")
        status = str(job.get("status") or "")
        if not job_id or status not in _NOTIFY_JOB_STATUSES:
            continue
        if _seen_job_notifications.get(job_id) == status:
            continue
        _seen_job_notifications[job_id] = status
        request_text = str((job.get("input") or {}).get("user_request") or "").strip()
        label = f"（{request_text[:40]}）" if request_text else ""
        status_text = {
            "waiting_approval": "报告已生成，等待确认是否保存到知识库",
            "succeeded": "已完成",
            "failed": "执行失败",
            "cancelled": "已取消",
        }.get(status, status)
        print(f"\n[调研任务更新] {job_id}{label}：{status_text}")


def main() -> None:
    _make_console_utf8_tolerant()
    args = parse_args()
    agent = build_agent()
    config = {"configurable": {"thread_id": SESSION_ID}}
    set_observation_context(session_id=SESSION_ID, run_id=None)
    print("Deep Research ready. Commands: /memory, /dream, /exit")

    while True:
        check_unread_job_notifications()
        try:
            user_text = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return
        if not user_text:
            continue
        if user_text == "/exit":
            print("bye")
            return
        if user_text == "/memory":
            print(memory_store.read_memory_index())
            continue
        if user_text == "/dream":
            run_id = new_run_id()
            set_observation_context(session_id=SESSION_ID, run_id=run_id)
            report = memory_store.run_autodream(dream_model=build_dream_model())
            print(f"AutoDream report generated: {report}")
            set_observation_context(session_id=SESSION_ID, run_id=None)
            continue

        try:
            run_id = new_run_id()
            set_observation_context(session_id=SESSION_ID, run_id=run_id)
            emit_event(
                "agent.turn.started",
                "main",
                "User turn started.",
                data={"user_chars": len(user_text)},
            )
            payload = {"messages": [{"role": "user", "content": user_text}]}
            print("\nDeep Research> ", end="", flush=True)
            final_text = stream_with_hitl(agent, payload, config, debug=args.debug)
            print()
            memory_store.append_daily_log_turn(SESSION_ID, user_text, final_text)
            emit_event(
                "agent.turn.finished",
                "main",
                "User turn finished.",
                data={"assistant_chars": len(final_text)},
            )
        except Exception as exc:
            emit_event(
                "agent.turn.failed",
                "main",
                "User turn failed.",
                level="error",
                data={"error": str(exc)},
            )
            print(f"\nDeep Research error: {exc}")
        finally:
            set_observation_context(session_id=SESSION_ID, run_id=None)


def stream_with_hitl(
    agent: Any,
    payload: dict[str, Any] | Command,
    config: dict[str, Any],
    *,
    debug: bool = False,
) -> str:
    """Stream agent tokens and handle HITL interrupts in a CLI loop."""
    next_payload: dict[str, Any] | Command = payload
    accumulated_text = ""
    _last_msg_type = ""
    while True:
        interrupted = False
        interrupts: list[Any] = []
        _last_msg_type = ""

        for chunk in agent.stream(
            next_payload,
            config=config,
            stream_mode=["updates", "messages"],
            version="v2",
        ):
            chunk_type, data = normalize_stream_chunk(chunk)

            if chunk_type == "messages":
                msg_type, token_text = token_content_typed(data)
                if not token_text:
                    continue

                if debug:
                    # Debug mode: print all message types with icons
                    if msg_type != _last_msg_type:
                        if _last_msg_type != "":
                            print()  # newline to separate from previous content
                        icon = _DEBUG_ICONS.get(msg_type, f"[{msg_type}] ")
                        print(icon, end="", flush=True)
                        _last_msg_type = msg_type
                    print(token_text, end="", flush=True)
                continue

            if chunk_type == "updates":
                update_interrupts = extract_interrupts(data)
                if update_interrupts:
                    interrupts.extend(update_interrupts)
                    interrupted = True
                    emit_event(
                        "hitl.interrupt.created",
                        "main",
                        "HITL interrupt was created.",
                        data={"interrupt_count": len(update_interrupts)},
                    )
                else:
                    maybe_text = latest_update_ai_text(data)
                    if maybe_text:
                        accumulated_text = maybe_text

        # Clean mode: print final AI response from updates stream
        if not debug and accumulated_text:
            print(accumulated_text)

        if not interrupts:
            return accumulated_text.strip()

        if interrupted:
            print()
        decisions = []
        for interrupt in interrupts:
            print("\n[HITL interrupt]")
            print(format_interrupt(interrupt))
            decision_count = interrupt_decision_count(interrupt)
            for index in range(decision_count):
                if decision_count > 1:
                    print(f"Decision {index + 1}/{decision_count}")
                decision = read_decision()
                decision_type = str(decision.get("type", "unknown"))
                event_suffix = {
                    "approve": "approved",
                    "reject": "rejected",
                    "edit": "edited",
                }.get(decision_type, decision_type)
                emit_event(
                    f"hitl.decision.{event_suffix}",
                    "main",
                    "Human reviewer provided a HITL decision.",
                    data={"decision_type": decision_type},
                )
                decisions.append(decision)
        print("\nDeep Research> ", end="", flush=True)
        next_payload = Command(resume={"decisions": decisions})


def normalize_stream_chunk(chunk: Any) -> tuple[str | None, Any]:
    """Support both v2 dict chunks and tuple chunks from multi stream modes."""
    if isinstance(chunk, dict) and "type" in chunk:
        return chunk.get("type"), chunk.get("data")
    if isinstance(chunk, tuple) and len(chunk) == 2:
        return str(chunk[0]), chunk[1]
    if isinstance(chunk, dict):
        if "__interrupt__" in chunk:
            return "updates", chunk
        if "messages" in chunk:
            return "updates", chunk
    return None, chunk


def token_content(data: Any) -> str:
    """Extract token text from a v2 messages chunk."""
    if not isinstance(data, tuple) or not data:
        return ""
    token = data[0]
    content = getattr(token, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "".join(pieces)
    return ""


def token_content_typed(data: Any) -> tuple[str, str]:
    """Extract token text and message type from a v2 messages chunk.

    Returns (msg_type, content). msg_type is one of:
    "ai", "system", "human", "tool", or "".
    """
    if not isinstance(data, tuple) or not data:
        return "", ""
    token = data[0]
    msg_type = getattr(token, "type", "") or ""

    content = getattr(token, "content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        text = "".join(pieces)
    else:
        text = ""

    return msg_type, text


def latest_update_ai_text(data: Any) -> str:
    """Best-effort final text extraction from updates when token streaming is sparse."""
    if not isinstance(data, dict):
        return ""
    for node_data in data.values():
        if not isinstance(node_data, dict):
            continue
        messages = node_data.get("messages") or []
        for message in reversed(messages):
            content = getattr(message, "content", None)
            msg_type = getattr(message, "type", "")
            if content and msg_type in {"ai", "assistant", ""}:
                return str(content)
    return ""


# Tool names whose *result* a channel-scoped caller (chat_runtime_service.py::main_chat_handler)
# needs to see even though the decision to call them now lives entirely inside the Agent's own
# reasoning rather than a router-layer hard route -- see
# docs/modules/agent-dispatcher-role-v0.1.md 3.7/3.8. Feishu's existing progress/revision
# bridges (backend/services/feishu_service.py::_start_research_progress_bridge /
# _start_section_revision_bridge) key off route_channel_message's returned `status` field
# ("research_started" / "section_revision_started"); extract_tool_results is how that status
# gets reconstructed post-hoc from the Agent's tool-call trace instead of a route-time branch.
DISPATCH_BRIDGE_TOOL_NAMES = {"start_research_job_tool", "revise_report_section_tool", "start_single_url_brief_tool"}


def extract_tool_results(data: Any, tool_names: set[str]) -> dict[str, dict[str, Any]]:
    """Best-effort extraction of JSON tool-call results from an updates chunk, keyed by tool
    name, limited to `tool_names`. Only tools in this codebase that return a JSON string are
    usable here (all four research-job dispatch tools do)."""
    found: dict[str, dict[str, Any]] = {}
    if not isinstance(data, dict):
        return found
    for node_data in data.values():
        if not isinstance(node_data, dict):
            continue
        for message in node_data.get("messages") or []:
            if getattr(message, "type", "") != "tool":
                continue
            name = getattr(message, "name", "")
            if name not in tool_names:
                continue
            try:
                payload = json.loads(str(getattr(message, "content", "") or ""))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                found[name] = payload
    return found


def extract_interrupts(result: Any) -> list[Any]:
    graph_interrupts = getattr(result, "interrupts", None)
    if graph_interrupts:
        return list(graph_interrupts)
    if not isinstance(result, dict):
        return []
    for key in ("__interrupt__", "interrupts"):
        value = result.get(key)
        if value:
            return list(value if isinstance(value, (list, tuple)) else [value])
    return []


def format_interrupt(interrupt: Any) -> str:
    value = getattr(interrupt, "value", interrupt)
    return str(value)


def interrupt_decision_count(interrupt: Any) -> int:
    """Return how many HITL decisions are required for an interrupt."""
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return interrupt_decision_count(value[0])
    if isinstance(value, dict):
        requests = value.get("action_requests") or value.get("actions") or []
        if isinstance(requests, list) and requests:
            return len(requests)
    return 1


def read_decision() -> dict[str, Any]:
    while True:
        raw = input("Approve, edit, or reject? [a/e/r] ").strip().lower()
        if raw in {"a", "approve"}:
            return {"type": "approve"}
        if raw in {"r", "reject"}:
            feedback = input("Rejection feedback: ").strip() or "Rejected by human reviewer."
            return {"type": "reject", "message": feedback}
        if raw in {"e", "edit"}:
            path = input("Edited path under demo_workspace: ").strip()
            content = input("Edited content: ")
            return {
                "type": "edit",
                "edited_action": {
                    "name": "write_file",
                    "args": {"mode": "write", "path": path, "content": content},
                },
            }
        print("Please enter a, e, or r.")


def final_message_text(result: Any) -> str:
    value = getattr(result, "value", result)
    messages = value.get("messages", []) if isinstance(value, dict) else []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        msg_type = getattr(message, "type", "")
        if content and msg_type in {"ai", "assistant", ""}:
            return str(content)
    return str(result)


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"Missing dependency: {missing}\n"
            "Install LangChain v1 dependencies, for example:\n"
            "  pip install 'langchain>=1.0.0' 'langchain-openai>=0.3.0' 'langgraph>=1.0.0'",
            file=sys.stderr,
        )
        raise
