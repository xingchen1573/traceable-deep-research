"""Shared, secret-safe model credential scheduling for research calls."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from langchain_openai import ChatOpenAI

from .audit import invoke_with_model_trace

ResearchRole = Literal["evidence", "writer", "ranker"]


class ProviderFallbackExhaustedError(RuntimeError):
    def __init__(self, primary_error: Exception | None, fallback_error: Exception) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        primary_name = type(primary_error).__name__ if primary_error is not None else "forced_fallback"
        super().__init__(
            f"provider fallback exhausted after {primary_name}: "
            f"{type(fallback_error).__name__}: {fallback_error}"
        )


@dataclass
class _CredentialState:
    api_key: str
    index: int
    active: int = 0
    calls: int = 0
    failures: int = 0
    rate_limits: int = 0
    cooldown_until: float = 0.0
    disabled: bool = False
    last_error: str = ""


@dataclass
class ResearchModelPool:
    api_keys: list[str]
    per_key_concurrency: int = 1
    acquire_timeout_seconds: float = 30.0
    rate_limit_cooldown_seconds: float = 20.0
    gateway_managed: bool = False
    _states: list[_CredentialState] = field(init=False, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        unique = list(dict.fromkeys(key.strip() for key in self.api_keys if key.strip()))
        if not unique:
            raise ValueError("research model pool requires at least one API key")
        self.api_keys = unique
        self.per_key_concurrency = max(1, self.per_key_concurrency)
        self._states = [_CredentialState(api_key=key, index=index) for index, key in enumerate(unique)]
        self._condition = threading.Condition()

    @contextmanager
    def credential(self, *, timeout_seconds: float | None = None) -> Iterator[_CredentialState]:
        state = self._acquire(timeout_seconds=timeout_seconds)
        try:
            yield state
        finally:
            with self._condition:
                state.active = max(0, state.active - 1)
                self._condition.notify_all()

    def _acquire(self, *, timeout_seconds: float | None = None) -> _CredentialState:
        acquire_timeout = self.acquire_timeout_seconds if timeout_seconds is None else max(0.0, timeout_seconds)
        deadline = time.monotonic() + acquire_timeout
        with self._condition:
            while True:
                now = time.monotonic()
                if all(state.disabled for state in self._states):
                    raise RuntimeError("all research model credentials are unavailable")
                for offset in range(len(self._states)):
                    index = (self._cursor + offset) % len(self._states)
                    state = self._states[index]
                    if not state.disabled and state.active < self.per_key_concurrency and state.cooldown_until <= now:
                        state.active += 1
                        state.calls += 1
                        self._cursor = (index + 1) % len(self._states)
                        return state
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError("research model credential pool is saturated")
                cooldowns = [
                    state.cooldown_until - now
                    for state in self._states
                    if not state.disabled and state.cooldown_until > now
                ]
                wait_for = min([remaining, 0.25, *cooldowns]) if cooldowns else min(remaining, 0.25)
                self._condition.wait(timeout=max(0.01, wait_for))

    def record_success(self, state: _CredentialState) -> None:
        with self._condition:
            state.last_error = ""

    def record_failure(self, state: _CredentialState, exc: Exception) -> bool:
        message = str(exc).lower()
        rate_limited = "429" in message or "rpm exhausted" in message or "rate limit" in message
        quota_exhausted = "insufficient_quota" in message or "allocated quota exceeded" in message
        with self._condition:
            state.failures += 1
            state.last_error = "quota_exhausted" if quota_exhausted else "rate_limited" if rate_limited else type(exc).__name__
            if quota_exhausted and not self.gateway_managed:
                # A balance/account quota error is not a short TPM/RPM burst. Reusing the same
                # credential every cooldown interval only spends retries on a request that
                # cannot recover without external account changes. A process restart rebuilds
                # the pool after the operator updates credentials or quota.
                state.disabled = True
            elif rate_limited or quota_exhausted:
                state.rate_limits += 1
                # A managed gateway (for example gpt-load) owns the real upstream key ring.
                # The local credential identifies the gateway, not the exhausted upstream key,
                # so disabling it would prevent the gateway from rotating the next request to a
                # healthy key. Gateway cooldown defaults to zero for the same reason.
                if not self.gateway_managed:
                    state.cooldown_until = time.monotonic() + self.rate_limit_cooldown_seconds
            self._condition.notify_all()
        return rate_limited or (quota_exhausted and self.gateway_managed)

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            return {
                "key_count": len(self._states),
                "per_key_concurrency": self.per_key_concurrency,
                "gateway_managed": self.gateway_managed,
                "active_count": sum(state.active for state in self._states),
                "call_count": sum(state.calls for state in self._states),
                "failure_count": sum(state.failures for state in self._states),
                "rate_limit_count": sum(state.rate_limits for state in self._states),
                "keys": [{
                    "key_index": state.index,
                    "active": state.active,
                    "calls": state.calls,
                    "failures": state.failures,
                    "rate_limits": state.rate_limits,
                    "disabled": state.disabled,
                    "cooldown_seconds": round(max(0.0, state.cooldown_until - now), 3),
                    "last_error": state.last_error,
                } for state in self._states],
            }


_POOL_LOCK = threading.Lock()
# Keyed by role ("evidence"/"writer") -- see docs/modules/model-config-completion-v0.1.md 3.4.
# Each role gets its own pool/signature so evidence extraction and writing can use fully
# independent providers (base_url + model + key set) without interfering with each other.
_POOL: dict[str, ResearchModelPool] = {}
_POOL_SIGNATURE: dict[str, tuple[Any, ...]] = {}
_PREFERRED_PROVIDER_POOL: ResearchModelPool | None = None
_PREFERRED_PROVIDER_SIGNATURE: tuple[Any, ...] | None = None
_MIMO_FALLBACK_POOL: ResearchModelPool | None = None
_MIMO_FALLBACK_SIGNATURE: tuple[Any, ...] | None = None
_PROVIDER_FALLBACK_LOCK = threading.Lock()
_PROVIDER_FALLBACK_DIAGNOSTICS: dict[str, Any] = {
    "triggered_count": 0,
    "success_count": 0,
    "failure_count": 0,
    "trigger_reasons": {},
    "stages": {},
}


def research_model_pool(role: ResearchRole = "evidence") -> ResearchModelPool:
    global _POOL, _POOL_SIGNATURE
    keys = configured_keys(role)
    # `ranker`'s acquire timeout is intentionally split from the shared
    # RESEARCH_MODEL_KEY_ACQUIRE_TIMEOUT_SECONDS (still used by evidence/writer): a real job
    # (case-ranker-concurrency) showed 7-8 questions firing a ranker call concurrently against
    # only 4 keys at per_key_concurrency=1, so tail requests queue for 0-22s AND then see 10-22s
    # of real model latency on top -- the combined wait routinely landed in the 30-40s range and
    # tripped the shared 30s default. Raising the shared constant would have also loosened the
    # acquire ceiling for the evidence-extraction and report-writing paths, which is a much
    # bigger blast radius than intended for a diagnostic-only (shadow-mode) scoring pass -- see
    # project_source_pool_llm_boost_ranker_timeout_investigation_20260801.
    acquire_timeout_key = "RESEARCH_RANKER_ACQUIRE_TIMEOUT_SECONDS" if role == "ranker" else "RESEARCH_MODEL_KEY_ACQUIRE_TIMEOUT_SECONDS"
    acquire_timeout_default = 90.0 if role == "ranker" else 30.0
    signature = (
        tuple(keys),
        _env_int("RESEARCH_MODEL_PER_KEY_CONCURRENCY", 1),
        _env_float(acquire_timeout_key, acquire_timeout_default),
        _env_float("RESEARCH_MODEL_RATE_LIMIT_COOLDOWN_SECONDS", 20.0),
    )
    with _POOL_LOCK:
        if _POOL.get(role) is None or signature != _POOL_SIGNATURE.get(role):
            _POOL[role] = ResearchModelPool(
                api_keys=keys,
                per_key_concurrency=signature[1],
                acquire_timeout_seconds=signature[2],
                rate_limit_cooldown_seconds=signature[3],
            )
            _POOL_SIGNATURE[role] = signature
        return _POOL[role]


def configured_preferred_provider_keys() -> list[str]:
    if unified_provider_enabled():
        return configured_unified_provider_keys()
    return _configured_legacy_preferred_provider_keys()


def configured_unified_provider_keys() -> list[str]:
    primary = os.getenv("RESEARCH_UNIFIED_MODEL_API_KEY", "").strip()
    extras = [item.strip() for item in os.getenv("RESEARCH_UNIFIED_MODEL_EXTRA_API_KEYS", "").split(",")]
    return list(dict.fromkeys(item for item in [primary, *extras] if item))


def unified_provider_enabled() -> bool:
    return _env_bool("RESEARCH_UNIFIED_MODEL_CHANNEL_ENABLED", False) and bool(configured_unified_provider_keys())


def _configured_legacy_preferred_provider_keys() -> list[str]:
    primary = os.getenv("RESEARCH_PREFERRED_API_KEY", "").strip()
    extras = [item.strip() for item in os.getenv("RESEARCH_PREFERRED_EXTRA_API_KEYS", "").split(",")]
    return list(dict.fromkeys(item for item in [primary, *extras] if item))


def preferred_provider_enabled() -> bool:
    return unified_provider_enabled() or (
        _env_bool("RESEARCH_PREFERRED_PROVIDER_ENABLED", False)
        and bool(_configured_legacy_preferred_provider_keys())
    )


def preferred_provider_model_name() -> str:
    if unified_provider_enabled():
        return os.getenv("RESEARCH_UNIFIED_MODEL_NAME", "mimo-v2.5").strip() or "mimo-v2.5"
    return os.getenv("RESEARCH_PREFERRED_MODEL_NAME", "mimo-v2.5").strip() or "mimo-v2.5"


def preferred_provider_base_url() -> str | None:
    if unified_provider_enabled():
        return os.getenv("RESEARCH_UNIFIED_MODEL_BASE_URL", "").strip() or None
    return os.getenv("RESEARCH_PREFERRED_BASE_URL", "").strip() or None


def preferred_provider_extra_body() -> dict[str, Any] | None:
    if _env_bool("RESEARCH_PREFERRED_DISABLE_THINKING", True):
        return {"thinking": {"type": "disabled"}}
    return None


def preferred_provider_gateway_managed() -> bool:
    """Whether the preferred endpoint owns upstream key rotation and concurrency.

    In this mode the locally configured API key authenticates one gateway rather than one
    upstream model account. Local quota-disable and low per-key concurrency semantics would be
    incorrect because the gateway selects the actual upstream key for every request.
    """

    return _env_bool("RESEARCH_PREFERRED_GATEWAY_MANAGED", False)


def preferred_provider_per_key_concurrency() -> int:
    """Return the shared concurrency policy for the preferred route.

    A managed gateway credential fronts its own upstream key pool, so serializing all
    local calls behind that one credential defeats the gateway's purpose. Direct-provider
    credentials retain the conservative one-call default.
    """

    default = 64 if preferred_provider_gateway_managed() else 1
    return max(1, _env_int("RESEARCH_PREFERRED_PER_KEY_CONCURRENCY", default))


def preferred_provider_key_acquire_timeout_seconds() -> float:
    """Return the shared credential-pool wait budget for the preferred route."""

    default = 300.0 if preferred_provider_gateway_managed() else 30.0
    return max(0.0, _env_float("RESEARCH_PREFERRED_KEY_ACQUIRE_TIMEOUT_SECONDS", default))


def preferred_provider_rate_limit_cooldown_seconds() -> float:
    """Return the local cooldown policy for the preferred route."""

    if preferred_provider_gateway_managed():
        return 0.0
    return max(0.0, _env_float("RESEARCH_PREFERRED_RATE_LIMIT_COOLDOWN_SECONDS", 20.0))


def preferred_provider_gateway_retries() -> int:
    """Return the one authoritative managed-gateway retry budget."""

    if not preferred_provider_gateway_managed():
        return 0
    return max(0, _env_int("RESEARCH_PREFERRED_GATEWAY_RETRIES", 2))


def _mimo_default_headers(env_name: str) -> dict[str, str]:
    return {"User-Agent": os.getenv(env_name, "curl/8.0").strip() or "curl/8.0"}


def preferred_provider_model_pool() -> ResearchModelPool:
    global _PREFERRED_PROVIDER_POOL, _PREFERRED_PROVIDER_SIGNATURE
    keys = configured_preferred_provider_keys()
    gateway_managed = preferred_provider_gateway_managed()
    signature = (
        tuple(keys),
        preferred_provider_per_key_concurrency(),
        preferred_provider_key_acquire_timeout_seconds(),
        preferred_provider_rate_limit_cooldown_seconds(),
        gateway_managed,
    )
    with _POOL_LOCK:
        if _PREFERRED_PROVIDER_POOL is None or signature != _PREFERRED_PROVIDER_SIGNATURE:
            _PREFERRED_PROVIDER_POOL = ResearchModelPool(
                api_keys=keys,
                per_key_concurrency=signature[1],
                acquire_timeout_seconds=signature[2],
                rate_limit_cooldown_seconds=signature[3],
                gateway_managed=signature[4],
            )
            _PREFERRED_PROVIDER_SIGNATURE = signature
        return _PREFERRED_PROVIDER_POOL


def configured_mimo_fallback_keys() -> list[str]:
    primary = os.getenv("MIMO_API_KEY", "").strip()
    extras = [item.strip() for item in os.getenv("MIMO_EXTRA_API_KEYS", "").split(",")]
    return list(dict.fromkeys(item for item in [primary, *extras] if item))


def mimo_fallback_enabled() -> bool:
    return _env_bool("MIMO_FALLBACK_ENABLED", False) and bool(configured_mimo_fallback_keys())


def mimo_fallback_model_name() -> str:
    return os.getenv("MIMO_MODEL_NAME", "mimo-v2.5").strip() or "mimo-v2.5"


def mimo_fallback_base_url() -> str:
    return os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1").strip()


def mimo_fallback_extra_body() -> dict[str, Any] | None:
    if _env_bool("MIMO_FALLBACK_DISABLE_THINKING", True):
        return {"thinking": {"type": "disabled"}}
    return None


def mimo_fallback_model_pool() -> ResearchModelPool:
    global _MIMO_FALLBACK_POOL, _MIMO_FALLBACK_SIGNATURE
    keys = configured_mimo_fallback_keys()
    signature = (
        tuple(keys),
        _env_int("MIMO_FALLBACK_PER_KEY_CONCURRENCY", 1),
        _env_float("MIMO_FALLBACK_KEY_ACQUIRE_TIMEOUT_SECONDS", 30.0),
        _env_float("MIMO_FALLBACK_RATE_LIMIT_COOLDOWN_SECONDS", 20.0),
    )
    with _POOL_LOCK:
        if _MIMO_FALLBACK_POOL is None or signature != _MIMO_FALLBACK_SIGNATURE:
            _MIMO_FALLBACK_POOL = ResearchModelPool(
                api_keys=keys,
                per_key_concurrency=signature[1],
                acquire_timeout_seconds=signature[2],
                rate_limit_cooldown_seconds=signature[3],
            )
            _MIMO_FALLBACK_SIGNATURE = signature
        return _MIMO_FALLBACK_POOL


def should_use_provider_fallback(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in ("401", "403", "authentication", "permission denied", "invalid api key")):
        return False
    return any(marker in text for marker in (
        "429",
        "rate limit",
        "rate_limited",
        "rpm exhausted",
        "timeout",
        "timed out",
        "connecterror",
        "connectionerror",
        "connection reset",
        "connection aborted",
        "remoteprotocolerror",
        "server disconnected",
        "eof occurred",
    ))


def _is_transient_connection_error(exc: Exception) -> bool:
    """Return whether an upstream call failed before a model response was available.

    These failures normally complete in milliseconds and are safe to retry once through the
    existing credential pool. Keep this narrower than timeouts and provider status errors so the
    shared runtime does not silently multiply expensive model attempts.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in (
        "apiconnectionerror",
        "connecterror",
        "connectionerror",
        "connection error",
        "connection reset",
        "connection aborted",
        "remoteprotocolerror",
        "server disconnected",
        "eof occurred",
    ))


def _is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in (
        "timeout",
        "timed out",
        "deadline expired",
    ))


def _is_invalid_provider_response_error(exc: Exception) -> bool:
    """Recognize the known OpenAI-compatible response-shape failure.

    The OpenAI SDK can return raw text when an upstream gateway responds with a
    non-JSON body.  ChatOpenAI then calls ``model_dump`` on that string before
    our JSON parsing layer can inspect it.  Keep this check deliberately exact
    so unrelated AttributeErrors are not retried as provider failures.
    """

    return (
        isinstance(exc, AttributeError)
        and str(exc).strip() == "'str' object has no attribute 'model_dump'"
    )


def _is_retryable_gateway_error(exc: Exception) -> bool:
    """Return whether a fresh gateway request may be routed to another upstream key."""

    if _is_invalid_provider_response_error(exc):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in (
        "401",
        "403",
        "authentication",
        "permission denied",
        "invalid api key",
        "bad request",
    )):
        return False
    # A direct provider normally uses 404 to report a genuinely invalid model name. A managed
    # gateway can also return this exact error transiently when the selected upstream route/key
    # does not currently expose the requested model. Retrying is only enabled by callers with a
    # positive gateway retry budget, so direct-provider calls still fail fast and a truly invalid
    # model remains bounded by that small budget.
    if any(marker in text for marker in (
        "model is not found",
        "model not found",
        "model route not found",
    )) and ("404" in text or "notfounderror" in text or "not_found_error" in text):
        return True
    return should_use_provider_fallback(exc) or any(marker in text for marker in (
        "500",
        "502",
        "503",
        "504",
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ))


def record_provider_fallback(stage: str, outcome: str, reason: str = "") -> None:
    stage_name = str(stage or "unknown")
    reason_name = str(reason or "unknown")
    with _PROVIDER_FALLBACK_LOCK:
        if outcome == "triggered":
            _PROVIDER_FALLBACK_DIAGNOSTICS["triggered_count"] += 1
            reasons = _PROVIDER_FALLBACK_DIAGNOSTICS["trigger_reasons"]
            reasons[reason_name] = int(reasons.get(reason_name) or 0) + 1
        elif outcome == "success":
            _PROVIDER_FALLBACK_DIAGNOSTICS["success_count"] += 1
        elif outcome == "failure":
            _PROVIDER_FALLBACK_DIAGNOSTICS["failure_count"] += 1
        stages = _PROVIDER_FALLBACK_DIAGNOSTICS["stages"]
        stage_counts = stages.setdefault(stage_name, {"triggered": 0, "success": 0, "failure": 0})
        if outcome in stage_counts:
            stage_counts[outcome] += 1


def provider_fallback_diagnostics() -> dict[str, Any]:
    with _PROVIDER_FALLBACK_LOCK:
        return {
            "enabled": mimo_fallback_enabled(),
            "fallback_model": mimo_fallback_model_name(),
            "key_count": len(configured_mimo_fallback_keys()),
            "triggered_count": int(_PROVIDER_FALLBACK_DIAGNOSTICS["triggered_count"]),
            "success_count": int(_PROVIDER_FALLBACK_DIAGNOSTICS["success_count"]),
            "failure_count": int(_PROVIDER_FALLBACK_DIAGNOSTICS["failure_count"]),
            "trigger_reasons": dict(_PROVIDER_FALLBACK_DIAGNOSTICS["trigger_reasons"]),
            "stages": {
                key: dict(value) for key, value in _PROVIDER_FALLBACK_DIAGNOSTICS["stages"].items()
            },
        }


def invoke_preferred_chat(
    messages: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    temperature: float = 0,
    system_prompt: str | None = None,
    deadline_monotonic: float | None = None,
    parent_call_id: str | None = None,
    call_kind: str = "model",
) -> Any:
    """Invoke the configured preferred provider without silently changing providers.

    Knowledge Q&A uses this entry point whenever the preferred/unified route is the
    authoritative user-facing route. Keeping this call strict is important: a failed
    request must remain visible as a preferred-route failure instead of quietly falling
    through to the research writer/evidence endpoint.
    """

    if not preferred_provider_enabled():
        raise RuntimeError("preferred research model provider is not configured")
    chat_messages = [{"role": "system", "content": system_prompt}, *messages] if system_prompt else messages
    response = _invoke_provider_chat(
        chat_messages,
        pool=preferred_provider_model_pool(),
        model_name=preferred_provider_model_name(),
        base_url=preferred_provider_base_url(),
        timeout_seconds=timeout_seconds,
        max_rate_limit_retries=_env_int(
            "RESEARCH_PREFERRED_RATE_LIMIT_RETRIES",
            0 if preferred_provider_gateway_managed() else 1,
        ),
        max_connection_retries=_env_int(
            "RESEARCH_PREFERRED_CONNECTION_RETRIES",
            0 if preferred_provider_gateway_managed() else 1,
        ),
        max_timeout_retries=_env_int("RESEARCH_PREFERRED_TIMEOUT_RETRIES", 0),
        max_gateway_retries=preferred_provider_gateway_retries(),
        temperature=temperature,
        deadline_monotonic=deadline_monotonic,
        extra_body=preferred_provider_extra_body(),
        max_tokens=None,
        default_headers=_mimo_default_headers("RESEARCH_PREFERRED_USER_AGENT"),
        response_format_json=False,
        provider_name="preferred",
        parent_call_id=parent_call_id,
        call_kind=call_kind,
    )
    return _mark_response_provider(response, "preferred")


def invoke_research_chat(
    messages: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    evidence: bool = False,
    role: ResearchRole | None = None,
    max_rate_limit_retries: int | None = None,
    max_connection_retries: int | None = None,
    temperature: float = 0,
    system_prompt: str | None = None,
    model_name: str | None = None,
    deadline_monotonic: float | None = None,
    allow_provider_fallback: bool = False,
    force_fallback_provider: bool = False,
    skip_preferred_provider: bool = False,
    fallback_stage: str = "research",
    provider_fallback_reason: str = "",
    parent_call_id: str | None = None,
    call_kind: str = "model",
) -> Any:
    """Invoke the shared research model pool.

    ``temperature``/``system_prompt``/``model_name`` default to the historical
    deterministic, no-system-prompt behavior so existing callers (evidence
    extraction, judging, entailment) are unaffected. Only callers that need
    long-form writing (the section writer/editorial passes) should override
    them.

    ``evidence`` selects which independently-configurable role's pool/base_url/model
    this call uses (``True`` -> "evidence", ``False`` -> "writer"). See
    docs/modules/model-config-completion-v0.1.md 3.4 for the fallback rules: an
    unconfigured writer role transparently reuses the evidence role's keys/base_url,
    so deployments that never set ``RESEARCH_WRITER_*`` see no behavior change.

    ``role``, when given, takes precedence over ``evidence`` and selects any
    ``ResearchRole`` directly (e.g. ``"ranker"`` for the source-candidate LLM boost) --
    added so a third role doesn't need its own boolean flag threaded through every
    existing call site.
    """
    resolved_role: ResearchRole = role if role is not None else ("evidence" if evidence else "writer")
    retries = max_rate_limit_retries
    if retries is None:
        retries = _env_int("RESEARCH_MODEL_RATE_LIMIT_RETRIES", 1)
    connection_retries = max_connection_retries
    if connection_retries is None:
        connection_retries = _env_int("RESEARCH_MODEL_CONNECTION_RETRIES", 1)
    chat_messages = [{"role": "system", "content": system_prompt}, *messages] if system_prompt else messages

    preferred_error: Exception | None = None
    if preferred_provider_enabled() and not force_fallback_provider and not skip_preferred_provider:
        try:
            response = invoke_preferred_chat(
                messages,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                system_prompt=system_prompt,
                deadline_monotonic=deadline_monotonic,
                parent_call_id=parent_call_id,
                call_kind=call_kind,
            )
        except Exception as exc:
            preferred_error = exc
        else:
            return response

    if force_fallback_provider:
        if not mimo_fallback_enabled():
            raise RuntimeError("MiMo fallback provider is not configured")
        reason = provider_fallback_reason or "forced_after_invalid_response"
        record_provider_fallback(fallback_stage, "triggered", reason)
        try:
            response = _invoke_provider_chat(
                chat_messages,
                pool=mimo_fallback_model_pool(),
                model_name=mimo_fallback_model_name(),
                base_url=mimo_fallback_base_url(),
                timeout_seconds=timeout_seconds,
                max_rate_limit_retries=_env_int("MIMO_FALLBACK_RATE_LIMIT_RETRIES", 1),
                max_connection_retries=_env_int("MIMO_FALLBACK_CONNECTION_RETRIES", 1),
                max_timeout_retries=0,
                max_gateway_retries=0,
                temperature=temperature,
                deadline_monotonic=deadline_monotonic,
                extra_body=mimo_fallback_extra_body(),
                max_tokens=None,
                default_headers=_mimo_default_headers("MIMO_FALLBACK_USER_AGENT"),
                response_format_json=True,
                provider_name="mimo_fallback",
                parent_call_id=parent_call_id,
                call_kind="fallback" if call_kind == "model" else call_kind,
            )
        except Exception as fallback_exc:
            record_provider_fallback(fallback_stage, "failure")
            raise ProviderFallbackExhaustedError(None, fallback_exc) from fallback_exc
        record_provider_fallback(fallback_stage, "success")
        return _mark_response_provider(response, "mimo_fallback")

    pool = research_model_pool(resolved_role)
    extra_body: dict[str, Any] | None = None
    if resolved_role == "evidence" and _env_bool("RESEARCH_EVIDENCE_DISABLE_THINKING", True):
        extra_body = {"thinking": {"type": "disabled"}}
    try:
        response = _invoke_provider_chat(
            chat_messages,
            pool=pool,
            model_name=model_name or research_role_model_name(resolved_role),
            base_url=research_role_base_url(resolved_role),
            timeout_seconds=timeout_seconds,
            max_rate_limit_retries=retries,
            max_connection_retries=connection_retries,
            max_timeout_retries=0,
            max_gateway_retries=0,
            temperature=temperature,
            deadline_monotonic=deadline_monotonic,
            extra_body=extra_body,
            max_tokens=None,
            default_headers=None,
            response_format_json=False,
            provider_name=f"primary:{resolved_role}",
            parent_call_id=parent_call_id,
            call_kind=call_kind,
        )
        return _mark_response_provider(response, "primary")
    except Exception as exc:
        if not allow_provider_fallback or not mimo_fallback_enabled() or not should_use_provider_fallback(exc):
            if preferred_error is not None:
                raise ProviderFallbackExhaustedError(preferred_error, exc) from exc
            raise
        reason = "rate_limited" if "429" in str(exc).lower() or "rate limit" in str(exc).lower() else type(exc).__name__
        record_provider_fallback(fallback_stage, "triggered", reason)
        try:
            fallback_parent_call_id = str(getattr(exc, "harness_call_id", "") or parent_call_id or "") or None
            response = _invoke_provider_chat(
                chat_messages,
                pool=mimo_fallback_model_pool(),
                model_name=mimo_fallback_model_name(),
                base_url=mimo_fallback_base_url(),
                timeout_seconds=timeout_seconds,
                max_rate_limit_retries=_env_int("MIMO_FALLBACK_RATE_LIMIT_RETRIES", 1),
                max_connection_retries=_env_int("MIMO_FALLBACK_CONNECTION_RETRIES", 1),
                max_timeout_retries=0,
                max_gateway_retries=0,
                temperature=temperature,
                deadline_monotonic=deadline_monotonic,
                extra_body=mimo_fallback_extra_body(),
                max_tokens=None,
                default_headers=_mimo_default_headers("MIMO_FALLBACK_USER_AGENT"),
                response_format_json=True,
                provider_name="mimo_fallback",
                parent_call_id=fallback_parent_call_id,
                call_kind="fallback",
            )
        except Exception as fallback_exc:
            record_provider_fallback(fallback_stage, "failure")
            raise ProviderFallbackExhaustedError(exc, fallback_exc) from fallback_exc
        record_provider_fallback(fallback_stage, "success")
        return _mark_response_provider(response, "mimo_fallback")


def _invoke_provider_chat(
    chat_messages: list[dict[str, Any]],
    *,
    pool: ResearchModelPool,
    model_name: str,
    base_url: str | None,
    timeout_seconds: float,
    max_rate_limit_retries: int,
    max_connection_retries: int,
    max_timeout_retries: int,
    max_gateway_retries: int,
    temperature: float,
    deadline_monotonic: float | None,
    extra_body: dict[str, Any] | None,
    max_tokens: int | None,
    default_headers: dict[str, str] | None,
    response_format_json: bool,
    provider_name: str,
    parent_call_id: str | None,
    call_kind: str,
) -> Any:
    last_error: Exception | None = None
    retry_parent_call_id = parent_call_id
    attempt = 0
    rate_limit_retries_used = 0
    connection_retries_used = 0
    timeout_retries_used = 0
    gateway_retries_used = 0
    while True:
        remaining = deadline_monotonic - time.monotonic() if deadline_monotonic is not None else None
        if remaining is not None and remaining <= 0:
            raise TimeoutError("research model total deadline expired before credential acquisition")
        acquire_timeout = min(pool.acquire_timeout_seconds, remaining) if remaining is not None else None
        with pool.credential(timeout_seconds=acquire_timeout) as credential:
            remaining = deadline_monotonic - time.monotonic() if deadline_monotonic is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("research model total deadline expired after credential acquisition")
            request_timeout = min(timeout_seconds, remaining) if remaining is not None else timeout_seconds
            model_kwargs: dict[str, Any] = {
                "model": model_name,
                "api_key": credential.api_key,
                "base_url": base_url,
                "temperature": temperature,
                "timeout": max(0.1, request_timeout),
                "extra_body": extra_body,
                # Without an explicit max_retries, the underlying openai SDK silently applies
                # its own default (openai.DEFAULT_MAX_RETRIES = 2) inside model.invoke() below --
                # on a timeout it retries up to 2 more times, each re-running the full
                # `timeout` again with exponential backoff in between. That happens entirely
                # inside this one model.invoke() call, invisible to every deadline/budget check
                # in this module and in research_service.py, which can only look at elapsed time
                # *between* invoke_research_chat() calls. Empirically this turned a configured
                # single-attempt timeout into ~3x the wall time on a real hang (see
                # docs/evaluations/deep-research-multi-topic-question-loop-first-run-e2e-20260725.md
                # section 5 item 5 / docs/evaluations/deep-research-revision-timeout-hidden-retry-fix-
                # e2e-20260725.md) -- e.g. the 62.8s revision cap actually ran 100.2s. This mirrors
                # the exact same unconfigured-max_retries hang documented in
                # docs/modules/feishu-hitl-approve-hang-v0.1.md for the main chat model; that fix
                # never covered this separate research model call path. Retry semantics that *are*
                # budget-aware already exist one layer up (this function's own
                # max_rate_limit_retries loop, _validation_model()'s transient-connection retry,
                # writer.py's RESEARCH_WRITER_MAX_RETRIES) -- this only removes the SDK's invisible,
                # budget-blind extra attempts underneath them.
                "max_retries": _env_int("RESEARCH_MODEL_MAX_RETRIES", 0),
            }
            if max_tokens is not None:
                # ChatOpenAI exposes this as ``max_tokens`` and translates it to the
                # provider-facing ``max_completion_tokens`` field in the request payload.
                model_kwargs["max_tokens"] = max_tokens
            if default_headers:
                model_kwargs["default_headers"] = dict(default_headers)
            request_model_kwargs: dict[str, Any] = {}
            if response_format_json:
                request_model_kwargs["response_format"] = {"type": "json_object"}
            if request_model_kwargs:
                model_kwargs["model_kwargs"] = request_model_kwargs
            model = ChatOpenAI(**model_kwargs)
            try:
                response = invoke_with_model_trace(
                    lambda: model.invoke(chat_messages),
                    messages=chat_messages,
                    requested_provider=provider_name,
                    requested_model=model_name,
                    actual_provider=provider_name,
                    call_kind="retry" if attempt > 0 else call_kind,
                    parent_call_id=retry_parent_call_id,
                    attempt=attempt + 1,
                    parameters={
                        "temperature": temperature,
                        "timeout_seconds": max(0.1, request_timeout),
                        "response_format": {"type": "json_object"} if response_format_json else None,
                        "max_tokens": max_tokens,
                        "default_headers": dict(default_headers or {}),
                        "max_retries": model_kwargs["max_retries"],
                        "extra_body": extra_body,
                        "base_url": base_url,
                        "credential_index": credential.index,
                    },
                )
                pool.record_success(credential)
                return response
            except Exception as exc:
                last_error = exc
                retry_parent_call_id = str(getattr(exc, "harness_call_id", "") or retry_parent_call_id or "") or None
                rate_limited = pool.record_failure(credential, exc)
                if rate_limited and rate_limit_retries_used < max(0, max_rate_limit_retries):
                    rate_limit_retries_used += 1
                    attempt += 1
                    continue
                if _is_timeout_error(exc) and timeout_retries_used < max(0, max_timeout_retries):
                    timeout_retries_used += 1
                    attempt += 1
                    continue
                if _is_transient_connection_error(exc) and connection_retries_used < max(0, max_connection_retries):
                    connection_retries_used += 1
                    attempt += 1
                    time.sleep(0.25)
                    continue
                if _is_retryable_gateway_error(exc) and gateway_retries_used < max(0, max_gateway_retries):
                    gateway_retries_used += 1
                    attempt += 1
                    # Give a managed gateway a brief chance to retire the malformed
                    # upstream response before it selects a key for the fresh request.
                    time.sleep(min(1.0, 0.25 * (2 ** (gateway_retries_used - 1))))
                    continue
                raise
    if last_error:
        raise last_error
    raise RuntimeError("research model invocation failed without an error")


def _mark_response_provider(response: Any, provider: str) -> Any:
    try:
        metadata = dict(getattr(response, "response_metadata", None) or {})
        metadata["harness_provider"] = provider
        response.response_metadata = metadata
    except Exception:
        pass
    return response


def model_pool_diagnostics() -> dict[str, Any]:
    """Per-role pool diagnostics, keyed ``{"evidence": {...}, "writer": {...}, "ranker": {...}}``.

    If the writer or ranker role has no dedicated configuration it reuses the evidence
    role's key list (see ``configured_keys``), so those dicts will be identical in that
    case -- this is expected, not a bug.
    """
    result: dict[str, Any] = {}
    for role in ("evidence", "writer", "ranker"):
        try:
            result[role] = research_model_pool(role).diagnostics()
        except ValueError as exc:
            result[role] = {"key_count": 0, "status": "not_configured", "error": str(exc)}
    if preferred_provider_enabled():
        result["preferred_provider"] = {
            **preferred_provider_model_pool().diagnostics(),
            "model": preferred_provider_model_name(),
            "base_url_configured": bool(preferred_provider_base_url()),
        }
    else:
        result["preferred_provider"] = {"key_count": 0, "status": "not_configured"}
    result["provider_fallback"] = provider_fallback_diagnostics()
    return result


def research_role_model_name(role: ResearchRole) -> str:
    if role == "writer":
        return os.getenv("RESEARCH_WRITER_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini")
    if role == "ranker":
        return os.getenv("RESEARCH_RANKER_MODEL_NAME") or research_role_model_name("evidence")
    return os.getenv("RESEARCH_EVIDENCE_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini")


def research_role_base_url(role: ResearchRole) -> str | None:
    if role == "writer":
        return (
            os.getenv("RESEARCH_WRITER_BASE_URL")
            or os.getenv("RESEARCH_EVIDENCE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
    if role == "ranker":
        return os.getenv("RESEARCH_RANKER_BASE_URL") or research_role_base_url("evidence")
    return os.getenv("RESEARCH_EVIDENCE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None


def configured_keys(role: ResearchRole = "evidence") -> list[str]:
    """Resolve the API key pool for one research role.

    ``evidence``: primary key is ``RESEARCH_EVIDENCE_API_KEY`` (falls back to
    ``OPENAI_API_KEY``); extra keys merge the new ``RESEARCH_EVIDENCE_EXTRA_API_KEYS``
    variable with the legacy ``RESEARCH_MODEL_EXTRA_API_KEYS`` variable (deduped) so
    existing production configuration (Milestone A) keeps working unchanged.

    ``writer``: primary key is ``RESEARCH_WRITER_API_KEY``. If that is unset, the
    whole writer role reuses the evidence role's key list -- this is the historical
    shared-pool behavior and must not change for deployments that never configured
    ``RESEARCH_WRITER_*``.

    ``ranker``: primary key is ``RESEARCH_RANKER_API_KEY``, same unset-falls-back-to-
    evidence pattern as ``writer`` -- a deployment that never configures
    ``RESEARCH_RANKER_*`` gets the source-candidate LLM boost running on the same
    cheap evidence-role model/keys with zero extra setup.
    """
    if role == "writer":
        primary = os.getenv("RESEARCH_WRITER_API_KEY", "").strip()
        if not primary:
            return configured_keys("evidence")
        extras = [item.strip() for item in os.getenv("RESEARCH_WRITER_EXTRA_API_KEYS", "").split(",")]
        return list(dict.fromkeys(item for item in [primary, *extras] if item))

    if role == "ranker":
        primary = os.getenv("RESEARCH_RANKER_API_KEY", "").strip()
        if not primary:
            return configured_keys("evidence")
        extras = [item.strip() for item in os.getenv("RESEARCH_RANKER_EXTRA_API_KEYS", "").split(",")]
        return list(dict.fromkeys(item for item in [primary, *extras] if item))

    primary = os.getenv("RESEARCH_EVIDENCE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    extras_new = [item.strip() for item in os.getenv("RESEARCH_EVIDENCE_EXTRA_API_KEYS", "").split(",")]
    extras_legacy = [item.strip() for item in os.getenv("RESEARCH_MODEL_EXTRA_API_KEYS", "").split(",")]
    return list(dict.fromkeys(item for item in [primary.strip(), *extras_new, *extras_legacy] if item))


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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")
