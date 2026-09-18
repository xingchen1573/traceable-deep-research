"""Isolated replay of one saved research model call."""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_openai import ChatOpenAI

from .audit import ROOT, read_trace_events, sanitized_json_safe, stable_hash
from .model_runtime import configured_keys, configured_mimo_fallback_keys, configured_preferred_provider_keys


REPLAY_ROOT = ROOT / "data" / "research_replays"
JSON_REPAIR_MODES = {"none", "production"}


def load_model_call(job_id: str, *, call_id: str | None = None, stage: str | None = None) -> dict[str, Any]:
    calls = [event for event in read_trace_events(job_id) if event.get("event") == "model_call"]
    if call_id:
        match = next((event for event in calls if str(event.get("call_id")) == call_id), None)
        if match is None:
            raise ValueError(f"model call not found: {job_id}/{call_id}")
        return match
    if stage:
        stage_calls = [event for event in calls if str(event.get("stage")) == stage]
        if not stage_calls:
            raise ValueError(f"model call stage not found: {job_id}/{stage}")
        return sorted(stage_calls, key=lambda item: int(item.get("sequence") or 0))[-1]
    if not calls:
        raise ValueError(f"no model calls found for job: {job_id}")
    return sorted(calls, key=lambda item: int(item.get("sequence") or 0))[-1]


def _messages_from_call(
    call: dict[str, Any],
    *,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    systems = call.get("system_prompt") or []
    if system_prompt_override is not None:
        systems = [system_prompt_override]
    for content in systems:
        messages.append({"role": "system", "content": str(content or "")})
    model_input = [dict(item) for item in call.get("model_input") or [] if isinstance(item, dict)]
    if user_prompt_override is not None:
        replaced = False
        for item in model_input:
            if str(item.get("role") or "").lower() in {"user", "human"}:
                item["content"] = user_prompt_override
                replaced = True
                break
        if not replaced:
            model_input.append({"role": "user", "content": user_prompt_override})
    messages.extend(model_input)
    return messages


def _credential_for_call(call: dict[str, Any]) -> str | None:
    provider = str(call.get("actual_provider") or call.get("requested_provider") or "")
    parameters = call.get("parameters") if isinstance(call.get("parameters"), dict) else {}
    try:
        credential_index = max(0, int(parameters.get("credential_index") or 0))
    except (TypeError, ValueError):
        credential_index = 0
    if provider == "mimo_fallback":
        keys = configured_mimo_fallback_keys()
        return keys[credential_index % len(keys)] if keys else None
    if provider == "preferred":
        keys = configured_preferred_provider_keys()
        return keys[credential_index % len(keys)] if keys else None
    role = provider.partition(":")[2]
    if role not in {"evidence", "writer", "ranker"}:
        role = "writer" if str(call.get("stage")) in {"section_writer", "editorial"} else "evidence"
    keys = configured_keys(role)  # type: ignore[arg-type]
    return keys[credential_index % len(keys)] if keys else None


def _default_invoke(call: dict[str, Any], messages: list[dict[str, Any]], overrides: dict[str, Any]) -> Any:
    parameters = dict(call.get("parameters") or {})
    model_name = str(overrides.get("model") or call.get("actual_model") or call.get("requested_model") or "").strip()
    if not model_name:
        raise ValueError("saved call does not contain a model name")
    timeout = float(overrides.get("timeout_seconds") or parameters.get("timeout_seconds") or 180)
    temperature = float(
        overrides["temperature"] if overrides.get("temperature") is not None else parameters.get("temperature") or 0
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": _credential_for_call(call),
        "base_url": parameters.get("base_url"),
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": 0,
    }
    if parameters.get("extra_body") is not None:
        kwargs["extra_body"] = parameters["extra_body"]
    if parameters.get("max_tokens") is not None:
        kwargs["max_tokens"] = parameters["max_tokens"]
    if parameters.get("response_format"):
        kwargs["model_kwargs"] = {"response_format": parameters["response_format"]}
    return ChatOpenAI(**kwargs).invoke(messages)


def _production_parse_json(content: str) -> tuple[dict[str, Any], Exception | None]:
    from backend.services.research_service import _parse_json_or_error

    return _parse_json_or_error(content)


def _repair_schema(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() not in {"user", "human"}:
            continue
        parsed, _error = _production_parse_json(str(message.get("content") or ""))
        schema = parsed.get("schema") if isinstance(parsed, dict) else None
        return schema if isinstance(schema, dict) else None
    return None


def _production_repair_json(
    call: dict[str, Any],
    messages: list[dict[str, Any]],
    content: str,
    error: Exception | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    from backend.services.research_service import _repair_json_response

    parameters = dict(call.get("parameters") or {})
    provider = str(call.get("actual_provider") or call.get("requested_provider") or "")
    role = provider.partition(":")[2]
    if role not in {"evidence", "writer", "ranker"}:
        role = "writer" if str(call.get("stage")) in {"section_writer", "editorial"} else "evidence"
    systems = [str(message.get("content") or "") for message in messages if message.get("role") == "system"]
    temperature = float(
        overrides["temperature"] if overrides.get("temperature") is not None else parameters.get("temperature") or 0
    )
    model_name = str(overrides.get("model") or call.get("actual_model") or call.get("requested_model") or "").strip()
    return _repair_json_response(
        content,
        error if isinstance(error, json.JSONDecodeError) else None,
        schema=_repair_schema(messages),
        temperature=temperature,
        system_prompt="\n\n".join(systems) or None,
        model_name=model_name or None,
        evidence=role == "evidence",
        role=role,
        allow_provider_fallback=False,
        fallback_stage=str(call.get("stage") or "research"),
        force_fallback_provider=provider == "mimo_fallback",
        skip_preferred_provider=provider != "preferred",
        parent_call_id=None,
    )


def replay_model_call(
    *,
    job_id: str,
    call_id: str | None = None,
    stage: str | None = None,
    model_override: str | None = None,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    temperature_override: float | None = None,
    timeout_seconds_override: float | None = None,
    json_repair_mode: str = "none",
    output_root: Path | None = None,
    invoke_fn: Callable[[dict[str, Any], list[dict[str, Any]], dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Replay one model call without invoking jobs, persistence, or knowledge services."""
    if json_repair_mode not in JSON_REPAIR_MODES:
        raise ValueError(f"unsupported JSON repair mode: {json_repair_mode}")
    call = load_model_call(job_id, call_id=call_id, stage=stage)
    messages = _messages_from_call(
        call,
        system_prompt_override=system_prompt_override,
        user_prompt_override=user_prompt_override,
    )
    overrides = {
        "model": model_override,
        "temperature": temperature_override,
        "timeout_seconds": timeout_seconds_override,
        "system_prompt_changed": system_prompt_override is not None,
        "user_prompt_changed": user_prompt_override is not None,
        "json_repair_mode": json_repair_mode,
    }
    replay_id = f"replay_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    replay_dir = (output_root or REPLAY_ROOT) / replay_id
    replay_dir.mkdir(parents=True, exist_ok=False)
    request_document = {
        "schema_version": "research-replay/v2",
        "replay_id": replay_id,
        "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds"),
        "original_job_id": job_id,
        "original_call_id": call.get("call_id"),
        "original_stage": call.get("stage"),
        "experimental_variables": sanitized_json_safe(overrides),
        "side_effect_policy": {
            "job_state_write": False,
            "database_write": False,
            "knowledge_base_write": False,
            "external_side_effects": "model_provider_call_only",
        },
        "messages": sanitized_json_safe(messages),
        "input_hash": stable_hash(messages),
        "saved_call": call,
    }
    (replay_dir / "request.json").write_text(
        json.dumps(request_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    started = dt.datetime.now(dt.timezone.utc).astimezone()
    try:
        response = (invoke_fn or _default_invoke)(call, messages, overrides)
        content = str(getattr(response, "content", response) or "")
        metadata = sanitized_json_safe(getattr(response, "response_metadata", None) or {})
        usage = sanitized_json_safe(getattr(response, "usage_metadata", None) or {})
        parsed, parse_error = _production_parse_json(content)
        repair_attempted = False
        repair_error = None
        parse_status = "success" if parsed else "parse_error"
        if not parsed and json_repair_mode == "production":
            repair_attempted = True
            try:
                parsed = _production_repair_json(call, messages, content, parse_error, overrides)
                parse_status = "repaired" if parsed else "repair_failed"
            except BaseException as exc:
                repair_error = {"type": type(exc).__name__, "message": str(exc)}
                parse_status = "repair_failed"
        result = {
            "status": "success" if content.strip() else "empty_response",
            "raw_response": content,
            "parsed_result": parsed or None,
            "json_parse": {
                "parser": "backend.services.research_service._parse_json_or_error",
                "status": parse_status,
                "repair_mode": json_repair_mode,
                "repair_attempted": repair_attempted,
                "parse_error": str(parse_error) if parse_error else None,
                "repair_error": repair_error,
            },
            "response_metadata": metadata,
            "usage": usage,
            "output_hash": stable_hash(content),
        }
    except BaseException as exc:
        result = {
            "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "raw_response": None,
            "parsed_result": None,
            "output_hash": None,
        }
    finished = dt.datetime.now(dt.timezone.utc).astimezone()
    result_document = {
        "schema_version": "research-replay-result/v2",
        "replay_id": replay_id,
        "original_job_id": job_id,
        "original_call_id": call.get("call_id"),
        "started_at": started.isoformat(timespec="milliseconds"),
        "finished_at": finished.isoformat(timespec="milliseconds"),
        "duration_ms": int((finished - started).total_seconds() * 1000),
        "experimental_variables": sanitized_json_safe(overrides),
        **result,
    }
    result_path = replay_dir / "result.json"
    result_path.write_text(json.dumps(result_document, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **result_document,
        "replay_dir": str(replay_dir),
        "request_path": str(replay_dir / "request.json"),
        "result_path": str(result_path),
    }
