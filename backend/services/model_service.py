"""Model/provider service functions."""

from __future__ import annotations

import os
import hashlib
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import openai

from ..research.model_runtime import (
    ResearchRole,
    configured_keys,
    research_role_base_url,
    research_role_model_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
MODEL_CONFIG_KEYS = {
    "MODEL_NAME",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_EXTRA_API_KEYS",
    "MIMO_EXTRA_API_KEYS",
    "RECALL_MODEL_NAME",
    "RECALL_BASE_URL",
    "RECALL_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIM",
    "DREAM_MODEL_NAME",
    "DREAM_BASE_URL",
    "DREAM_API_KEY",
    "RESEARCH_EVIDENCE_MODEL_NAME",
    "RESEARCH_EVIDENCE_BASE_URL",
    "RESEARCH_EVIDENCE_API_KEY",
    "RESEARCH_EVIDENCE_EXTRA_API_KEYS",
    "RESEARCH_WRITER_MODEL_NAME",
    "RESEARCH_WRITER_BASE_URL",
    "RESEARCH_WRITER_API_KEY",
    "RESEARCH_WRITER_EXTRA_API_KEYS",
}

# Env vars that back one of the long-lived agents cached on app.state -- see
# backend/main.py::build_agent/build_review_agent/build_kb_qa_agent. Used by
# affected_agent_attrs() so routes/model_management.py can reset only the agents that
# actually depend on whatever the caller just changed (docs/modules/model-config-completion-v0.1.md 3.2).
_ROLE_TO_AGENT_ATTRS: dict[str, tuple[str, ...]] = {
    # build_model() backs both the main agent and the kb_qa agent.
    "MODEL_NAME": ("agent", "kb_qa_agent"),
    "OPENAI_BASE_URL": ("agent", "kb_qa_agent"),
    "OPENAI_API_KEY": ("agent", "kb_qa_agent"),
    "OPENAI_EXTRA_API_KEYS": ("agent", "kb_qa_agent"),
    "MIMO_EXTRA_API_KEYS": ("agent", "kb_qa_agent"),
    # build_recall_model() is only used inside build_agent()'s middleware stack.
    "RECALL_MODEL_NAME": ("agent",),
    "RECALL_BASE_URL": ("agent",),
    "RECALL_API_KEY": ("agent",),
    # build_dream_model() backs the review agent only.
    "DREAM_MODEL_NAME": ("review_agent",),
    "DREAM_BASE_URL": ("review_agent",),
    "DREAM_API_KEY": ("review_agent",),
    # EMBEDDING_*/RESEARCH_*: not cached on any agent, already read fresh per call.
}

# Roles checked with one real minimal chat completion.
_CHAT_ROLE_TARGETS = {"chat_model", "recall_model", "dream_model"}
# High-frequency research roles: multiple keys, checked individually.
_MULTI_KEY_ROLE_TARGETS = {"research_evidence_model", "research_writer_model"}

_RESEARCH_ROLE_ENV = {
    "evidence": ("RESEARCH_EVIDENCE_API_KEY", "RESEARCH_EVIDENCE_EXTRA_API_KEYS"),
    "writer": ("RESEARCH_WRITER_API_KEY", "RESEARCH_WRITER_EXTRA_API_KEYS"),
}


def model_config() -> dict:
    return {
        "chat_model": {
            "model": os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            "base_url_configured": bool(os.getenv("OPENAI_BASE_URL")),
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        },
        "recall_model": {
            "model": os.getenv("RECALL_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            "base_url_configured": bool(os.getenv("RECALL_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
            "api_key_configured": bool(os.getenv("RECALL_API_KEY") or os.getenv("OPENAI_API_KEY")),
        },
        "embedding": {
            "model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
            "dim": int(os.getenv("EMBEDDING_DIM", "1024")),
            "base_url_configured": bool(os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
            "api_key_configured": bool(os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")),
        },
        "dream_model": {
            "model": os.getenv("DREAM_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            "base_url_configured": bool(os.getenv("DREAM_BASE_URL") or os.getenv("OPENAI_BASE_URL")),
            "api_key_configured": bool(os.getenv("DREAM_API_KEY") or os.getenv("OPENAI_API_KEY")),
        },
        "research_evidence_model": {
            **_research_role_config("evidence"),
        },
        "research_writer_model": {
            **_research_role_config("writer"),
        },
    }


def _research_role_config(role: ResearchRole) -> dict:
    base_url = research_role_base_url(role)
    entries = _research_role_key_entries(role)
    inherits_from = "research_evidence_model" if role == "writer" and not os.getenv("RESEARCH_WRITER_API_KEY", "").strip() else None
    return {
        "model": research_role_model_name(role),
        "base_url_configured": bool(base_url),
        "base_url": _safe_base_url(base_url),
        "base_url_source": _research_role_base_url_source(role),
        "key_count": len(entries),
        "keys": [_public_key_entry(item) for item in entries],
        "inherits_keys_from": inherits_from,
    }


def _safe_base_url(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = urlsplit(text)
        if not parsed.scheme or not parsed.hostname:
            return None
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return None


def _research_role_base_url_source(role: ResearchRole) -> str | None:
    if role == "writer":
        if os.getenv("RESEARCH_WRITER_BASE_URL", "").strip():
            return "dedicated"
        if os.getenv("RESEARCH_EVIDENCE_BASE_URL", "").strip():
            return "inherited_evidence"
        if os.getenv("OPENAI_BASE_URL", "").strip():
            return "inherited_chat"
        return None
    if os.getenv("RESEARCH_EVIDENCE_BASE_URL", "").strip():
        return "dedicated"
    if os.getenv("OPENAI_BASE_URL", "").strip():
        return "inherited_chat"
    return None


def _api_key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _mask_api_key(api_key: str) -> str:
    key = api_key.strip()
    if len(key) <= 4:
        return f"{key[:1]}..."
    if len(key) <= 8:
        return f"{key[:2]}...{key[-1:]}"
    return f"{key[:7]}...{key[-4:]}"


def _split_env_keys(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def _dedupe_key_entries(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for entry in entries:
        key = entry["api_key"]
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


def _research_role_key_entries(role: ResearchRole) -> list[dict]:
    if role == "writer" and not os.getenv("RESEARCH_WRITER_API_KEY", "").strip():
        return [
            {
                "api_key": key,
                "source": "inherited_evidence",
                "deletable": False,
                "shared": True,
            }
            for key in configured_keys("evidence")
        ]

    primary_name, extra_name = _RESEARCH_ROLE_ENV[role]
    primary = os.getenv(primary_name, "").strip()
    entries: list[dict] = []
    if primary:
        entries.append({"api_key": primary, "source": "dedicated", "deletable": True, "shared": False})
    elif role == "evidence":
        chat_key = os.getenv("OPENAI_API_KEY", "").strip()
        if chat_key:
            entries.append({"api_key": chat_key, "source": "inherited_chat", "deletable": False, "shared": True})

    entries.extend(
        {"api_key": key, "source": "dedicated", "deletable": True, "shared": False}
        for key in _split_env_keys(extra_name)
    )
    if role == "evidence":
        entries.extend(
            {"api_key": key, "source": "legacy_shared", "deletable": True, "shared": True}
            for key in _split_env_keys("RESEARCH_MODEL_EXTRA_API_KEYS")
        )
    return _dedupe_key_entries(entries)


def _public_key_entry(entry: dict) -> dict:
    api_key = entry["api_key"]
    return {
        "id": _api_key_id(api_key),
        "masked": _mask_api_key(api_key),
        "source": entry["source"],
        "deletable": bool(entry["deletable"]),
        "shared": bool(entry["shared"]),
    }


def affected_agent_attrs(updated_fields: list[str]) -> set[str]:
    """Which app.state agent attributes need to be reset after these env vars changed."""
    attrs: set[str] = set()
    for changed_field in updated_fields:
        attrs.update(_ROLE_TO_AGENT_ATTRS.get(changed_field, ()))
    return attrs


def check_models(targets: list[str], timeout_seconds: int = 30) -> dict:
    """Run one real, minimal request per target and report ok/auth_error/timeout/rate_limited/unknown.

    Multi-key research roles (research_evidence_model/research_writer_model) test every
    configured key individually and additionally return a ``keys`` list (same shape as
    ``model_runtime.py::model_pool_diagnostics()``'s per-key output).
    """
    results = []
    for target in targets:
        if target in _MULTI_KEY_ROLE_TARGETS:
            results.append(_check_research_role(target, timeout_seconds))
        elif target == "embedding":
            results.append(_check_embedding(timeout_seconds))
        elif target in _CHAT_ROLE_TARGETS:
            results.append(_check_chat_role(target, timeout_seconds))
        else:
            results.append({"target": target, "status": "error", "latency_ms": 0, "message": "unknown target"})
    return {"results": results, "timeout_seconds": timeout_seconds}


def providers() -> dict:
    return {
        "providers": [
            {"provider": "openai_compatible", "enabled": bool(os.getenv("OPENAI_API_KEY"))},
            {"provider": "siliconflow_embedding", "enabled": bool(os.getenv("EMBEDDING_API_KEY"))},
            {"provider": "lightrag_local", "enabled": True},
        ]
    }


def update_model_config(payload: dict) -> dict:
    updates = _model_config_updates(payload)
    if not updates:
        return {
            "status": "saved",
            "restart_required": False,
            "updated_fields": [],
            "config_path": ".env",
            "message": "没有可写入的配置变更。",
        }
    _write_env_updates(updates)
    os.environ.update(updates)
    return {
        "status": "saved",
        "restart_required": False,
        "updated_fields": sorted(updates.keys()),
        "config_path": ".env",
        "message": "配置已保存并生效。",
    }


def add_research_role_key(role: ResearchRole, api_key: str) -> dict:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("API key cannot be empty")
    if any(entry["api_key"] == key and entry["source"] == "dedicated" for entry in _research_role_key_entries(role)):
        raise ValueError("This API key is already configured for the role")

    primary_name, extra_name = _RESEARCH_ROLE_ENV[role]
    primary = os.getenv(primary_name, "").strip()
    extras = _split_env_keys(extra_name)
    if primary:
        extras.append(key)
        updates = {extra_name: ",".join(dict.fromkeys(extras))}
    else:
        updates = {primary_name: key, extra_name: ",".join(dict.fromkeys(extras))}
    _write_and_apply_env_updates(updates)
    return {
        "status": "added",
        "key": _public_key_entry({"api_key": key, "source": "dedicated", "deletable": True, "shared": False}),
        "config": _research_role_config(role),
    }


def delete_research_role_key(role: ResearchRole, key_id: str) -> dict:
    entries = _research_role_key_entries(role)
    matches = [entry for entry in entries if _api_key_id(entry["api_key"]) == key_id]
    if not matches:
        raise ValueError("API key not found")
    entry = matches[0]
    if not entry["deletable"]:
        raise ValueError("Inherited API keys must be managed from their source role")

    key = entry["api_key"]
    if entry["source"] == "legacy_shared":
        remaining_legacy = [item for item in _split_env_keys("RESEARCH_MODEL_EXTRA_API_KEYS") if item != key]
        updates = {"RESEARCH_MODEL_EXTRA_API_KEYS": ",".join(remaining_legacy)}
    else:
        primary_name, extra_name = _RESEARCH_ROLE_ENV[role]
        primary = os.getenv(primary_name, "").strip()
        extras = [item for item in _split_env_keys(extra_name) if item != key]
        if primary == key:
            updates = {
                primary_name: extras[0] if extras else "",
                extra_name: ",".join(extras[1:]),
            }
        else:
            updates = {primary_name: primary, extra_name: ",".join(extras)}
    _write_and_apply_env_updates(updates)
    return {
        "status": "deleted",
        "deleted_key_id": key_id,
        "config": _research_role_config(role),
    }


def check_research_role_candidate(
    role: ResearchRole,
    api_key: str,
    *,
    timeout_seconds: int = 30,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("API key cannot be empty")
    return _check_research_key(
        role,
        key,
        timeout_seconds=timeout_seconds,
        model=model,
        base_url=base_url,
    )


def check_research_role_key(
    role: ResearchRole,
    key_id: str,
    *,
    timeout_seconds: int = 30,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    for entry in _research_role_key_entries(role):
        if _api_key_id(entry["api_key"]) == key_id:
            return _check_research_key(
                role,
                entry["api_key"],
                timeout_seconds=timeout_seconds,
                model=model,
                base_url=base_url,
            )
    raise ValueError("API key not found")


def _model_config_updates(payload: dict) -> dict[str, str]:
    chat = payload.get("chat_model") or {}
    recall = payload.get("recall_model") or {}
    embedding = payload.get("embedding") or {}
    dream = payload.get("dream_model") or {}
    evidence = payload.get("research_evidence_model") or {}
    writer = payload.get("research_writer_model") or {}
    updates: dict[str, str] = {}

    _set_if_present(updates, "MODEL_NAME", chat.get("model"))
    _set_if_present(updates, "OPENAI_BASE_URL", chat.get("base_url"))
    _set_secret_if_present(updates, "OPENAI_API_KEY", chat.get("api_key"))

    if recall.get("follow_chat_model", True):
        _set_if_present(updates, "RECALL_MODEL_NAME", chat.get("model"))
    else:
        recall_model = str(recall.get("model") or "").strip()
        if not recall_model:
            raise ValueError("recall_model.model is required when follow_chat_model is false")
        updates["RECALL_MODEL_NAME"] = recall_model
    # Independent of follow_chat_model: only set when provided, so leaving these blank
    # keeps recall on the chat model's base_url/key (existing fallback behavior).
    _set_if_present(updates, "RECALL_BASE_URL", recall.get("base_url"))
    _set_secret_if_present(updates, "RECALL_API_KEY", recall.get("api_key"))

    _set_if_present(updates, "EMBEDDING_MODEL", embedding.get("model"))
    _set_if_present(updates, "EMBEDDING_BASE_URL", embedding.get("base_url"))
    _set_secret_if_present(updates, "EMBEDDING_API_KEY", embedding.get("api_key"))
    if embedding.get("dim") is not None:
        updates["EMBEDDING_DIM"] = str(int(embedding["dim"]))

    if dream:
        _set_if_present(updates, "DREAM_MODEL_NAME", dream.get("model"))
        _set_if_present(updates, "DREAM_BASE_URL", dream.get("base_url"))
        _set_secret_if_present(updates, "DREAM_API_KEY", dream.get("api_key"))

    if evidence:
        _set_if_present(updates, "RESEARCH_EVIDENCE_MODEL_NAME", evidence.get("model"))
        _set_if_present(updates, "RESEARCH_EVIDENCE_BASE_URL", evidence.get("base_url"))
        _set_role_api_keys(updates, "RESEARCH_EVIDENCE_API_KEY", "RESEARCH_EVIDENCE_EXTRA_API_KEYS", evidence.get("api_keys"))

    if writer:
        _set_if_present(updates, "RESEARCH_WRITER_MODEL_NAME", writer.get("model"))
        _set_if_present(updates, "RESEARCH_WRITER_BASE_URL", writer.get("base_url"))
        _set_role_api_keys(updates, "RESEARCH_WRITER_API_KEY", "RESEARCH_WRITER_EXTRA_API_KEYS", writer.get("api_keys"))

    return {key: value for key, value in updates.items() if key in MODEL_CONFIG_KEYS}


def _set_role_api_keys(updates: dict[str, str], primary_key: str, extra_key: str, api_keys: object) -> None:
    """Split a UI-managed key list into ``<ROLE>_API_KEY`` + ``<ROLE>_EXTRA_API_KEYS``.

    Unlike ``_set_if_present``, the extra-keys var is written whenever a non-empty key
    list is supplied (even if it collapses to a single key), so removing extra keys
    through the UI actually clears the stored list instead of leaving stale keys behind.
    """
    keys = [str(item).strip() for item in (api_keys or []) if str(item or "").strip()]
    if not keys:
        return
    updates[primary_key] = keys[0]
    updates[extra_key] = ",".join(keys[1:])


def _write_and_apply_env_updates(updates: dict[str, str]) -> None:
    filtered = {key: value for key, value in updates.items() if key in MODEL_CONFIG_KEYS or key == "RESEARCH_MODEL_EXTRA_API_KEYS"}
    _write_env_updates(filtered)
    os.environ.update(filtered)


def _set_if_present(updates: dict[str, str], key: str, value: object) -> None:
    text = str(value or "").strip()
    if text:
        updates[key] = text


def _set_secret_if_present(updates: dict[str, str], key: str, value: object) -> None:
    text = str(value or "").strip()
    if text:
        updates[key] = text


def _write_env_updates(updates: dict[str, str]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        key = _env_line_key(line)
        if key and key in updates:
            output.append(f"{key}={_quote_env_value(updates[key])}")
            seen.add(key)
        else:
            output.append(line)

    missing = [key for key in updates if key not in seen]
    if missing and output and output[-1].strip():
        output.append("")
    for key in missing:
        output.append(f"{key}={_quote_env_value(updates[key])}")

    fd, tmp_name = tempfile.mkstemp(prefix=".env.", dir=str(ENV_PATH.parent), text=True)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(output).rstrip() + "\n")
        os.replace(tmp_path, ENV_PATH)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def _env_line_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    if key.startswith("export "):
        key = key.removeprefix("export ").strip()
    return key if key else None


def _quote_env_value(value: str) -> str:
    if value == "":
        return '""'
    if any(char.isspace() or char in value for char in ['"', "'", "#", "="]):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


# ---------------------------------------------------------------------------
# Real connectivity checks (3.1)
# ---------------------------------------------------------------------------


def _chat_role_credentials(target: str) -> tuple[str, str | None, str | None]:
    if target == "chat_model":
        return (
            os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            os.getenv("OPENAI_API_KEY") or None,
            os.getenv("OPENAI_BASE_URL") or None,
        )
    if target == "recall_model":
        return (
            os.getenv("RECALL_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            os.getenv("RECALL_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            os.getenv("RECALL_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        )
    if target == "dream_model":
        return (
            os.getenv("DREAM_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
            os.getenv("DREAM_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            os.getenv("DREAM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        )
    raise ValueError(f"unsupported chat role target: {target}")


def _check_chat_role(target: str, timeout_seconds: int) -> dict:
    model_name, api_key, base_url = _chat_role_credentials(target)
    if not api_key:
        return {"target": target, "status": "missing_config", "latency_ms": 0, "message": "API key is not configured"}
    started = time.time()
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=timeout_seconds,
        )
        return {
            "target": target,
            "status": "ok",
            "latency_ms": int((time.time() - started) * 1000),
            "message": "real completion request succeeded",
        }
    except Exception as exc:
        status, message = _classify_check_error(exc)
        return {"target": target, "status": status, "latency_ms": int((time.time() - started) * 1000), "message": message}


def _resolve_embedding_credentials() -> tuple[str | None, str | None]:
    """Resolve (api_key, base_url) as one atomic provider pair, not independently-chosen fields.

    Real .env has EMBEDDING_API_KEY/EMBEDDING_BASE_URL unset, the real dedicated embedding
    credentials under the LightRAG-style EMBEDDING_BINDING_API_KEY/EMBEDDING_BINDING_HOST (same
    fallback workers.py's rerank embedding lookup already relies on), and OPENAI_API_KEY/
    OPENAI_BASE_URL pointing at a completely different, embedding-incapable chat provider.
    Picking api_key and base_url via independent `or` chains previously mixed
    EMBEDDING_BINDING_API_KEY with OPENAI_BASE_URL -- confirmed via real verification to 404
    against the wrong provider -- so candidates here are tried as matched pairs instead."""
    candidates = [
        (os.getenv("EMBEDDING_API_KEY"), os.getenv("EMBEDDING_BASE_URL")),
        (os.getenv("EMBEDDING_BINDING_API_KEY"), os.getenv("EMBEDDING_BINDING_HOST")),
        (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")),
    ]
    for api_key, base_url in candidates:
        if api_key:
            return api_key, base_url or None
    return None, None


def _check_embedding(timeout_seconds: int) -> dict:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    api_key, base_url = _resolve_embedding_credentials()
    if not api_key:
        return {"target": "embedding", "status": "missing_config", "latency_ms": 0, "message": "API key is not configured"}
    started = time.time()
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        client.embeddings.create(model=model_name, input="ping", timeout=timeout_seconds)
        return {
            "target": "embedding",
            "status": "ok",
            "latency_ms": int((time.time() - started) * 1000),
            "message": "real embedding request succeeded",
        }
    except Exception as exc:
        status, message = _classify_check_error(exc)
        return {"target": "embedding", "status": status, "latency_ms": int((time.time() - started) * 1000), "message": message}


def _check_research_role(target: str, timeout_seconds: int) -> dict:
    role: ResearchRole = "evidence" if target == "research_evidence_model" else "writer"
    entries = _research_role_key_entries(role)
    if not entries:
        return {
            "target": target,
            "status": "missing_config",
            "latency_ms": 0,
            "message": "no API key configured for this role",
            "keys": [],
        }
    started_total = time.time()

    def _check_one_key(index: int, entry: dict) -> dict:
        return {
            "key_index": index,
            **_check_research_key(role, entry["api_key"], timeout_seconds=timeout_seconds),
        }

    # Keys are independent network calls -- run them concurrently so N keys cost
    # ~1 timeout instead of N timeouts when several keys are rate-limited/dead.
    with ThreadPoolExecutor(max_workers=len(entries)) as pool:
        key_results = list(pool.map(lambda item: _check_one_key(*item), enumerate(entries)))
    ok_count = sum(1 for item in key_results if item["status"] == "ok")
    overall = "ok" if ok_count == len(key_results) else ("error" if ok_count == 0 else "partial")
    return {
        "target": target,
        "status": overall,
        "latency_ms": int((time.time() - started_total) * 1000),
        "message": f"{ok_count}/{len(key_results)} keys ok",
        "keys": key_results,
    }


def _check_research_key(
    role: ResearchRole,
    api_key: str,
    *,
    timeout_seconds: int,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    started = time.time()
    key_id = _api_key_id(api_key)
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=str(base_url or "").strip() or research_role_base_url(role),
            timeout=timeout_seconds,
        )
        client.chat.completions.create(
            model=str(model or "").strip() or research_role_model_name(role),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=timeout_seconds,
        )
        return {
            "key_id": key_id,
            "masked": _mask_api_key(api_key),
            "status": "ok",
            "latency_ms": int((time.time() - started) * 1000),
            "message": "real completion request succeeded",
        }
    except Exception as exc:
        status, message = _classify_check_error(exc)
        return {
            "key_id": key_id,
            "masked": _mask_api_key(api_key),
            "status": status,
            "latency_ms": int((time.time() - started) * 1000),
            "message": message,
        }


def _classify_check_error(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    if isinstance(exc, openai.APITimeoutError):
        return "timeout", message
    if isinstance(exc, openai.RateLimitError):
        return "rate_limited", message
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "auth_error", message
    return "unknown", message
