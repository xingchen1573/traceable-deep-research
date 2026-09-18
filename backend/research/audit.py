"""Low-overhead, fail-open audit storage for the research pipeline."""

from __future__ import annotations

import contextvars
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_ARTIFACT_DIR = ROOT / "data" / "research_artifacts"
SCHEMA_VERSION = "research-audit/v1"
TRACE_SCHEMA_VERSION = "research-model-trace/v1"
_INTERNAL_TRACE_KEYS = {"__audit__", "_audit", "_trace", "trace_context"}
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|cookie|secret|password|passwd|access[_-]?token|refresh[_-]?token|client[_-]?secret)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|secret|password|access[_-]?token|refresh[_-]?token)"
    r"(\s*[:=]\s*)([^\s,;\"']+)"
)
_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "research_audit_context",
    default={},
)
_CODE_VERSION_LOCK = threading.Lock()
_CODE_VERSION: dict[str, Any] | None = None
_CALL_CONTEXT_LOCK = threading.Lock()
_CALL_CONTEXT: dict[tuple[str, str], dict[str, Any]] = {}
_CALL_CONTEXT_LIMIT = 20_000


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def audit_enabled() -> bool:
    return _env_bool("RESEARCH_AUDIT_ENABLED", False)


def trace_enabled() -> bool:
    return audit_enabled() and _env_bool("RESEARCH_MODEL_TRACE_ENABLED", True)


def current_audit_context() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})


@contextmanager
def audit_run(
    *,
    job_id: str,
    parent_job_id: str | None = None,
    run_id: str | None = None,
    attempt: int = 1,
    retry_of_job_id: str | None = None,
    replay_of_call_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    previous = _CONTEXT.get()
    next_value = {
        **previous,
        "job_id": str(job_id),
        "parent_job_id": str(parent_job_id or "") or None,
        "run_id": str(run_id or f"run_{uuid.uuid4().hex[:12]}"),
        "trace_id": str(previous.get("trace_id") or f"trace_{uuid.uuid4().hex[:16]}"),
        "attempt": max(1, int(attempt or 1)),
        "retry_of_job_id": str(retry_of_job_id or "") or None,
        "replay_of_call_id": str(replay_of_call_id or "") or None,
    }
    token = _CONTEXT.set(next_value)
    try:
        yield dict(next_value)
    finally:
        _CONTEXT.reset(token)


@contextmanager
def audit_scope(**values: Any) -> Iterator[dict[str, Any]]:
    previous = _CONTEXT.get()
    next_value = {**previous, **{key: value for key, value in values.items() if value is not None}}
    token = _CONTEXT.set(next_value)
    try:
        yield dict(next_value)
    finally:
        _CONTEXT.reset(token)


def submit_with_audit_context(executor: Any, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    copied = contextvars.copy_context()
    return executor.submit(copied.run, fn, *args, **kwargs)


def _secret_values() -> list[str]:
    values: list[str] = []
    for key, value in os.environ.items():
        if not _SECRET_KEY_RE.search(key) or len(value) < 8:
            continue
        values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _redact_text(value: str) -> str:
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", value)
    redacted = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
    for secret in _secret_values():
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except Exception:
        return value
    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return value
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        query.append((key, "[REDACTED]" if _SECRET_KEY_RE.search(key) else item))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in _INTERNAL_TRACE_KEYS:
                continue
            output[key_text] = "[REDACTED]" if _SECRET_KEY_RE.search(key_text) else redact_secrets(item)
        return output
    if isinstance(value, (list, tuple, set)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _redact_url(_redact_text(value))
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return json_safe(model_dump())
        except Exception:
            pass
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        try:
            return json_safe(as_dict())
        except Exception:
            pass
    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": type(value).__name__,
            "content": json_safe(content),
            "response_metadata": json_safe(getattr(value, "response_metadata", None) or {}),
            "usage_metadata": json_safe(getattr(value, "usage_metadata", None) or {}),
        }
    return repr(value)


def sanitized_json_safe(value: Any) -> Any:
    return redact_secrets(json_safe(value))


def stable_hash(value: Any) -> str:
    encoded = json.dumps(sanitized_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))[:160]


def _job_audit_dir(job_id: str) -> Path:
    return RESEARCH_ARTIFACT_DIR / _safe_name(job_id) / "audit"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_artifact(value: Any, *, kind: str, job_id: str | None = None) -> dict[str, Any]:
    context = current_audit_context()
    resolved_job_id = str(job_id or context.get("job_id") or "")
    if not audit_enabled() or not resolved_job_id:
        return {}
    try:
        safe_value = sanitized_json_safe(value)
        digest = stable_hash(safe_value)
        path = _job_audit_dir(resolved_job_id) / "artifacts" / f"sha256_{digest}.json"
        if not path.exists():
            _atomic_write_json(path, safe_value)
        return {
            "kind": str(kind),
            "sha256": digest,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
    except Exception:
        return {}


def _locked_sequence(job_id: str) -> int:
    audit_dir = _job_audit_dir(job_id)
    audit_dir.mkdir(parents=True, exist_ok=True)
    lock_path = audit_dir / ".sequence.lock"
    counter_path = audit_dir / ".sequence"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                current = int(counter_path.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                current = 0
            next_value = current + 1
            temporary = counter_path.with_name(f".{counter_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(str(next_value), encoding="utf-8")
            os.replace(temporary, counter_path)
            return next_value
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def code_version() -> dict[str, Any]:
    global _CODE_VERSION
    with _CODE_VERSION_LOCK:
        if _CODE_VERSION is not None:
            return dict(_CODE_VERSION)
        commit = os.getenv("GIT_COMMIT", "").strip()
        if not commit:
            try:
                commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
            except Exception:
                commit = ""
        source_files = [
            ROOT / "backend" / "services" / "research_service.py",
            ROOT / "backend" / "research" / "model_runtime.py",
            ROOT / "backend" / "research" / "writer.py",
            ROOT / "backend" / "research" / "evidence_native.py",
            ROOT / "backend" / "research" / "quality.py",
            ROOT / "prompts.py",
        ]
        digest = hashlib.sha256()
        for path in source_files:
            try:
                digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
                digest.update(path.read_bytes())
            except OSError:
                continue
        _CODE_VERSION = {
            "git_commit": commit or None,
            "research_source_hash": digest.hexdigest(),
        }
        return dict(_CODE_VERSION)


def configuration_summary() -> dict[str, Any]:
    prefixes = ("RESEARCH_", "MODEL_", "OPENAI_BASE_URL", "MIMO_", "HARNESS_ENV")
    result: dict[str, Any] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith(prefixes) or _SECRET_KEY_RE.search(key):
            continue
        result[key] = redact_secrets(value)
    return result


def _artifact_refs(values: dict[str, Any] | None, *, direction: str, job_id: str) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name, value in (values or {}).items():
        if isinstance(value, dict) and value.get("sha256") and value.get("path"):
            refs[str(name)] = value
        else:
            ref = write_artifact(value, kind=f"{direction}:{name}", job_id=job_id)
            if ref:
                refs[str(name)] = ref
    return refs


_CHECKPOINT_STAGE_CALL_STAGES: dict[str, set[str]] = {
    "planner": {"planner", "question_generation", "reconnaissance"},
    "query_generation": {"query_generation"},
    "search": {"query_generation", "retrieval", "ranker"},
    "retrieval_round": {"query_generation", "retrieval", "ranker"},
    "content_extraction": {"content_extraction"},
    "chunk_generation": {"content_extraction"},
    "evidence_admission": {"content_extraction", "evidence_admission"},
    "coverage": {"coverage_judge"},
    "finding_synthesis": {"finding_synthesis"},
    "evidence_native_blueprint": set(),
    "evidence_context": set(),
    "section_draft": {"section_writer", "writer"},
    "section_prevalidation": {"validation", "semantic_validation", "entailment_validation"},
    "editorial_claim_bank": {"editorial"},
    "editorial": {"editorial"},
    "draft": {"section_writer", "writer", "editorial"},
    "validation": {
        "validation",
        "semantic_validation",
        "entailment_validation",
        "evidence_usage_validation",
        "json_repair",
    },
    "final_status": {"section_writer", "editorial", "semantic_validation", "entailment_validation"},
}


def _checkpoint_head(job_id: str) -> dict[str, Any]:
    try:
        return json.loads((_job_audit_dir(job_id) / "checkpoint_head.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_model_lineage(job_id: str, stage: str) -> list[dict[str, Any]]:
    accepted_stages = _CHECKPOINT_STAGE_CALL_STAGES.get(stage, {stage})
    calls: list[dict[str, Any]] = []
    for event in read_trace_events(job_id):
        if event.get("event") != "model_call" or str(event.get("stage") or "") not in accepted_stages:
            continue
        calls.append({
            "call_id": event.get("call_id"),
            "parent_call_id": event.get("parent_call_id"),
            "stage": event.get("stage"),
            "question_id": event.get("question_id"),
            "section_id": event.get("section_id"),
            "attempt": event.get("attempt"),
            "status": event.get("status"),
            "prompt_hash": event.get("prompt_hash"),
            "input_hash": event.get("input_hash"),
            "output_hash": event.get("output_hash"),
        })
    return calls


def write_stage_checkpoint(
    stage: str,
    *,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    transition_reason: str = "",
    status: str = "succeeded",
    error: Any = None,
    timeout: Any = None,
    interrupt: Any = None,
    prompt_hashes: dict[str, str] | None = None,
    job_id: str | None = None,
) -> str:
    context = current_audit_context()
    resolved_job_id = str(job_id or context.get("job_id") or "")
    if not audit_enabled() or not resolved_job_id:
        return ""
    try:
        audit_dir = _job_audit_dir(resolved_job_id)
        audit_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_lock_path = audit_dir / ".checkpoint.lock"
        with checkpoint_lock_path.open("a+", encoding="utf-8") as checkpoint_lock:
            fcntl.flock(checkpoint_lock.fileno(), fcntl.LOCK_EX)
            sequence = _locked_sequence(resolved_job_id)
            checkpoint_dir = audit_dir / "checkpoints"
            previous_path = audit_dir / "checkpoint_head.json"
            previous = _checkpoint_head(resolved_job_id)
            model_lineage = _stage_model_lineage(resolved_job_id, str(stage))
            resolved_prompt_hashes = dict(prompt_hashes or {})
            if not resolved_prompt_hashes:
                resolved_prompt_hashes = {
                    str(item["call_id"]): str(item["prompt_hash"])
                    for item in model_lineage
                    if item.get("call_id") and item.get("prompt_hash")
                }
            document = {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_id": f"chk_{sequence:06d}_{uuid.uuid4().hex[:10]}",
                "sequence": sequence,
                "job_id": resolved_job_id,
                "parent_job_id": context.get("parent_job_id"),
                "run_id": context.get("run_id"),
                "trace_id": context.get("trace_id"),
                "stage": str(stage),
                "attempt": int(context.get("attempt") or 1),
                "timestamp": _timestamp(),
                "scope": {
                    "question_id": context.get("question_id"),
                    "section_id": context.get("section_id"),
                    "round": context.get("round"),
                },
                "status": str(status),
                "input_artifacts": _artifact_refs(inputs, direction="input", job_id=resolved_job_id),
                "output_artifacts": _artifact_refs(outputs, direction="output", job_id=resolved_job_id),
                "transition_reason": str(transition_reason or ""),
                "lineage": {
                    "retry_of_job_id": context.get("retry_of_job_id"),
                    "replay_of_call_id": context.get("replay_of_call_id"),
                    "previous_checkpoint_id": previous.get("checkpoint_id"),
                    "previous_checkpoint_sha256": previous.get("sha256"),
                    "previous_checkpoint_path": previous.get("path"),
                    "model_calls": model_lineage,
                },
                "configuration": configuration_summary(),
                "code_version": code_version(),
                "prompt_hashes": resolved_prompt_hashes,
                "error": sanitized_json_safe(error) if error is not None else None,
                "timeout": sanitized_json_safe(timeout) if timeout is not None else None,
                "interrupt": sanitized_json_safe(interrupt) if interrupt is not None else None,
            }
            digest = stable_hash(document)
            document["checkpoint_sha256"] = digest
            path = checkpoint_dir / f"{sequence:06d}_{_safe_name(stage)}.json"
            _atomic_write_json(path, document)
            _atomic_write_json(previous_path, {
                "checkpoint_id": document["checkpoint_id"],
                "sha256": digest,
                "path": str(path),
            })
            fcntl.flock(checkpoint_lock.fileno(), fcntl.LOCK_UN)
            return str(path)
    except Exception:
        return ""


def infer_scope_from_payload(payload: Any) -> dict[str, str | None]:
    safe = json_safe(payload)
    question_id: str | None = None
    section_id: str | None = None

    def visit(value: Any, depth: int = 0) -> None:
        nonlocal question_id, section_id
        if depth > 5 or (question_id and section_id):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                if key_text == "question_id" and item and not question_id:
                    question_id = str(item)
                elif key_text in {"section_id", "document_id"} and item and not section_id:
                    section_id = str(item)
                else:
                    visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value[:20]:
                visit(item, depth + 1)

    visit(safe)
    return {"question_id": question_id, "section_id": section_id}


def infer_stage_from_payload(payload: Any, fallback: str = "research_model") -> str:
    task = ""
    if isinstance(payload, dict):
        task = str(payload.get("task") or "").lower()
    rules = (
        ("json", "repair", "json_repair"),
        ("semantic coverage", "research question", "coverage_judge"),
        ("judge whether report claims are supported", "", "entailment_validation"),
        ("evaluate one section of a deep research report", "", "semantic_validation"),
        ("classify each unused evidence item", "", "evidence_usage_validation"),
        ("report critic flagged", "", "semantic_validation"),
        ("research plan", "", "planner"),
        ("research question", "", "question_generation"),
        ("success criteria", "", "question_generation"),
        ("search quer", "", "query_generation"),
        ("atomic evidence", "extract", "content_extraction"),
        ("evidence", "admission", "evidence_admission"),
        ("coverage", "", "coverage_judge"),
        ("analytical findings", "", "finding_synthesis"),
        ("analytical report section", "", "section_writer"),
        ("editorial", "", "editorial"),
        ("entail", "", "entailment_validation"),
        ("critic", "", "semantic_validation"),
        ("validate", "claim", "semantic_validation"),
    )
    for first, second, stage in rules:
        if first in task and (not second or second in task):
            return stage
    return str(fallback or "research_model")


@contextmanager
def payload_audit_scope(payload: Any, *, fallback_stage: str = "research_model") -> Iterator[dict[str, Any]]:
    scope = infer_scope_from_payload(payload)
    current = current_audit_context()
    existing_stage = str(current.get("stage") or "")
    stage = (
        existing_stage
        if existing_stage and existing_stage not in {"research", "research_model", "research_json"}
        else infer_stage_from_payload(payload, fallback_stage)
    )
    with audit_scope(
        stage=stage,
        question_id=scope.get("question_id"),
        section_id=scope.get("section_id"),
    ) as context:
        yield context


def _messages_payload(messages: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in list(messages or []):
        if isinstance(item, dict):
            output.append({str(key): json_safe(value) for key, value in item.items()})
            continue
        role = getattr(item, "type", None) or getattr(item, "role", None) or type(item).__name__
        row = {"role": str(role), "content": json_safe(getattr(item, "content", item))}
        additional = getattr(item, "additional_kwargs", None)
        if additional:
            row["additional_kwargs"] = json_safe(additional)
        output.append(row)
    return output


def _response_details(response: Any) -> dict[str, Any]:
    generations = getattr(response, "generations", None)
    if generations:
        try:
            generation = generations[0]
            if isinstance(generation, list):
                generation = generation[0]
            message = getattr(generation, "message", generation)
            return _response_details(message)
        except Exception:
            pass
    metadata = json_safe(getattr(response, "response_metadata", None) or {})
    usage = json_safe(getattr(response, "usage_metadata", None) or {})
    content = json_safe(getattr(response, "content", response))
    return {
        "raw_response": content,
        "response_metadata": metadata,
        "usage": usage or metadata.get("token_usage") or {},
        "finish_reason": metadata.get("finish_reason") or metadata.get("stop_reason"),
        "actual_model": metadata.get("model_name") or metadata.get("model") or None,
    }


def _error_status(exc: BaseException) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if (
        isinstance(exc, AttributeError)
        and str(exc).strip() == "'str' object has no attribute 'model_dump'"
    ):
        return "invalid_provider_response"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "parse" in text or "json" in text:
        return "parse_error"
    return "provider_error"


def _error_response_details(exc: BaseException) -> dict[str, Any]:
    """Capture a small, sanitized diagnostic for provider response failures."""

    response = getattr(exc, "response", None)
    if response is None:
        return {}
    try:
        headers = getattr(response, "headers", None) or {}
        content_type = str(headers.get("content-type", ""))
    except Exception:
        content_type = ""
    try:
        body = str(getattr(response, "text", "") or "")[:2000]
    except Exception:
        body = ""
    return {
        "error_response": {
            "response_type": type(response).__name__,
            "status_code": getattr(response, "status_code", None),
            "content_type": content_type,
            "body_excerpt": body,
        }
    }


def _append_trace(job_id: str, payload: dict[str, Any]) -> None:
    path = _job_audit_dir(job_id) / "model_calls.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    line = json.dumps(sanitized_json_safe(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(descriptor, line.encode("utf-8"))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _remember_call_context(job_id: str, call_id: str, context: dict[str, Any]) -> None:
    with _CALL_CONTEXT_LOCK:
        _CALL_CONTEXT[(job_id, call_id)] = dict(context)
        while len(_CALL_CONTEXT) > _CALL_CONTEXT_LIMIT:
            _CALL_CONTEXT.pop(next(iter(_CALL_CONTEXT)))


def _call_trace_context(job_id: str, call_id: str) -> dict[str, Any]:
    with _CALL_CONTEXT_LOCK:
        remembered = _CALL_CONTEXT.get((job_id, call_id))
        if remembered is not None:
            return dict(remembered)
    try:
        for event in reversed(read_trace_events(job_id)):
            if event.get("event") != "model_call" or str(event.get("call_id") or "") != call_id:
                continue
            return {
                "trace_id": event.get("trace_id"),
                "run_id": event.get("run_id"),
                "parent_job_id": event.get("parent_job_id"),
                "parent_call_id": event.get("parent_call_id"),
                "stage": event.get("stage"),
                "question_id": event.get("question_id"),
                "section_id": event.get("section_id"),
                "attempt": event.get("attempt"),
            }
    except Exception:
        pass
    return {}


def invoke_with_model_trace(
    invoke: Callable[[], Any],
    *,
    messages: Any,
    requested_provider: str,
    requested_model: str | None,
    parameters: dict[str, Any] | None = None,
    actual_provider: str | None = None,
    call_kind: str = "model",
    parent_call_id: str | None = None,
    attempt: int = 1,
) -> Any:
    context = current_audit_context()
    job_id = str(context.get("job_id") or "")
    if not trace_enabled() or not job_id:
        return invoke()
    try:
        call_id = f"call_{uuid.uuid4().hex[:18]}"
        sequence = _locked_sequence(job_id)
        started_wall = _timestamp()
        started = time.perf_counter()
        request_messages = _messages_payload(messages)
        system_prompt = [row.get("content") for row in request_messages if str(row.get("role")) == "system"]
        model_input = [row for row in request_messages if str(row.get("role")) != "system"]
        checkpoint = _checkpoint_head(job_id)
        call_context = {
            "trace_id": context.get("trace_id"),
            "run_id": context.get("run_id"),
            "parent_job_id": context.get("parent_job_id"),
            "parent_call_id": parent_call_id or context.get("parent_call_id"),
            "stage": context.get("stage") or "research_model",
            "question_id": context.get("question_id"),
            "section_id": context.get("section_id"),
            "attempt": int(attempt or 1),
        }
        base = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event": "model_call",
            "sequence": sequence,
            "call_id": call_id,
            "call_kind": call_kind,
            "job_id": job_id,
            **call_context,
            "started_at": started_wall,
            "requested_provider": requested_provider,
            "requested_model": requested_model,
            "actual_provider": actual_provider or requested_provider,
            "parameters": sanitized_json_safe(parameters or {}),
            "system_prompt": sanitized_json_safe(system_prompt),
            "model_input": sanitized_json_safe(model_input),
            "input_hash": stable_hash({"system_prompt": system_prompt, "model_input": model_input}),
            "prompt_hash": stable_hash(request_messages),
            "code_version": code_version(),
            "lineage": {
                "input_checkpoint_id": checkpoint.get("checkpoint_id"),
                "input_checkpoint_sha256": checkpoint.get("sha256"),
                "input_checkpoint_path": checkpoint.get("path"),
            },
        }
        _remember_call_context(job_id, call_id, call_context)
    except Exception:
        return invoke()
    try:
        response = invoke()
    except BaseException as exc:
        try:
            record = {
                **base,
                **_error_response_details(exc),
                "finished_at": _timestamp(),
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "status": _error_status(exc),
                "success": False,
                "error": json_safe(exc),
                "raw_response": None,
                "parsed_result": None,
                "output_hash": None,
            }
            try:
                _append_trace(job_id, record)
            except Exception:
                pass
            try:
                setattr(exc, "harness_call_id", call_id)
            except Exception:
                pass
        except Exception:
            pass
        raise
    try:
        details = _response_details(response)
        record = {
            **base,
            **details,
            "actual_model": details.get("actual_model") or requested_model,
            "finished_at": _timestamp(),
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "status": "success" if str(details.get("raw_response") or "").strip() else "empty_response",
            "success": bool(str(details.get("raw_response") or "").strip()),
            "parsed_result": None,
            "output_hash": stable_hash(details.get("raw_response")),
        }
        try:
            _append_trace(job_id, record)
        except Exception:
            pass
        try:
            metadata = dict(getattr(response, "response_metadata", None) or {})
            metadata["harness_call_id"] = call_id
            metadata["harness_trace_sequence"] = sequence
            response.response_metadata = metadata
        except Exception:
            pass
    except Exception:
        pass
    return response


def record_parse_result(
    call_id: str | None,
    *,
    parsed_result: Any = None,
    status: str,
    error: Any = None,
    repair_call_id: str | None = None,
) -> None:
    context = current_audit_context()
    job_id = str(context.get("job_id") or "")
    if not trace_enabled() or not job_id or not call_id:
        return
    try:
        call_context = _call_trace_context(job_id, str(call_id))
        _append_trace(job_id, {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event": "model_parse",
            "sequence": _locked_sequence(job_id),
            "trace_id": call_context.get("trace_id") or context.get("trace_id"),
            "call_id": str(call_id),
            "repair_call_id": repair_call_id,
            "job_id": job_id,
            "parent_job_id": call_context.get("parent_job_id") or context.get("parent_job_id"),
            "run_id": call_context.get("run_id") or context.get("run_id"),
            "parent_call_id": call_context.get("parent_call_id"),
            "stage": call_context.get("stage") or context.get("stage") or "research_model",
            "question_id": call_context.get("question_id") or context.get("question_id"),
            "section_id": call_context.get("section_id") or context.get("section_id"),
            "attempt": call_context.get("attempt") or context.get("attempt") or 1,
            "timestamp": _timestamp(),
            "status": str(status),
            "parsed_result": sanitized_json_safe(parsed_result),
            "parse_error": sanitized_json_safe(error) if error is not None else None,
            "parsed_output_hash": stable_hash(parsed_result) if parsed_result is not None else None,
        })
    except Exception:
        return


def trace_path(job_id: str) -> Path:
    return _job_audit_dir(job_id) / "model_calls.jsonl"


def read_trace_events(job_id: str) -> list[dict[str, Any]]:
    path = trace_path(job_id)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
