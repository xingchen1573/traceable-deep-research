"""JSON-backed storage for the three eval case sets used by the evaluation jobs.

Cases used to be hardcoded Python literals (tools/knowledge_eval.py's
HARNESS_AGENT_EVAL_SET, scripts/run_research_multi_topic_probe.py's DEFAULT_CASES,
scripts/run_single_url_brief_eval.py's DEFAULT_CASES). They now live under
config/eval_cases/*.json so the frontend can list/edit/delete them without a
code change + redeploy.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:
    from ...tools._registry import ROOT
except ImportError:  # pragma: no cover
    from tools._registry import ROOT

EVAL_CASES_DIR = ROOT / "config" / "eval_cases"

# knowledge_qa is scoped by topic (with a "default" fallback list); the other two
# eval types are flat lists.
_TOPIC_SCOPED_TYPES = {"knowledge_qa"}
_FILES = {
    "knowledge_qa": EVAL_CASES_DIR / "knowledge_qa.json",
    "research_probe": EVAL_CASES_DIR / "research_probe.json",
    "single_url_brief": EVAL_CASES_DIR / "single_url_brief.json",
}


def _knowledge_qa_version_store(data: dict[str, Any]) -> dict[str, Any]:
    store = data.setdefault("topic_versions", {})
    return store if isinstance(store, dict) else {}


def _validate_eval_type(eval_type: str) -> Path:
    path = _FILES.get(eval_type)
    if path is None:
        raise ValueError(f"unsupported eval_type: {eval_type}")
    return path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {} if path.stem == "knowledge_qa" else []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise


def load_cases(eval_type: str, *, topic: str | None = None) -> list[dict[str, Any]]:
    path = _validate_eval_type(eval_type)
    data = _read_json(path)
    if eval_type in _TOPIC_SCOPED_TYPES:
        topics = data.get("topics") or {}
        if topic and topic in topics:
            return list(topics[topic])
        return list(data.get("default") or [])
    return list(data)


def has_topic_cases(eval_type: str, topic: str | None) -> bool:
    path = _validate_eval_type(eval_type)
    if eval_type not in _TOPIC_SCOPED_TYPES or not topic:
        return False
    data = _read_json(path)
    topics = data.get("topics") or {}
    return topic in topics and isinstance(topics.get(topic), list)


def list_scopes(eval_type: str) -> list[str]:
    _validate_eval_type(eval_type)
    if eval_type not in _TOPIC_SCOPED_TYPES:
        return []
    path = _FILES[eval_type]
    data = _read_json(path)
    return sorted((data.get("topics") or {}).keys())


def upsert_case(eval_type: str, case: dict[str, Any], *, topic: str | None = None) -> dict[str, Any]:
    if not case.get("id"):
        raise ValueError("case must include a non-empty 'id' field")
    path = _validate_eval_type(eval_type)
    data = _read_json(path)
    if eval_type in _TOPIC_SCOPED_TYPES:
        scope_key = topic or "default"
        if scope_key == "default":
            bucket = list(data.get("default") or [])
        else:
            topics = data.setdefault("topics", {})
            bucket = list(topics.get(scope_key) or [])
        bucket = [item for item in bucket if item.get("id") != case["id"]] + [case]
        if scope_key == "default":
            data["default"] = bucket
        else:
            data.setdefault("topics", {})[scope_key] = bucket
    else:
        bucket = [item for item in data if item.get("id") != case["id"]] + [case]
        data = bucket
    _write_json_atomic(path, data)
    return case


def replace_cases(eval_type: str, cases: list[dict[str, Any]], *, topic: str | None = None) -> list[dict[str, Any]]:
    path = _validate_eval_type(eval_type)
    normalized = [dict(case) for case in cases if isinstance(case, dict)]
    if not normalized:
        raise ValueError("cases must contain at least one item")
    ids = [str(case.get("id") or "").strip() for case in normalized]
    if any(not case_id for case_id in ids):
        raise ValueError("every case must include a non-empty 'id' field")
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")

    data = _read_json(path)
    if eval_type in _TOPIC_SCOPED_TYPES:
        scope_key = str(topic or "").strip()
        if not scope_key:
            raise ValueError(f"topic is required for eval_type: {eval_type}")
        data.setdefault("topics", {})[scope_key] = normalized
    else:
        data = normalized
    _write_json_atomic(path, data)
    return normalized


def activate_topic_version(
    topic: str,
    cases: list[dict[str, Any]],
    *,
    eval_set_id: str,
    version: int,
    case_set_hash: str,
    knowledge_content_hash: str,
    approved_at: str,
) -> dict[str, Any]:
    """Persist an immutable knowledge-QA case-set version and make it active."""
    path = _validate_eval_type("knowledge_qa")
    normalized = [dict(case) for case in cases if isinstance(case, dict)]
    if not normalized:
        raise ValueError("cases must contain at least one item")
    data = _read_json(path)
    data.setdefault("topics", {})[topic] = normalized
    store = _knowledge_qa_version_store(data)
    topic_store = store.setdefault(topic, {"active_version": version, "versions": []})
    versions = [
        item
        for item in (topic_store.get("versions") or [])
        if isinstance(item, dict) and int(item.get("version") or 0) != int(version)
    ]
    record = {
        "eval_set_id": eval_set_id,
        "version": int(version),
        "case_set_hash": case_set_hash,
        "knowledge_content_hash": knowledge_content_hash,
        "approved_at": approved_at,
        "case_count": len(normalized),
        "cases": normalized,
    }
    versions.append(record)
    versions.sort(key=lambda item: int(item.get("version") or 0))
    topic_store["active_version"] = int(version)
    topic_store["versions"] = versions
    _write_json_atomic(path, data)
    return record


def list_topic_versions(topic: str) -> list[dict[str, Any]]:
    path = _validate_eval_type("knowledge_qa")
    data = _read_json(path)
    topic_store = (_knowledge_qa_version_store(data).get(topic) or {})
    active_version = int(topic_store.get("active_version") or 0)
    records = []
    for raw in topic_store.get("versions") or []:
        if not isinstance(raw, dict):
            continue
        record = {key: value for key, value in raw.items() if key != "cases"}
        record["active"] = int(record.get("version") or 0) == active_version
        records.append(record)
    return sorted(records, key=lambda item: int(item.get("version") or 0), reverse=True)


def get_active_topic_version(topic: str) -> dict[str, Any] | None:
    path = _validate_eval_type("knowledge_qa")
    data = _read_json(path)
    topic_store = (_knowledge_qa_version_store(data).get(topic) or {})
    active_version = int(topic_store.get("active_version") or 0)
    for raw in topic_store.get("versions") or []:
        if isinstance(raw, dict) and int(raw.get("version") or 0) == active_version:
            return dict(raw)
    return None


def switch_topic_version(topic: str, version: int) -> dict[str, Any]:
    path = _validate_eval_type("knowledge_qa")
    data = _read_json(path)
    topic_store = (_knowledge_qa_version_store(data).get(topic) or {})
    selected = next(
        (
            raw
            for raw in topic_store.get("versions") or []
            if isinstance(raw, dict) and int(raw.get("version") or 0) == int(version)
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"knowledge QA case-set version not found: {topic} v{version}")
    cases = [dict(case) for case in selected.get("cases") or [] if isinstance(case, dict)]
    if not cases:
        raise ValueError(f"knowledge QA case-set version has no cases: {topic} v{version}")
    data.setdefault("topics", {})[topic] = cases
    topic_store["active_version"] = int(version)
    _write_json_atomic(path, data)
    return {key: value for key, value in selected.items() if key != "cases"}


def delete_case(eval_type: str, case_id: str, *, topic: str | None = None) -> None:
    path = _validate_eval_type(eval_type)
    data = _read_json(path)
    if eval_type in _TOPIC_SCOPED_TYPES:
        scope_key = topic or "default"
        if scope_key == "default":
            data["default"] = [item for item in (data.get("default") or []) if item.get("id") != case_id]
        else:
            topics = data.setdefault("topics", {})
            topics[scope_key] = [item for item in (topics.get(scope_key) or []) if item.get("id") != case_id]
    else:
        data = [item for item in data if item.get("id") != case_id]
    _write_json_atomic(path, data)
