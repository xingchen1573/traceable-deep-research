"""File-based memory governance for Deep Research."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .observability import emit_event
    from .prompts import AUTODREAM_PROMPT_TEMPLATE
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event
    from prompts import AUTODREAM_PROMPT_TEMPLATE


ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "memory"
TOPICS_DIR = MEMORY_DIR / "topics"
DAILY_LOGS_DIR = MEMORY_DIR / "daily_logs"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
DREAM_REPORT = MEMORY_DIR / "dream_report.md"
DREAM_STATE = MEMORY_DIR / "dream_state.json"


@dataclass
class TopicRecord:
    topic_id: str
    title: str
    summary: str
    path: str
    trust_level: str = "medium"
    updated_at: str = ""


def default_dream_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "report_path": str(DREAM_REPORT.relative_to(ROOT)),
        "generated_at": None,
        "logs_until": None,
        "reviewed_at": None,
        "review_decision": None,
        "review_note": None,
        "last_dream_date": None,
    }


class SurfacedMemoryRegistry:
    """Track topic bodies already surfaced in the current process."""

    def __init__(self) -> None:
        self._surfaced: defaultdict[str, set[str]] = defaultdict(set)

    def mark(self, session_id: str, topic_id: str) -> None:
        safe_id = sanitize_topic_id(topic_id)
        if safe_id:
            self._surfaced[session_id or "default"].add(safe_id)

    def mark_many(self, session_id: str, topic_ids: list[str]) -> None:
        for topic_id in topic_ids:
            self.mark(session_id, topic_id)

    def is_surfaced(self, session_id: str, topic_id: str) -> bool:
        return sanitize_topic_id(topic_id) in self._surfaced.get(session_id or "default", set())

    def surfaced(self, session_id: str) -> set[str]:
        return set(self._surfaced.get(session_id or "default", set()))


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def today_text() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def sanitize_topic_id(topic_id: str) -> str:
    return "".join(ch for ch in topic_id.strip() if ch.isalnum() or ch in "-_").strip("-_")


def _jsonl_read(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows[-limit:] if limit else rows


def _jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    return [
        part
        for part in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", lowered)
        if len(part) >= 2
    ]


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta


def _first_content_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().strip("#").strip()
        if stripped and stripped != "---" and ":" not in stripped:
            return stripped[:120]
    return "No summary yet."


def _topic_records() -> list[TopicRecord]:
    records: list[TopicRecord] = []
    for path in sorted(TOPICS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        records.append(
            TopicRecord(
                topic_id=path.stem,
                title=str(meta.get("title") or path.stem),
                summary=str(meta.get("summary") or _first_content_line(text)),
                trust_level=str(meta.get("trust_level") or "medium"),
                path=f"memory/topics/{path.name}",
                updated_at=str(meta.get("updated_at") or now_text()),
            )
        )
    return records


def list_memory_topics() -> list[TopicRecord]:
    """Return topic index metadata without loading all topic bodies."""
    ensure_memory_files()
    return _topic_records()


def load_memory_topics(topic_ids: list[str]) -> list[tuple[TopicRecord, str]]:
    """Safely read multiple topic bodies by topic id."""
    ensure_memory_files()
    records = {record.topic_id: record for record in _topic_records()}
    loaded: list[tuple[TopicRecord, str]] = []
    seen: set[str] = set()
    for topic_id in topic_ids:
        safe_id = sanitize_topic_id(topic_id)
        if not safe_id or safe_id in seen:
            continue
        record = records.get(safe_id)
        if record is None:
            continue
        topic_path = ROOT / record.path
        if not topic_path.exists():
            continue
        loaded.append((record, topic_path.read_text(encoding="utf-8")))
        seen.add(safe_id)
    return loaded


def ensure_memory_files() -> None:
    """Create the MVP memory layout and seed examples if missing."""
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not DREAM_STATE.exists():
        DREAM_STATE.write_text(
            json.dumps(default_dream_state(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    seed_topic = TOPICS_DIR / "memory_policy.md"
    if not seed_topic.exists():
        seed_topic.write_text(
            "---\n"
            "title: memory_policy\n"
            "summary: Memory is a hint, not truth; verify before acting.\n"
            "trust_level: high\n"
            f"updated_at: {now_text()}\n"
            "---\n\n"
            "# Memory Policy\n\n"
            "- Long-term memory is a navigational hint, not a source of truth.\n"
            "- Verify code paths, package choices, and external facts before acting.\n"
            "- Formal memory writes must create the topic file first, then refresh MEMORY.md.\n"
            "- AutoDream reviews daily logs and writes dream_report.md; it does not directly update formal memory.\n"
            "- Good long-term memory includes stable user profile facts, durable preferences, working rules, and verified project conventions.\n"
            "- Do not store one-off requests, secrets, credentials, sensitive private data, or unverified project facts.\n",
            encoding="utf-8",
        )

    refresh_memory_index()


def read_dream_state() -> dict[str, Any]:
    ensure_memory_files()
    try:
        data = json.loads(DREAM_STATE.read_text(encoding="utf-8"))
    except Exception:
        data = default_dream_state()
    state = default_dream_state()
    if isinstance(data, dict):
        state.update(data)
    return state


def write_dream_state(state: dict[str, Any]) -> dict[str, Any]:
    merged = default_dream_state()
    merged.update(state)
    DREAM_STATE.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    emit_event(
        "dream.state.updated",
        "memory",
        "Dream state was updated.",
        data={"status": merged.get("status"), "review_decision": merged.get("review_decision")},
    )
    return merged


def pending_dream_review_exists() -> bool:
    return read_dream_state().get("status") == "pending_review"


def latest_daily_log_date() -> str | None:
    ensure_memory_files()
    paths = sorted(DAILY_LOGS_DIR.glob("*.jsonl"))
    return paths[-1].stem if paths else None


def mark_dream_pending(logs_until: str | None) -> dict[str, Any]:
    state = read_dream_state()
    state.update(
        {
            "status": "pending_review",
            "report_path": str(DREAM_REPORT.relative_to(ROOT)),
            "generated_at": now_text(),
            "logs_until": logs_until,
            "reviewed_at": None,
            "review_decision": None,
            "review_note": None,
            "last_dream_date": logs_until,
        }
    )
    return write_dream_state(state)


def close_dream_review(decision: str, note: str | None = None) -> dict[str, Any]:
    state = read_dream_state()
    state.update(
        {
            "status": "closed",
            "reviewed_at": now_text(),
            "review_decision": decision,
            "review_note": note,
        }
    )
    emit_event(
        "dream.reviewed",
        "memory",
        "Dream review decision was recorded.",
        level="info",
        data={"decision": decision, "has_note": bool(note)},
    )
    return write_dream_state(state)


def should_run_autodream_tick() -> tuple[bool, str, str | None]:
    state = read_dream_state()
    latest_log = latest_daily_log_date()
    if state.get("status") == "pending_review":
        return False, "pending_review", latest_log
    if latest_log is None:
        return False, "no_daily_logs", None
    if state.get("last_dream_date") == latest_log:
        return False, "no_new_logs", latest_log
    return True, "ready", latest_log


def refresh_memory_index() -> str:
    """Regenerate MEMORY.md from topic files."""
    records = _topic_records()
    lines = [
        "# Deep Research Memory Index",
        "",
        "> Long-term memory index. This file is injected as lightweight navigation.",
        "> Topic bodies are loaded only when relevant. Memory is a hint, not truth.",
        "",
        "## Topics",
    ]
    if not records:
        lines.append("- No topics yet.")
    for record in records:
        lines.append(
            "- "
            f"**{record.topic_id}** | title={record.title} | trust={record.trust_level} | "
            f"updated={record.updated_at} | path={record.path} | {record.summary}"
        )
    lines.extend(
        [
            "",
            "## Rules",
            "- Good long-term memory: stable user profile facts, durable preferences, working rules, verified project conventions, and compact lessons from completed tasks.",
            "- Bad long-term memory: one-off requests, temporary state, secrets, credentials, sensitive private data, and unverified project facts.",
            "- Formal write path: write topic file first, then refresh this MEMORY.md index.",
            "- Before using memory, verify facts that may change.",
            "",
            "## Logs",
            "- Daily logs: memory/daily_logs/YYYY-MM-DD.jsonl",
            "- Dream report: memory/dream_report.md",
        ]
    )
    MEMORY_INDEX.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return MEMORY_INDEX.read_text(encoding="utf-8")


def read_memory_index() -> str:
    ensure_memory_files()
    return MEMORY_INDEX.read_text(encoding="utf-8")


def read_dream_report() -> str:
    ensure_memory_files()
    return DREAM_REPORT.read_text(encoding="utf-8") if DREAM_REPORT.exists() else ""


def search_memory(query: str, limit: int = 3) -> str:
    """Search topic metadata and topic bodies without loading every topic body."""
    ensure_memory_files()
    query_tokens = _tokenize(query)
    scored: list[tuple[int, TopicRecord, str]] = []
    for record in _topic_records():
        topic_path = ROOT / record.path
        body = topic_path.read_text(encoding="utf-8") if topic_path.exists() else ""
        haystack = f"{record.topic_id} {record.title} {record.summary} {body[:1200]}".lower()
        score = sum(1 for token in query_tokens if token in haystack)
        if score:
            scored.append((score, record, body[:500]))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return "No relevant memory topics found. Treat memory as absent."
    lines = ["Relevant memory topics:"]
    for score, record, preview in scored[:limit]:
        lines.append(
            f"- {record.topic_id} (score={score}, trust={record.trust_level}, path={record.path})\n"
            f"  summary: {record.summary}\n"
            f"  preview: {preview.strip()[:240]}"
        )
    return "\n".join(lines)


def load_memory_topic(topic_id: str) -> str:
    ensure_memory_files()
    safe_id = sanitize_topic_id(topic_id)
    path = TOPICS_DIR / f"{safe_id}.md"
    if not path.exists():
        return f"Memory topic not found: {topic_id}"
    return path.read_text(encoding="utf-8")


def write_memory_topic(
    topic_id: str,
    title: str,
    summary: str,
    body: str,
    trust_level: str = "medium",
) -> str:
    """Write a formal topic file first, then refresh MEMORY.md.

    This is the formal memory write path. AutoDream is review-only and writes
    dream_report.md, not formal topic files.
    """
    ensure_memory_files()
    safe_id = sanitize_topic_id(topic_id)
    if not safe_id:
        return "Denied: topic_id must contain letters, numbers, dash, or underscore."
    clean_trust = trust_level if trust_level in {"low", "medium", "high"} else "medium"
    topic_path = TOPICS_DIR / f"{safe_id}.md"
    payload = (
        "---\n"
        f"title: {title.strip()[:120] or safe_id}\n"
        f"summary: {summary.strip()[:240] or 'No summary.'}\n"
        f"trust_level: {clean_trust}\n"
        f"updated_at: {now_text()}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )
    topic_path.write_text(payload, encoding="utf-8")
    refresh_memory_index()
    return (
        f"Wrote formal memory topic memory/topics/{topic_path.name} and refreshed MEMORY.md. "
        "Remember: memory is a hint, not truth; verify before use."
    )


def append_daily_log_turn(session_id: str, user_message: str, assistant_summary: str) -> None:
    ensure_memory_files()
    date_text = today_text()
    path = DAILY_LOGS_DIR / f"{date_text}.jsonl"
    _jsonl_append(
        path,
        {
            "timestamp": now_text(),
            "date": date_text,
            "session_id": session_id,
            "user": user_message[:1200],
            "assistant_summary": assistant_summary[:1200],
        },
    )
    emit_event(
        "memory.daily_log.appended",
        "memory",
        "Appended one turn to the daily memory log.",
        session_id=session_id,
        data={
            "path": str(path.relative_to(ROOT)),
            "user_chars": len(user_message),
            "assistant_summary_chars": len(assistant_summary),
        },
    )


def read_recent_daily_logs(max_files: int = 7, limit: int = 120) -> list[dict[str, Any]]:
    ensure_memory_files()
    rows: list[dict[str, Any]] = []
    for path in sorted(DAILY_LOGS_DIR.glob("*.jsonl"))[-max_files:]:
        rows.extend(_jsonl_read(path))
    return rows[-limit:]


def run_autodream(dream_model: Any | None = None) -> str:
    """Manual LLM AutoDream pass: review-only, never mutates formal topics."""
    ensure_memory_files()
    emit_event("dream.started", "memory", "AutoDream review-only pass started.")
    daily_logs = read_recent_daily_logs()
    emit_event(
        "dream.daily_logs.loaded",
        "memory",
        "Loaded recent daily logs for AutoDream.",
        data={"entries": len(daily_logs)},
    )
    if not daily_logs:
        DREAM_REPORT.write_text(_empty_dream_report(), encoding="utf-8")
        emit_event(
            "dream.report.generated",
            "memory",
            "Generated AutoDream report without daily logs.",
            data={"path": str(DREAM_REPORT.relative_to(ROOT)), "entries": 0},
        )
        mark_dream_pending(logs_until=latest_daily_log_date())
        return str(DREAM_REPORT.relative_to(ROOT))
    if dream_model is None:
        DREAM_REPORT.write_text(_missing_model_dream_report(daily_logs), encoding="utf-8")
        emit_event(
            "dream.report.generated",
            "memory",
            "Generated AutoDream report without a dream model.",
            level="warning",
            data={"path": str(DREAM_REPORT.relative_to(ROOT)), "entries": len(daily_logs)},
        )
        mark_dream_pending(logs_until=latest_daily_log_date())
        return str(DREAM_REPORT.relative_to(ROOT))

    prompt = build_autodream_prompt(daily_logs)
    response = dream_model.invoke(prompt)
    report = _content_from_response(response).strip() or "No memory changes needed."
    DREAM_REPORT.write_text(_normalize_dream_report(report), encoding="utf-8")
    emit_event(
        "dream.report.generated",
        "memory",
        "Generated AutoDream report with dream model.",
        data={
            "path": str(DREAM_REPORT.relative_to(ROOT)),
            "entries": len(daily_logs),
            "report_chars": len(report),
        },
    )
    mark_dream_pending(logs_until=latest_daily_log_date())
    return str(DREAM_REPORT.relative_to(ROOT))


def build_autodream_prompt(daily_logs: list[dict[str, Any]]) -> str:
    logs_text = json.dumps(daily_logs, ensure_ascii=False, indent=2)
    return AUTODREAM_PROMPT_TEMPLATE.format(
        memory_root=MEMORY_DIR.relative_to(ROOT),
        current_time=now_text(),
        memory_index=read_memory_index(),
        topic_bodies=_topic_bodies_for_dream(),
        daily_logs=logs_text,
    )


def _topic_bodies_for_dream(max_chars_per_topic: int = 3000) -> str:
    sections: list[str] = []
    for record in _topic_records():
        topic_path = ROOT / record.path
        body = topic_path.read_text(encoding="utf-8") if topic_path.exists() else ""
        sections.append(
            f"### {record.topic_id}\n"
            f"- path: {record.path}\n"
            f"- title: {record.title}\n"
            f"- trust_level: {record.trust_level}\n"
            f"- updated_at: {record.updated_at}\n"
            f"- summary: {record.summary}\n\n"
            f"{body[:max_chars_per_topic]}"
        )
    return "\n\n".join(sections) if sections else "No topic memories yet."


def _content_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content)


def _normalize_dream_report(report: str) -> str:
    header = (
        "# AutoDream Review Report\n\n"
        "> Review-only report. It does not rewrite MEMORY.md or formal topics.\n"
        f"> Generated at: {now_text()}\n\n"
    )
    body = report.strip()
    if body.startswith("# AutoDream Review Report"):
        return body + "\n"
    return header + body + "\n"


def _empty_dream_report() -> str:
    return _normalize_dream_report(
        "## Updated candidates\n"
        "No memory changes needed.\n\n"
        "## Created candidates\n"
        "No memory changes needed.\n\n"
        "## Removed or stale\n"
        "No memory changes needed.\n\n"
        "## Re-index suggestions\n"
        "No memory changes needed.\n\n"
        "## Conflicts\n"
        "No memory changes needed.\n\n"
        "## Summary\n"
        "No memory changes needed. No daily logs were available for review."
    )


def _missing_model_dream_report(daily_logs: list[dict[str, Any]]) -> str:
    return _normalize_dream_report(
        "## Updated candidates\n"
        "Not reviewed. No dream model was provided.\n\n"
        "## Created candidates\n"
        "Not reviewed. No dream model was provided.\n\n"
        "## Removed or stale\n"
        "Not reviewed. No dream model was provided.\n\n"
        "## Re-index suggestions\n"
        "Not reviewed. No dream model was provided.\n\n"
        "## Conflicts\n"
        "Not reviewed. No dream model was provided.\n\n"
        "## Summary\n"
        f"AutoDream found {len(daily_logs)} daily log entries, but no dream_model was provided."
    )
