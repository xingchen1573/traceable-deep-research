"""Proactive knowledge-card push service (docs/VISION.md 第5节).

Independent of the request/response cycle: a scheduler thread (start_push_scheduler below)
wakes up periodically and calls run_push_cycle(), which fans out over every channel that (a)
has a persistent topic binding (docs/VISION.md 第3节, backend/services/channel_service.py's
`topic_binding` column) and (b) has push explicitly enabled (default off, see
channel_service.get_push_config), condenses one document from that channel's bound
knowledge-base topic into a short "knowledge card", and delivers it via
backend/services/feishu_service.py::send_reply -- the exact same context-free send path
feishu_service already uses, just triggered by a clock tick instead of an inbound message.

Each channel owns a wall-clock time, IANA timezone, and persisted next due timestamp. Frequency
tiers still map to a minimum elapsed-hours gap between successful deliveries, while the persisted
timestamp lets a restarted scheduler catch up instead of losing a narrow clock window.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from . import channel_service

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent.parent / "knowledge_base"

_CARD_MAX_BODY_CHARS = 6000


def _research_service():
    # Lazy import, matching tools/research_jobs.py::_research_service()'s convention for
    # reaching backend.services.research_service -- avoids forcing that (large) module to load
    # at push_service import time and stays consistent with the project-wide pattern for
    # cross-service-module calls.
    try:
        from . import research_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_service
    return research_service


def _feishu_service():
    try:
        from . import feishu_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import feishu_service
    return feishu_service


def _knowledge_scope_service():
    try:
        from . import knowledge_scope_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import knowledge_scope_service
    return knowledge_scope_service


def _knowledge_tree_service():
    try:
        from . import knowledge_tree_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import knowledge_tree_service
    return knowledge_tree_service


# ---------------------------------------------------------------------------
# Document selection
# ---------------------------------------------------------------------------


_AUXILIARY_DOCUMENT_TITLES = {
    "入库建议",
    "knowledgeqa",
    "证据与参考",
    "关键evidence",
    "结构化sources",
    "references",
    "sources",
    "evidence",
    "来源索引",
    "证据索引",
}


def _normalized_document_title(path: Path) -> str:
    """Return a role-comparison key without section numbering or visual separators."""
    title = unicodedata.normalize("NFKC", path.stem).strip().casefold()
    title = re.sub(r"^\d{1,3}(?:\s*[_\-.:：、]\s*|\s+)", "", title)
    return re.sub(r"[\W_]+", "", title)


def _is_auxiliary_document(path: Path) -> bool:
    """Whether a Markdown document is unsuitable as a standalone learning-card source.

    Overview matching intentionally retains the legacy heuristic. Other roles are exact after
    normalization so substantive titles such as ``证据原生流水线`` or ``Source routing`` remain
    eligible for learning cards.
    """
    stem = unicodedata.normalize("NFKC", path.stem).strip().casefold()
    normalized = _normalized_document_title(path)
    if re.match(r"^00(?:\s*[_\-.:：、]\s*|\s+)", stem):
        return True
    if "总览" in normalized or "overview" in normalized:
        return True
    if normalized.startswith(("summary", "摘要", "总结")):
        return True
    return normalized in _AUXILIARY_DOCUMENT_TITLES


def _theme_subtopic_dirs(theme_dir: Path) -> list[Path]:
    """A subdirectory counts as a subtopic iff it directly holds Markdown docs -- same
    structural rule as tools/knowledge_manager.py::_theme_subtopic_dirs and
    tools/_registry.py::discover_topic_dirs (docs/VISION.md 第1节). Duplicated locally rather
    than imported: push_service.py deliberately avoids module-level `tools.*` imports (see
    _research_service()/_feishu_service() lazy-import convention above) to stay out of the
    tools/__init__.py -> backend.services -> workers -> tools load cycle."""
    return sorted(sub for sub in theme_dir.iterdir() if sub.is_dir() and list(sub.glob("*.md")))


def _rotation_pool(topic: str) -> list[Path]:
    """A single-segment `topic` may itself be a theme container (docs/VISION.md 第5节 主题级
    聚合轮询) -- if so, aggregate every subtopic's documents into one pool instead of the
    (empty) direct glob. A `theme/subtopic` two-segment topic is always a leaf and keeps the
    original single-directory behavior unchanged."""
    topic_dir = KNOWLEDGE_BASE / topic
    if not topic_dir.is_dir():
        return []
    if "/" not in topic:
        subtopic_dirs = _theme_subtopic_dirs(topic_dir)
        if subtopic_dirs:
            pool = [p for sub in subtopic_dirs for p in sub.glob("*.md") if not _is_auxiliary_document(p)]
            return sorted(pool, key=lambda p: p.relative_to(topic_dir).as_posix())
    return sorted(p for p in topic_dir.glob("*.md") if not _is_auxiliary_document(p))


def _document_identity(document: Path, topic: str) -> str:
    """Rotation-continuity key: equals the bare filename for a leaf topic pool (zero behavior
    change there) and "subtopic-dir/filename.md" for a theme-aggregated pool -- both are just
    the doc's path relative to the topic's own root dir, so one formula covers both shapes
    without the bare-filename identity colliding across subtopics that happen to share a
    filename (e.g. two subtopics each with their own "01_...")."""
    return document.relative_to(KNOWLEDGE_BASE / topic).as_posix()


def select_next_document(
    topic: str,
    *,
    content_strategy: str,
    last_pushed_doc: str | None,
    last_pushed_at: float | None,
) -> Path | None:
    pool = _rotation_pool(topic)
    if not pool:
        return None

    if content_strategy == "random":
        return random.choice(pool)

    new_docs = [p for p in pool if p.stat().st_mtime > (last_pushed_at or 0)]

    if content_strategy == "new_only":
        return min(new_docs, key=lambda p: p.stat().st_mtime) if new_docs else None

    # "rotation" (default): 新增插队 -- an unseen-since-last-push doc jumps the queue; the
    # oldest one is always pushed first, so multiple new docs queue up one per cycle instead of
    # all landing at once. With no new docs, continue the normal rotation order.
    if new_docs:
        return min(new_docs, key=lambda p: p.stat().st_mtime)

    names = [_document_identity(p, topic) for p in pool]
    if last_pushed_doc in names:
        next_index = (names.index(last_pushed_doc) + 1) % len(pool)
        return pool[next_index]
    return pool[0]


def _scope_rotation_pool(scope_type: str, scope_id: str) -> tuple[str, list[dict[str, Any]]]:
    scope = _knowledge_scope_service().resolve_scope(scope_type, scope_id)
    root = _knowledge_tree_service().KNOWLEDGE_ROOT
    pool: list[dict[str, Any]] = []
    for research in scope.get("research_items") or []:
        research_path = root / str(research["path"])
        declared = research.get("documents") if isinstance(research.get("documents"), list) else []
        paths = [research_path / str(item.get("filename") or "") for item in declared if item.get("filename")]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            paths = sorted(research_path.glob("*.md"))
        for path in paths:
            if _is_auxiliary_document(path):
                continue
            pool.append(
                {
                    "path": path,
                    "identity": f"{research['id']}/{path.name}",
                    "research_id": research["id"],
                    "research_title": research.get("title"),
                    "researched_at": research.get("researched_at"),
                }
            )
    pool.sort(key=lambda item: (str(item.get("researched_at") or ""), str(item["identity"])), reverse=True)
    return str(scope.get("title") or scope_id), pool


def select_next_scope_document(
    scope_type: str,
    scope_id: str,
    *,
    content_strategy: str,
    last_pushed_doc: str | None,
    last_pushed_at: float | None,
) -> tuple[str, dict[str, Any] | None]:
    title, pool = _scope_rotation_pool(scope_type, scope_id)
    if not pool:
        return title, None
    if content_strategy == "random":
        return title, random.choice(pool)
    new_items = [item for item in pool if item["path"].stat().st_mtime > (last_pushed_at or 0)]
    if content_strategy == "new_only":
        return title, min(new_items, key=lambda item: item["path"].stat().st_mtime) if new_items else None
    if new_items:
        return title, min(new_items, key=lambda item: item["path"].stat().st_mtime)
    identities = [str(item["identity"]) for item in pool]
    if last_pushed_doc in identities:
        return title, pool[(identities.index(last_pushed_doc) + 1) % len(pool)]
    return title, pool[0]


# ---------------------------------------------------------------------------
# Card generation + rendering
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def generate_knowledge_card(document_path: Path, *, topic: str) -> dict[str, Any]:
    """Condense one knowledge_base document into a short card: one headline + 1-2 pieces of
    evidence -- not a raw excerpt (docs/VISION.md 第5节: "AI 现场提炼的知识卡片...不是原文摘录")."""
    raw = document_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    title = frontmatter.get("title") or document_path.stem

    payload = {
        "task": (
            "Read the source document below (one knowledge-base entry) and distill it into a "
            "short spaced-repetition style card: one headline capturing the single most useful "
            "takeaway, plus 1-2 short pieces of supporting evidence drawn from the document. "
            "Paraphrase concisely -- this is a reminder card, not a full summary."
        ),
        "topic": topic,
        "document_title": title,
        "document_body": body.strip()[:_CARD_MAX_BODY_CHARS],
        "rules": [
            "只依据给定文档内容，不编造材料之外的事实。",
            "headline 一到两句话，是核心结论，不是文档标题的复述。",
            "evidence 是 1-2 条具体、简短的支撑点，不是整段原文摘录。",
            "全部使用中文。",
        ],
        "schema": {"headline": "一到两句话的核心结论", "evidence": ["具体支撑点", "..."]},
    }
    card = _research_service()._research_writer_json_model(payload)
    return {
        "headline": str(card.get("headline") or "").strip(),
        "evidence": [str(item).strip() for item in (card.get("evidence") or []) if str(item).strip()],
    }


def render_push_message(card: dict[str, Any], *, topic: str, document_title: str) -> str:
    lines = [f"📚 知识卡片 · {topic}", "", document_title, ""]
    headline = card.get("headline") or ""
    lines.append(headline or "（本轮未能提炼出核心结论）")
    evidence = card.get("evidence") or []
    if evidence:
        lines.append("")
        lines.append("依据：")
        lines.extend(f"- {item}" for item in evidence)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _is_due(*, now_ts: float, next_push_at: float | None) -> bool:
    """Pure function -- takes now_ts explicitly so tests never need to mock time.time()."""
    return next_push_at is not None and now_ts >= float(next_push_at)


def run_push_cycle(*, now_ts: float | None = None) -> list[dict[str, Any]]:
    """Orchestration entry point -- called by both the scheduler thread and tests."""
    if not _PUSH_CYCLE_LOCK.acquire(blocking=False):
        return []
    try:
        return _run_push_cycle(now_ts=now_ts)
    finally:
        _PUSH_CYCLE_LOCK.release()


def _run_push_cycle(*, now_ts: float | None = None) -> list[dict[str, Any]]:
    """Execute one non-overlapping scheduler tick."""
    now_ts = now_ts if now_ts is not None else time.time()

    results: list[dict[str, Any]] = []
    for channel in channel_service.list_push_candidates():
        if channel.get("platform") != "feishu":
            continue  # no other platform has a proactive-send path today
        channel_id = channel["channel_id"]
        try:
            result = _process_channel(channel, now_ts=now_ts)
        except Exception as exc:  # noqa: BLE001 - one channel's failure must not sink the cycle
            emit_event(
                "push.channel_failed",
                "push_service",
                "Push cycle failed for one channel.",
                data={"channel_id": channel_id, "reason": str(exc)[:300]},
                level="error",
            )
            result = {"channel_id": channel_id, "status": "error", "reason": str(exc)[:300]}
        results.append(result)
    return results


def _process_channel(channel: dict[str, Any], *, now_ts: float) -> dict[str, Any]:
    channel_id = channel["channel_id"]
    next_push_at = channel.get("next_push_at")
    if next_push_at is None:
        next_push_at = channel_service.initialize_next_push_at(channel_id, now_ts=now_ts)
    if not _is_due(now_ts=now_ts, next_push_at=next_push_at):
        return {"channel_id": channel_id, "status": "not_due"}

    scope_id = str(channel.get("scope_id") or "").strip()
    scope_type = str(channel.get("scope_type") or "").strip()
    if scope_id and scope_type:
        topic, selected = select_next_scope_document(
            scope_type,
            scope_id,
            content_strategy=channel.get("content_strategy") or "rotation",
            last_pushed_doc=channel.get("last_pushed_doc"),
            last_pushed_at=channel.get("last_pushed_at"),
        )
        document = selected["path"] if selected else None
        document_identity = str(selected["identity"]) if selected else ""
    else:
        topic = channel["topic_binding"]
        document = select_next_document(
            topic,
            content_strategy=channel.get("content_strategy") or "rotation",
            last_pushed_doc=channel.get("last_pushed_doc"),
            last_pushed_at=channel.get("last_pushed_at"),
        )
        document_identity = _document_identity(document, topic) if document else ""
    if document is None:
        # No eligible document is a completed schedule occurrence, not a transient failure.
        # Advance the cursor so a 10-minute scheduler tick does not repeatedly re-check it.
        channel_service.advance_push_schedule(channel_id, after_ts=now_ts)
        return {"channel_id": channel_id, "status": "no_content"}

    card = generate_knowledge_card(document, topic=topic)
    frontmatter, _ = _split_frontmatter(document.read_text(encoding="utf-8"))
    document_title = frontmatter.get("title") or document.stem
    message = render_push_message(card, topic=topic, document_title=document_title)

    _feishu_service().send_reply(chat_id=channel["external_id"], text=message)
    # record_push stores the qualified rotation-continuity identity (see _document_identity --
    # bare filename for a leaf topic, "subtopic/filename.md" for a theme pool); the result dict
    # and event payload below deliberately keep the bare filename instead, since those are for
    # human/log display, not for resolving `last_pushed_doc` on the next cycle.
    channel_service.record_push(channel_id, document=document_identity, pushed_at=now_ts)
    emit_event(
        "push.channel_pushed",
        "push_service",
        "Pushed a knowledge card to a channel.",
        data={"channel_id": channel_id, "topic": topic, "scope_type": scope_type or None, "scope_id": scope_id or None, "document": document.name},
    )
    return {"channel_id": channel_id, "status": "pushed", "document": document.name}


_PUSH_CYCLE_LOCK = threading.Lock()
_SCHEDULER_STATE_LOCK = threading.Lock()
_SCHEDULER_STOP_EVENT = threading.Event()
_SCHEDULER_THREAD: threading.Thread | None = None


def _scheduler_enabled() -> bool:
    return os.getenv("PUSH_SCHEDULER_ENABLED", "").lower() in {"1", "true", "yes"}


def scheduler_status() -> dict[str, Any]:
    interval = float(os.getenv("PUSH_TICK_INTERVAL_SECONDS", "600"))
    return {
        "enabled": _scheduler_enabled(),
        "running": bool(_SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive()),
        "tick_interval_seconds": interval,
    }


def _scheduler_loop() -> None:
    interval = scheduler_status()["tick_interval_seconds"]
    while not _SCHEDULER_STOP_EVENT.is_set():
        try:
            run_push_cycle()
        except Exception as exc:  # noqa: BLE001 - a tick failure must not kill the loop
            emit_event(
                "push.scheduler_tick_failed",
                "push_service",
                "Push scheduler tick raised.",
                data={"reason": str(exc)[:300]},
                level="error",
            )
        if _SCHEDULER_STOP_EVENT.wait(interval):
            break


def start_push_scheduler() -> None:
    """Starts the in-process push scheduler thread, gated by PUSH_SCHEDULER_ENABLED (default
    off) so importing backend.app in dev/test never silently spins up a background thread and
    real model/Feishu calls -- see docs/VISION.md 第5节."""
    global _SCHEDULER_THREAD
    if not _scheduler_enabled():
        return
    with _SCHEDULER_STATE_LOCK:
        if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
            return
        _SCHEDULER_STOP_EVENT.clear()
        thread = threading.Thread(target=_scheduler_loop, daemon=True, name="knowledge-push-scheduler")
        _SCHEDULER_THREAD = thread
        try:
            thread.start()
        except Exception:
            _SCHEDULER_THREAD = None
            _SCHEDULER_STOP_EVENT.set()
            raise


def stop_push_scheduler(*, timeout: float = 5.0) -> None:
    """Request scheduler shutdown and wait briefly; safe to call repeatedly."""
    global _SCHEDULER_THREAD
    with _SCHEDULER_STATE_LOCK:
        thread = _SCHEDULER_THREAD
        _SCHEDULER_STOP_EVENT.set()
        if thread is None:
            return
        if thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        if not thread.is_alive():
            _SCHEDULER_THREAD = None
