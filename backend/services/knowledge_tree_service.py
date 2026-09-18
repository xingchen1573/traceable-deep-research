"""Filesystem-backed personal knowledge hierarchy domain service.

This module implements the new KnowledgeBase -> Theme -> ResearchItem model in
parallel with the legacy topic service. Markdown remains the content source of
truth; readable directories and manifests are the hierarchy source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback for local development
    fcntl = None

try:
    from ...tools._registry import KNOWLEDGE_BASE as _DEFAULT_KNOWLEDGE_ROOT
except ImportError:  # pragma: no cover - top-level backend package fallback
    from tools._registry import KNOWLEDGE_BASE as _DEFAULT_KNOWLEDGE_ROOT


KNOWLEDGE_ROOT = _DEFAULT_KNOWLEDGE_ROOT
SCHEMA_VERSION = 1
KB_MANIFEST = "_kb.json"
THEME_MANIFEST = "_theme.json"
RESEARCH_MANIFEST = "_research.json"
TRASH_MANIFEST = "_trash.json"
OVERVIEW_FILENAME = "00_总览.md"
RESEARCH_HTML_FILENAME = "完整报告.html"
MANAGED_START = "<!-- harness-managed:start -->"
MANAGED_END = "<!-- harness-managed:end -->"

_OPERATIONS_DIR = ".operations"
_PREVIEWS_DIR = "previews"
_STAGING_DIR = ".staging"
_TRASH_DIR = ".trash"
_BACKUPS_DIR = "backups"
_PROCESS_LOCK = threading.RLock()
_INVALID_PATH_CHARS = re.compile(r"[\\/:\x00-\x1f]+")
_NUMBERED_CITATION_RE = re.compile(r"\[(\d+)\]")

NODE_MANIFESTS = {
    "knowledge_base": KB_MANIFEST,
    "theme": THEME_MANIFEST,
    "research": RESEARCH_MANIFEST,
}
SUPPORTED_ACTIONS = {
    "create_knowledge_base",
    "create_theme",
    "archive_research",
    "update_research_status",
    "rename_node",
    "move_research",
    "trash_node",
    "trash_legacy",
    "restore_node",
}


def list_tree(*, include_legacy: bool = True) -> dict[str, Any]:
    """Return the manifest-backed knowledge hierarchy and legacy root directories."""

    _ensure_layout()
    recover_incomplete_operations()
    knowledge_bases: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    for child in sorted(KNOWLEDGE_ROOT.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir() or _is_internal_path(child):
            continue
        manifest = _read_manifest(child / KB_MANIFEST, expected_type="knowledge_base", required=False)
        if manifest is None:
            if include_legacy:
                legacy.append(_legacy_entry(child))
            continue
        knowledge_bases.append(_knowledge_base_entry(child, manifest))
    knowledge_bases.sort(key=lambda item: (int(item.get("sort_order") or 0), str(item.get("title") or "").casefold()))
    return {
        "schema_version": SCHEMA_VERSION,
        "knowledge_bases": knowledge_bases,
        "legacy": legacy,
        "trash_count": len(list_trash()["items"]),
    }


def list_trash() -> dict[str, Any]:
    _ensure_layout()
    items: list[dict[str, Any]] = []
    trash_root = KNOWLEDGE_ROOT / _TRASH_DIR
    for type_dir in sorted(trash_root.iterdir(), key=lambda item: item.name) if trash_root.exists() else []:
        if not type_dir.is_dir():
            continue
        for item_dir in sorted(type_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not item_dir.is_dir():
                continue
            trash_meta = _read_json(item_dir / TRASH_MANIFEST, required=False) or {}
            if type_dir.name == "legacy":
                node = {
                    "type": "legacy",
                    "id": trash_meta.get("node_id"),
                    "title": trash_meta.get("title") or item_dir.name,
                    "path": item_dir,
                }
            else:
                node = _node_from_path(item_dir, expected_type=type_dir.name, include_children=False)
            if node is None:
                continue
            items.append(
                {
                    **node,
                    "trashed_at": trash_meta.get("trashed_at"),
                    "original_path": trash_meta.get("original_path"),
                    "trash_path": _relative(item_dir),
                }
            )
    items.sort(key=lambda item: str(item.get("trashed_at") or ""), reverse=True)
    return {"items": items, "count": len(items)}


def read_node_overview(node_type: str, node_id: str) -> dict[str, Any]:
    """Read a KnowledgeBase or Theme overview without exposing filesystem paths."""

    if node_type not in {"knowledge_base", "theme"}:
        raise ValueError("overview is only available for knowledge_base or theme")
    node = _find_node(node_id, expected_type=node_type)
    path = node["path"] / OVERVIEW_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"knowledge overview not found: {node_id}")
    return {
        "node_type": node_type,
        "node_id": node_id,
        "title": node.get("title") or node["manifest"].get("title"),
        "filename": OVERVIEW_FILENAME,
        "content": path.read_text(encoding="utf-8"),
    }


def read_research_document(research_id: str, filename: str) -> dict[str, Any]:
    """Read one manifest-listed Markdown document from a ResearchItem."""

    research = _find_node(research_id, expected_type="research")
    safe_name = Path(str(filename or "")).name
    if safe_name != filename or not safe_name.lower().endswith(".md"):
        raise ValueError("invalid research document filename")
    documents = research["manifest"].get("documents")
    document_meta = next(
        (
            item
            for item in documents
            if isinstance(item, dict) and str(item.get("filename") or "") == safe_name
        ),
        None,
    ) if isinstance(documents, list) else None
    allowed = {
        str(item.get("filename") or "")
        for item in documents
        if isinstance(item, dict)
    } if isinstance(documents, list) else set()
    if allowed and safe_name not in allowed:
        raise FileNotFoundError(f"research document not found: {safe_name}")
    path = research["path"] / safe_name
    if not path.is_file():
        raise FileNotFoundError(f"research document not found: {safe_name}")
    content = path.read_text(encoding="utf-8")
    html_filename = str((document_meta or {}).get("html_filename") or "").strip()
    return {
        "research_id": research_id,
        "title": research.get("title") or research["manifest"].get("title"),
        "filename": safe_name,
        "content": content,
        "has_html": bool(html_filename and (research["path"] / html_filename).is_file()),
        "html_filename": html_filename or None,
        "citations": _document_citations(content, document_meta or {}, research["manifest"]),
    }


def read_research_document_html(research_id: str, filename: str) -> dict[str, Any]:
    """Read the derived standalone HTML associated with one archived Markdown document."""

    research = _find_node(research_id, expected_type="research")
    safe_name = Path(str(filename or "")).name
    if safe_name != filename or not safe_name.lower().endswith(".md"):
        raise ValueError("invalid research document filename")
    documents = research["manifest"].get("documents")
    document_meta = next(
        (
            item
            for item in documents
            if isinstance(item, dict) and str(item.get("filename") or "") == safe_name
        ),
        None,
    ) if isinstance(documents, list) else None
    html_filename = str((document_meta or {}).get("html_filename") or "").strip()
    if not html_filename or Path(html_filename).name != html_filename or not html_filename.lower().endswith(".html"):
        raise FileNotFoundError(f"research document HTML not found: {safe_name}")
    path = research["path"] / html_filename
    if not path.is_file():
        raise FileNotFoundError(f"research document HTML not found: {safe_name}")
    return {
        "research_id": research_id,
        "filename": safe_name,
        "html_filename": html_filename,
        "content": path.read_text(encoding="utf-8"),
    }


def read_research_html(research_id: str) -> dict[str, Any]:
    """Read the complete derived HTML associated with one ResearchItem."""

    research = _find_node(research_id, expected_type="research")
    html_filename = str(research["manifest"].get("html_filename") or "").strip()
    if not html_filename or Path(html_filename).name != html_filename or not html_filename.lower().endswith(".html"):
        raise FileNotFoundError(f"research HTML not found: {research_id}")
    path = research["path"] / html_filename
    if not path.is_file():
        raise FileNotFoundError(f"research HTML not found: {research_id}")
    return {
        "research_id": research_id,
        "html_filename": html_filename,
        "content": path.read_text(encoding="utf-8"),
    }


def create_preview(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a durable preview without mutating the visible knowledge tree."""

    action_name = str(action or "").strip()
    if action_name not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported knowledge-tree action: {action_name}")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    _ensure_layout()
    with _tree_lock():
        recover_incomplete_operations()
        preview_id = f"kp_{uuid.uuid4().hex}"
        builder = {
            "create_knowledge_base": _preview_create_knowledge_base,
            "create_theme": _preview_create_theme,
            "archive_research": _preview_archive_research,
            "update_research_status": _preview_update_research_status,
            "rename_node": _preview_rename_node,
            "move_research": _preview_move_research,
            "trash_node": _preview_trash_node,
            "trash_legacy": _preview_trash_legacy,
            "restore_node": _preview_restore_node,
        }[action_name]
        plan = builder(dict(payload), preview_id)
        archive = plan.pop("archive", None)
        preview = {
            "schema_version": SCHEMA_VERSION,
            "preview_id": preview_id,
            "action": action_name,
            "state": "pending",
            "created_at": _now(),
            "changes": plan.pop("changes"),
            "expected": plan.pop("expected"),
            "plan": plan,
        }
        if archive is not None:
            preview["archive"] = archive
        _write_preview(preview)
        return _public_preview(preview)


def get_preview(preview_id: str) -> dict[str, Any]:
    preview = _read_preview(preview_id)
    return _public_preview(preview)


def confirm_preview(preview_id: str) -> dict[str, Any]:
    """Confirm and execute one preview under a cross-process filesystem lock."""

    _ensure_layout()
    with _tree_lock():
        recover_incomplete_operations()
        preview = _read_preview(preview_id)
        if preview.get("state") == "committed":
            return _public_preview(preview)
        if preview.get("state") != "pending":
            raise ValueError(f"preview is not pending: {preview.get('state')}")
        try:
            _validate_expected(preview.get("expected") or [])
        except ValueError as exc:
            preview["state"] = "stale"
            preview["error"] = str(exc)
            preview["updated_at"] = _now()
            _write_preview(preview)
            raise

        preview["state"] = "applying"
        preview["updated_at"] = _now()
        _write_preview(preview)
        try:
            result = _execute_plan(preview)
        except Exception as exc:
            preview["state"] = "failed"
            preview["error"] = str(exc) or type(exc).__name__
            preview["updated_at"] = _now()
            _write_preview(preview)
            raise
        preview["state"] = "committed"
        preview["result"] = result
        preview["confirmed_at"] = _now()
        preview["updated_at"] = preview["confirmed_at"]
        _write_preview(preview)
        return _public_preview(preview)


def confirm_research_archive(preview_id: str, *, source_job_id: str) -> dict[str, Any]:
    """Confirm an archive preview only when it belongs to the requesting research job."""

    preview = _read_preview(preview_id)
    if preview.get("action") != "archive_research":
        raise ValueError("preview is not a research archive")
    manifest = ((preview.get("plan") or {}).get("research_manifest") or {})
    if not manifest:
        result_path = _absolute(((preview.get("plan") or {}).get("result") or {}).get("path"))
        staged_target = _absolute((preview.get("plan") or {}).get("target"))
        staged_root = _absolute((preview.get("plan") or {}).get("stage"))
        try:
            relative = result_path.relative_to(staged_target)
        except ValueError as exc:
            raise ValueError("archive preview has an invalid research path") from exc
        manifest = _read_json(staged_root / relative / RESEARCH_MANIFEST, required=False) or {}
    if str(manifest.get("source_job_id") or "") != str(source_job_id or ""):
        raise ValueError("archive preview does not belong to this research job")
    return confirm_preview(preview_id)


def update_research_index_status(research_id: str, status: str, **details: Any) -> dict[str, Any]:
    """Update rebuildable index state without changing research content or hierarchy."""

    allowed = {"pending", "queued", "running", "ready", "failed", "stale", "disabled"}
    normalized = str(status or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"invalid research index status: {normalized}")
    with _tree_lock():
        research = _find_node(research_id, expected_type="research")
        manifest = dict(research["manifest"])
        manifest["index_status"] = {
            "status": normalized,
            **{key: value for key, value in details.items() if value is not None},
            "updated_at": _now(),
        }
        manifest["updated_at"] = _now()
        manifest["revision"] = int(manifest.get("revision") or 0) + 1
        _write_json(research["path"] / RESEARCH_MANIFEST, manifest)
        return {"research_id": research_id, "path": _relative(research["path"]), "index_status": manifest["index_status"]}


def attach_research_index_job(research_id: str, job_id: str) -> dict[str, Any]:
    """Attach an async job without moving a status that may already have advanced."""

    with _tree_lock():
        research = _find_node(research_id, expected_type="research")
        manifest = dict(research["manifest"])
        current = manifest.get("index_status") if isinstance(manifest.get("index_status"), dict) else {}
        manifest["index_status"] = {**current, "job_id": str(job_id), "updated_at": _now()}
        manifest["updated_at"] = _now()
        manifest["revision"] = int(manifest.get("revision") or 0) + 1
        _write_json(research["path"] / RESEARCH_MANIFEST, manifest)
        return {"research_id": research_id, "path": _relative(research["path"]), "index_status": manifest["index_status"]}


def get_research_node(research_id: str) -> dict[str, Any]:
    research = _find_node(research_id, expected_type="research")
    return {
        "research_id": research["id"],
        "path": _relative(research["path"]),
        "absolute_path": str(research["path"]),
        "manifest": research["manifest"],
    }


def update_research_reader_assets(
    research_id: str,
    documents: list[dict[str, Any]],
    citation_sources: list[dict[str, Any]],
    research_html: str = "",
) -> dict[str, Any]:
    """Backfill or regenerate derived HTML and citation metadata for one ResearchItem."""

    with _tree_lock():
        research = _find_node(research_id, expected_type="research")
        manifest = dict(research["manifest"])
        document_manifest = [dict(item) for item in manifest.get("documents") or [] if isinstance(item, dict)]
        by_filename = {str(item.get("filename") or ""): item for item in document_manifest}
        for raw in documents:
            filename = Path(str(raw.get("filename") or "")).name
            if filename not in by_filename or not filename.lower().endswith(".md"):
                raise ValueError(f"research document is not manifest-listed: {filename}")
            html_content = str(raw.get("html") or "").strip()
            if not html_content:
                continue
            html_filename = f"{Path(filename).stem}.html"
            html_path = research["path"] / html_filename
            _write_text(html_path, html_content.rstrip() + "\n")
            by_filename[filename].update(
                {
                    "html_filename": html_filename,
                    "html_bytes": html_path.stat().st_size,
                    "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
                }
            )
        manifest["documents"] = document_manifest
        research_html_content = str(research_html or "").strip()
        if research_html_content:
            html_path = research["path"] / RESEARCH_HTML_FILENAME
            _write_text(html_path, research_html_content.rstrip() + "\n")
            manifest.update(
                {
                    "html_filename": RESEARCH_HTML_FILENAME,
                    "html_bytes": html_path.stat().st_size,
                    "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
                }
            )
        manifest["citation_sources"] = _normalize_citation_sources(citation_sources)
        manifest["updated_at"] = _now()
        manifest["revision"] = int(manifest.get("revision") or 0) + 1
        _write_json(research["path"] / RESEARCH_MANIFEST, manifest)
        return {
            "research_id": research_id,
            "path": _relative(research["path"]),
            "documents": document_manifest,
            "html_filename": manifest.get("html_filename"),
            "citation_source_count": len(manifest["citation_sources"]),
        }


def recover_incomplete_operations() -> None:
    """Finish or mark interrupted single-rename operations after a process crash."""

    preview_dir = KNOWLEDGE_ROOT / _OPERATIONS_DIR / _PREVIEWS_DIR
    if not preview_dir.exists():
        return
    for path in sorted(preview_dir.glob("*.json")):
        preview = _read_json(path, required=False)
        if not preview or preview.get("state") != "applying":
            continue
        plan = preview.get("plan") or {}
        source = _absolute(plan.get("source")) if plan.get("source") else None
        target = _absolute(plan.get("target")) if plan.get("target") else None
        action = str(preview.get("action") or "")
        if action == "archive_research":
            _recover_archive_preview(path, preview)
            continue
        if action == "update_research_status":
            manifest_path = _absolute((plan.get("manifest_path")))
            current = _read_json(manifest_path, required=False) or {}
            expected_manifest = plan.get("manifest") or {}
            if (
                current.get("research_id") == expected_manifest.get("research_id")
                and current.get("revision") == expected_manifest.get("revision")
                and current.get("status") == expected_manifest.get("status")
            ):
                preview["state"] = "committed"
                preview["result"] = plan.get("result") or {}
                preview["recovered_at"] = _now()
            else:
                preview["state"] = "failed"
                preview["error"] = "research status update was interrupted before commit"
            preview["updated_at"] = _now()
            _write_json(path, preview)
            continue
        if target and target.exists() and (source is None or not source.exists()):
            if action in {"rename_node", "move_research"}:
                _write_json(target / str(plan["manifest_name"]), plan["manifest"])
            elif action in {"trash_node", "trash_legacy"}:
                _write_json(target / TRASH_MANIFEST, plan["trash_manifest"])
            elif action == "restore_node":
                (target / TRASH_MANIFEST).unlink(missing_ok=True)
            preview["state"] = "committed"
            preview["result"] = plan.get("result") or {}
            preview["recovered_at"] = _now()
            preview["updated_at"] = preview["recovered_at"]
            _write_json(path, preview)
        elif source and source.exists() and (target is None or not target.exists()):
            preview["state"] = "failed"
            preview["error"] = "operation interrupted before filesystem commit"
            preview["updated_at"] = _now()
            _write_json(path, preview)


def merge_managed_overview(existing: str, managed_markdown: str, *, title: str = "") -> str:
    """Replace only the Harness-managed block and preserve all human-authored text."""

    current = str(existing or "")
    managed = str(managed_markdown or "").strip()
    if MANAGED_START in managed or MANAGED_END in managed:
        raise ValueError("managed overview content must not contain management markers")
    block = f"{MANAGED_START}\n{managed}\n{MANAGED_END}"
    start_count = current.count(MANAGED_START)
    end_count = current.count(MANAGED_END)
    if start_count != end_count or start_count > 1:
        raise ValueError("overview contains invalid managed block markers")
    if start_count == 1:
        start = current.index(MANAGED_START)
        end = current.index(MANAGED_END, start) + len(MANAGED_END)
        return f"{current[:start]}{block}{current[end:]}".rstrip() + "\n"
    prefix = current.rstrip()
    if not prefix:
        prefix = f"# {str(title or '知识总览').strip()}"
    return f"{prefix}\n\n{block}\n"


def _preview_create_knowledge_base(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    title = _validate_title(payload.get("title"))
    target = KNOWLEDGE_ROOT / _safe_directory_name(title)
    if target.exists():
        raise ValueError(f"knowledge base path already exists: {target.name}")
    node_id = _new_id("kb")
    timestamp = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "knowledge_base",
        "knowledge_base_id": node_id,
        "title": title,
        "description": str(payload.get("description") or "").strip(),
        "status": "active",
        "sort_order": _as_int(payload.get("sort_order"), default=0),
        "tags": _clean_tags(payload.get("tags")),
        "created_at": timestamp,
        "updated_at": timestamp,
        "revision": 1,
    }
    return {
        "target": _relative(target),
        "manifest_name": KB_MANIFEST,
        "manifest": manifest,
        "overview": merge_managed_overview("", "## 当前覆盖\n\n暂无调研。\n\n## 调研目录\n\n暂无调研。\n\n## 知识缺口\n\n等待首次调研后更新。", title=title),
        "result": {"node_type": "knowledge_base", "node_id": node_id, "path": _relative(target)},
        "expected": [_expect_absent(target)],
        "changes": [
            {"op": "create", "node_type": "knowledge_base", "node_id": node_id, "path": _relative(target), "title": title},
        ],
    }


def _preview_create_theme(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    parent = _find_node(str(payload.get("knowledge_base_id") or ""), expected_type="knowledge_base")
    title = _validate_title(payload.get("title"))
    target = parent["path"] / _safe_directory_name(title)
    if target.exists():
        raise ValueError(f"theme path already exists: {target.name}")
    node_id = _new_id("theme")
    timestamp = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "theme",
        "theme_id": node_id,
        "knowledge_base_id": parent["id"],
        "title": title,
        "description": str(payload.get("description") or "").strip(),
        "status": "active",
        "sort_order": _as_int(payload.get("sort_order"), default=0),
        "created_at": timestamp,
        "updated_at": timestamp,
        "revision": 1,
    }
    return {
        "target": _relative(target),
        "manifest_name": THEME_MANIFEST,
        "manifest": manifest,
        "overview": merge_managed_overview("", "## 当前覆盖\n\n暂无调研。\n\n## 调研目录\n\n暂无调研。\n\n## 知识缺口\n\n等待首次调研后更新。", title=title),
        "result": {"node_type": "theme", "node_id": node_id, "path": _relative(target)},
        "expected": [_expect_present(parent["path"] / KB_MANIFEST), _expect_absent(target)],
        "changes": [
            {"op": "create", "node_type": "theme", "node_id": node_id, "path": _relative(target), "title": title},
        ],
    }


def _preview_archive_research(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    """Build an exact replacement tree so confirmation only swaps one KB directory."""

    documents = _prepare_research_documents(payload.get("documents"))
    title = _validate_title(payload.get("title"))
    summary = re.sub(r"\s+", " ", str(payload.get("summary") or "")).strip()[:1000]
    timestamp = str(payload.get("researched_at") or _now()).strip()
    research_id = _new_id("research")

    existing_kb_id = str(payload.get("knowledge_base_id") or "").strip()
    new_kb = payload.get("new_knowledge_base") if isinstance(payload.get("new_knowledge_base"), dict) else None
    if bool(existing_kb_id) == bool(new_kb):
        raise ValueError("choose exactly one existing or new knowledge base")

    existing_kb: dict[str, Any] | None = None
    if existing_kb_id:
        existing_kb = _find_node(existing_kb_id, expected_type="knowledge_base")
        kb_id = existing_kb["id"]
        kb_title = str(existing_kb["manifest"].get("title") or existing_kb["path"].name)
        target = existing_kb["path"]
    else:
        assert new_kb is not None
        kb_title = _validate_title(new_kb.get("title"))
        kb_id = _new_id("kb")
        target = KNOWLEDGE_ROOT / _safe_directory_name(kb_title)
        if target.exists():
            raise ValueError(f"knowledge base path already exists: {target.name}")

    stage_root = KNOWLEDGE_ROOT / _STAGING_DIR / preview_id
    replacement = stage_root / "knowledge_base"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    if existing_kb is not None:
        shutil.copytree(target, replacement)
    else:
        replacement.mkdir()
        now = _now()
        _write_json(
            replacement / KB_MANIFEST,
            {
                "schema_version": SCHEMA_VERSION,
                "type": "knowledge_base",
                "knowledge_base_id": kb_id,
                "title": kb_title,
                "description": str(new_kb.get("description") or "").strip(),
                "status": "active",
                "sort_order": _as_int(new_kb.get("sort_order"), default=0),
                "tags": _clean_tags(new_kb.get("tags")),
                "created_at": now,
                "updated_at": now,
                "revision": 1,
            },
        )

    theme_id = str(payload.get("theme_id") or "").strip()
    new_theme = payload.get("new_theme") if isinstance(payload.get("new_theme"), dict) else None
    if theme_id and new_theme:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise ValueError("choose either an existing or new theme")

    parent = replacement
    theme_title: str | None = None
    created_theme = False
    if theme_id:
        if existing_kb is None:
            shutil.rmtree(stage_root, ignore_errors=True)
            raise ValueError("an existing theme requires an existing knowledge base")
        theme = _find_node(theme_id, expected_type="theme")
        if theme["manifest"].get("knowledge_base_id") != kb_id:
            shutil.rmtree(stage_root, ignore_errors=True)
            raise ValueError("target theme does not belong to target knowledge base")
        parent = replacement / theme["path"].relative_to(target)
        theme_title = str(theme["manifest"].get("title") or theme["path"].name)
    elif new_theme:
        theme_title = _validate_title(new_theme.get("title"))
        theme_id = _new_id("theme")
        parent = replacement / _safe_directory_name(theme_title)
        if parent.exists():
            shutil.rmtree(stage_root, ignore_errors=True)
            raise ValueError(f"theme path already exists: {parent.name}")
        parent.mkdir()
        created_theme = True
        now = _now()
        _write_json(
            parent / THEME_MANIFEST,
            {
                "schema_version": SCHEMA_VERSION,
                "type": "theme",
                "theme_id": theme_id,
                "knowledge_base_id": kb_id,
                "title": theme_title,
                "description": str(new_theme.get("description") or "").strip(),
                "status": "active",
                "sort_order": _as_int(new_theme.get("sort_order"), default=0),
                "created_at": now,
                "updated_at": now,
                "revision": 1,
            },
        )
    else:
        theme_id = ""

    directory_name = _research_directory_name(timestamp, title, research_id)
    research_dir = parent / directory_name
    if research_dir.exists():
        research_dir = parent / f"{directory_name}__{research_id[-8:]}"
    research_dir.mkdir()

    document_manifest: list[dict[str, Any]] = []
    for document in documents:
        path = research_dir / document["filename"]
        _write_text(path, document["content"].rstrip() + "\n")
        metadata = {
            "title": document["title"],
            "filename": document["filename"],
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        html_content = str(document.get("html") or "").strip()
        if html_content:
            html_filename = f"{Path(document['filename']).stem}.html"
            html_path = research_dir / html_filename
            _write_text(html_path, html_content.rstrip() + "\n")
            metadata.update(
                {
                    "html_filename": html_filename,
                    "html_bytes": html_path.stat().st_size,
                    "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
                }
            )
        document_manifest.append(metadata)

    research_html_content = str(payload.get("html") or "").strip()
    research_html_metadata: dict[str, Any] = {}
    if research_html_content:
        html_path = research_dir / RESEARCH_HTML_FILENAME
        _write_text(html_path, research_html_content.rstrip() + "\n")
        research_html_metadata = {
            "html_filename": RESEARCH_HTML_FILENAME,
            "html_bytes": html_path.stat().st_size,
            "html_sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(),
        }

    sources = _normalize_sources(payload.get("sources"))
    citation_sources = _normalize_citation_sources(payload.get("citation_sources"))
    quality_review = _normalize_quality_review(payload.get("quality_review"))
    research_manifest = {
        "schema_version": SCHEMA_VERSION,
        "type": "research",
        "research_id": research_id,
        "knowledge_base_id": kb_id,
        "theme_id": theme_id or None,
        "source_job_id": str(payload.get("source_job_id") or "").strip() or None,
        "title": title,
        "summary": summary,
        "researched_at": timestamp,
        "status": "active",
        "tags": _clean_tags(payload.get("tags")),
        "documents": document_manifest,
        **research_html_metadata,
        "sources": sources,
        "source_count": len(sources),
        "citation_sources": citation_sources,
        "evidence_count": max(0, _as_int(payload.get("evidence_count"), default=0)),
        "comparison_budget": payload.get("comparison_budget") if isinstance(payload.get("comparison_budget"), dict) else {},
        "matrix_coverage": payload.get("matrix_coverage") if isinstance(payload.get("matrix_coverage"), dict) else {},
        "matrix_gaps_acknowledged": bool(payload.get("matrix_gaps_acknowledged")),
        **({"quality_review": quality_review} if quality_review is not None else {}),
        "index_status": {"status": "pending", "job_id": None, "updated_at": _now()},
        "created_at": _now(),
        "updated_at": _now(),
        "revision": 1,
    }
    _write_json(research_dir / RESEARCH_MANIFEST, research_manifest)

    if theme_id:
        _touch_manifest(parent / THEME_MANIFEST)
        _refresh_theme_overview(parent)
    _touch_manifest(replacement / KB_MANIFEST)
    _refresh_knowledge_base_overview(replacement)

    research_relative_to_kb = research_dir.relative_to(replacement)
    final_research_path = target / research_relative_to_kb
    backup = KNOWLEDGE_ROOT / _OPERATIONS_DIR / _BACKUPS_DIR / preview_id
    expected = [_expect_present(replacement)]
    expected.append(_expect_present(target) if existing_kb is not None else _expect_absent(target))
    changes: list[dict[str, Any]] = []
    if existing_kb is None:
        changes.append({"op": "create", "node_type": "knowledge_base", "node_id": kb_id, "path": _relative(target), "title": kb_title})
    if created_theme:
        changes.append({"op": "create", "node_type": "theme", "node_id": theme_id, "path": _relative(target / parent.relative_to(replacement)), "title": theme_title})
    changes.extend(
        [
            {"op": "create", "node_type": "research", "node_id": research_id, "path": _relative(final_research_path), "title": title},
            {"op": "update_overview", "node_type": "knowledge_base", "node_id": kb_id, "path": _relative(target / OVERVIEW_FILENAME)},
        ]
    )
    if theme_id:
        changes.append({"op": "update_overview", "node_type": "theme", "node_id": theme_id, "path": _relative(target / parent.relative_to(replacement) / OVERVIEW_FILENAME)})

    return {
        "target": _relative(target),
        "stage": _relative(replacement),
        "backup": _relative(backup),
        "research_manifest": research_manifest,
        "target_existed": existing_kb is not None,
        "result": {
            "node_type": "research",
            "node_id": research_id,
            "research_id": research_id,
            "knowledge_base_id": kb_id,
            "theme_id": theme_id or None,
            "path": _relative(final_research_path),
            "manifest_path": _relative(final_research_path / RESEARCH_MANIFEST),
            "document_paths": [_relative(final_research_path / item["filename"]) for item in document_manifest],
            "index_status": "pending",
        },
        "archive": {
            "title": title,
            "summary": summary,
            "knowledge_base": {"id": kb_id, "title": kb_title, "created": existing_kb is None},
            "theme": {"id": theme_id, "title": theme_title, "created": created_theme} if theme_id else None,
            "documents": document_manifest,
            "tags": research_manifest["tags"],
            "source_count": len(sources),
            "evidence_count": research_manifest["evidence_count"],
            **({"quality_review": quality_review} if quality_review is not None else {}),
        },
        "expected": expected,
        "changes": changes,
    }


def _preview_rename_node(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    node = _find_node(str(payload.get("node_id") or ""))
    title = _validate_title(payload.get("title"))
    target_name = _directory_name_for_node(node["type"], title, node["manifest"])
    target = node["path"].parent / target_name
    if target == node["path"]:
        raise ValueError("new title resolves to the existing directory name")
    if target.exists():
        raise ValueError(f"target path already exists: {target.name}")
    manifest = {**node["manifest"], "title": title, "updated_at": _now(), "revision": int(node["manifest"].get("revision") or 0) + 1}
    return {
        "source": _relative(node["path"]),
        "target": _relative(target),
        "manifest_name": NODE_MANIFESTS[node["type"]],
        "manifest": manifest,
        "result": {"node_type": node["type"], "node_id": node["id"], "path": _relative(target)},
        "expected": [_expect_present(node["path"]), _expect_absent(target)],
        "changes": [
            {"op": "rename", "node_type": node["type"], "node_id": node["id"], "from": _relative(node["path"]), "to": _relative(target), "title": title},
        ],
    }


def _preview_update_research_status(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    research = _find_node(str(payload.get("research_id") or ""), expected_type="research")
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"active", "possibly_stale", "superseded", "archived"}:
        raise ValueError("research status must be active, possibly_stale, superseded, or archived")
    if status == research["manifest"].get("status"):
        raise ValueError("research already has the requested status")
    superseded_by = str(payload.get("superseded_by_research_id") or "").strip() or None
    if status == "superseded" and superseded_by:
        replacement = _find_node(superseded_by, expected_type="research")
        if replacement["id"] == research["id"]:
            raise ValueError("research cannot supersede itself")
    manifest = {
        **research["manifest"],
        "status": status,
        "superseded_by_research_id": superseded_by if status == "superseded" else None,
        "status_reason": str(payload.get("reason") or "").strip()[:500] or None,
        "updated_at": _now(),
        "revision": int(research["manifest"].get("revision") or 0) + 1,
    }
    return {
        "manifest_path": _relative(research["path"] / RESEARCH_MANIFEST),
        "manifest": manifest,
        "result": {
            "node_type": "research",
            "node_id": research["id"],
            "path": _relative(research["path"]),
            "status": status,
            "superseded_by_research_id": manifest["superseded_by_research_id"],
        },
        "expected": [_expect_present(research["path"])],
        "changes": [
            {
                "op": "update_status",
                "node_type": "research",
                "node_id": research["id"],
                "from": research["manifest"].get("status"),
                "to": status,
            }
        ],
    }


def _preview_move_research(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    research = _find_node(str(payload.get("research_id") or ""), expected_type="research")
    target_kb = _find_node(str(payload.get("knowledge_base_id") or ""), expected_type="knowledge_base")
    theme_id = str(payload.get("theme_id") or "").strip()
    target_parent = target_kb["path"]
    target_theme_id: str | None = None
    expected = [_expect_present(research["path"]), _expect_present(target_kb["path"] / KB_MANIFEST)]
    if theme_id:
        theme = _find_node(theme_id, expected_type="theme")
        if theme["manifest"].get("knowledge_base_id") != target_kb["id"]:
            raise ValueError("target theme does not belong to target knowledge base")
        target_parent = theme["path"]
        target_theme_id = theme["id"]
        expected.append(_expect_present(theme["path"] / THEME_MANIFEST))
    target = target_parent / research["path"].name
    if target == research["path"]:
        raise ValueError("research item is already in the requested location")
    if target.exists():
        raise ValueError(f"target research path already exists: {target.name}")
    manifest = {
        **research["manifest"],
        "knowledge_base_id": target_kb["id"],
        "theme_id": target_theme_id,
        "updated_at": _now(),
        "revision": int(research["manifest"].get("revision") or 0) + 1,
    }
    expected.append(_expect_absent(target))
    return {
        "source": _relative(research["path"]),
        "target": _relative(target),
        "manifest_name": RESEARCH_MANIFEST,
        "manifest": manifest,
        "result": {"node_type": "research", "node_id": research["id"], "path": _relative(target)},
        "expected": expected,
        "changes": [
            {"op": "move", "node_type": "research", "node_id": research["id"], "from": _relative(research["path"]), "to": _relative(target)},
        ],
    }


def _preview_trash_node(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    node = _find_node(str(payload.get("node_id") or ""))
    target = KNOWLEDGE_ROOT / _TRASH_DIR / node["type"] / f"{node['id']}__{node['path'].name}"
    if target.exists():
        raise ValueError(f"trash target already exists for node: {node['id']}")
    timestamp = _now()
    trash_manifest = {
        "schema_version": SCHEMA_VERSION,
        "node_id": node["id"],
        "node_type": node["type"],
        "original_path": _relative(node["path"]),
        "trashed_at": timestamp,
        "preview_id": preview_id,
    }
    return {
        "source": _relative(node["path"]),
        "target": _relative(target),
        "trash_manifest": trash_manifest,
        "result": {"node_type": node["type"], "node_id": node["id"], "trash_path": _relative(target)},
        "expected": [_expect_present(node["path"]), _expect_absent(target)],
        "changes": [
            {"op": "trash", "node_type": node["type"], "node_id": node["id"], "from": _relative(node["path"]), "to": _relative(target)},
        ],
    }


def _preview_trash_legacy(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    legacy_topic = str(payload.get("legacy_topic") or "").strip()
    if not legacy_topic or "/" in legacy_topic or "\\" in legacy_topic or legacy_topic.startswith("."):
        raise ValueError("legacy_topic must be one root directory name")
    source = KNOWLEDGE_ROOT / legacy_topic
    if not source.exists() or not source.is_dir() or _is_internal_path(source):
        raise ValueError(f"legacy topic not found: {legacy_topic}")
    if (source / KB_MANIFEST).exists():
        raise ValueError("manifest-backed knowledge bases must use trash_node")
    node_id = f"legacy_{hashlib.sha256(legacy_topic.encode('utf-8')).hexdigest()[:32]}"
    target = KNOWLEDGE_ROOT / _TRASH_DIR / "legacy" / f"{node_id}__{source.name}"
    if target.exists():
        raise ValueError(f"trash target already exists for legacy topic: {legacy_topic}")
    trash_manifest = {
        "schema_version": SCHEMA_VERSION,
        "node_id": node_id,
        "node_type": "legacy",
        "title": legacy_topic,
        "original_path": _relative(source),
        "trashed_at": _now(),
        "preview_id": preview_id,
    }
    return {
        "source": _relative(source),
        "target": _relative(target),
        "trash_manifest": trash_manifest,
        "result": {"node_type": "legacy", "node_id": node_id, "trash_path": _relative(target)},
        "expected": [_expect_present(source), _expect_absent(target)],
        "changes": [
            {"op": "trash", "node_type": "legacy", "node_id": node_id, "from": _relative(source), "to": _relative(target)},
        ],
    }


def _preview_restore_node(payload: dict[str, Any], preview_id: str) -> dict[str, Any]:
    node_id = str(payload.get("node_id") or "").strip()
    trashed = _find_trashed_node(node_id)
    original = _absolute(trashed["trash_manifest"].get("original_path"))
    if original.exists():
        raise ValueError(f"restore target already exists: {_relative(original)}")
    return {
        "source": _relative(trashed["path"]),
        "target": _relative(original),
        "result": {"node_type": trashed["type"], "node_id": trashed["id"], "path": _relative(original)},
        "expected": [_expect_present(trashed["path"]), _expect_absent(original)],
        "changes": [
            {"op": "restore", "node_type": trashed["type"], "node_id": trashed["id"], "from": _relative(trashed["path"]), "to": _relative(original)},
        ],
    }


def _execute_plan(preview: dict[str, Any]) -> dict[str, Any]:
    action = str(preview.get("action") or "")
    plan = preview.get("plan") or {}
    if action == "archive_research":
        return _execute_archive_plan(preview)
    if action == "update_research_status":
        _write_json(_absolute(plan["manifest_path"]), plan["manifest"])
        return plan["result"]
    if action in {"create_knowledge_base", "create_theme"}:
        target = _absolute(plan["target"])
        stage = KNOWLEDGE_ROOT / _STAGING_DIR / str(preview["preview_id"])
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        _write_json(stage / str(plan["manifest_name"]), plan["manifest"])
        _write_text(stage / OVERVIEW_FILENAME, str(plan["overview"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, target)
        return plan["result"]

    source = _absolute(plan["source"])
    target = _absolute(plan["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    try:
        if action in {"rename_node", "move_research"}:
            _write_json(target / str(plan["manifest_name"]), plan["manifest"])
        elif action in {"trash_node", "trash_legacy"}:
            _write_json(target / TRASH_MANIFEST, plan["trash_manifest"])
        elif action == "restore_node":
            (target / TRASH_MANIFEST).unlink(missing_ok=True)
    except Exception:
        if target.exists() and not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, source)
        raise
    return plan["result"]


def _execute_archive_plan(preview: dict[str, Any]) -> dict[str, Any]:
    plan = preview.get("plan") or {}
    target = _absolute(plan["target"])
    stage = _absolute(plan["stage"])
    backup = _absolute(plan["backup"])
    target_existed = bool(plan.get("target_existed"))
    target.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)

    if target_existed:
        os.replace(target, backup)
        try:
            os.replace(stage, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(stage, target)
    shutil.rmtree(stage.parent, ignore_errors=True)
    return plan["result"]


def _recover_archive_preview(path: Path, preview: dict[str, Any]) -> None:
    plan = preview.get("plan") or {}
    target = _absolute(plan.get("target"))
    stage = _absolute(plan.get("stage"))
    backup = _absolute(plan.get("backup"))
    target_existed = bool(plan.get("target_existed"))

    if target.exists() and _archive_result_exists(target, plan.get("result") or {}):
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(stage.parent, ignore_errors=True)
        preview["state"] = "committed"
        preview["result"] = plan.get("result") or {}
        preview["recovered_at"] = _now()
    elif target_existed and not target.exists() and backup.exists():
        if stage.exists():
            os.replace(stage, target)
            shutil.rmtree(backup, ignore_errors=True)
            preview["state"] = "committed"
            preview["result"] = plan.get("result") or {}
            preview["recovered_at"] = _now()
        else:
            os.replace(backup, target)
            preview["state"] = "failed"
            preview["error"] = "archive transaction restored its backup after interruption"
    elif not target_existed and not target.exists() and stage.exists():
        os.replace(stage, target)
        preview["state"] = "committed"
        preview["result"] = plan.get("result") or {}
        preview["recovered_at"] = _now()
    else:
        preview["state"] = "failed"
        preview["error"] = "archive transaction could not be recovered automatically"
    preview["updated_at"] = _now()
    _write_json(path, preview)


def _archive_result_exists(target: Path, result: dict[str, Any]) -> bool:
    result_path = _absolute(result.get("path"))
    try:
        relative = result_path.relative_to(target)
    except ValueError:
        return False
    manifest = target / relative / RESEARCH_MANIFEST
    value = _read_json(manifest, required=False) or {}
    return value.get("research_id") == result.get("research_id")


def _knowledge_base_entry(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    themes: list[dict[str, Any]] = []
    research_items: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        theme_manifest = _read_manifest(child / THEME_MANIFEST, expected_type="theme", required=False)
        if theme_manifest is not None:
            themes.append(_theme_entry(child, theme_manifest))
            continue
        research_manifest = _read_manifest(child / RESEARCH_MANIFEST, expected_type="research", required=False)
        if research_manifest is not None:
            research_items.append(_research_entry(child, research_manifest))
    themes.sort(key=lambda item: (int(item.get("sort_order") or 0), str(item.get("title") or "").casefold()))
    research_items.sort(key=lambda item: str(item.get("researched_at") or ""), reverse=True)
    return {
        **manifest,
        "id": manifest["knowledge_base_id"],
        "path": _relative(path),
        "overview_path": _relative(path / OVERVIEW_FILENAME) if (path / OVERVIEW_FILENAME).exists() else None,
        "themes": themes,
        "research_items": research_items,
        "research_count": len(research_items) + sum(len(theme["research_items"]) for theme in themes),
    }


def _theme_entry(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    research_items: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        research_manifest = _read_manifest(child / RESEARCH_MANIFEST, expected_type="research", required=False)
        if research_manifest is not None:
            research_items.append(_research_entry(child, research_manifest))
    research_items.sort(key=lambda item: str(item.get("researched_at") or ""), reverse=True)
    return {
        **manifest,
        "id": manifest["theme_id"],
        "path": _relative(path),
        "overview_path": _relative(path / OVERVIEW_FILENAME) if (path / OVERVIEW_FILENAME).exists() else None,
        "research_items": research_items,
        "research_count": len(research_items),
    }


def _research_entry(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    documents = manifest.get("documents") if isinstance(manifest.get("documents"), list) else []
    if not documents:
        documents = [
            {"filename": item.name, "bytes": item.stat().st_size}
            for item in sorted(path.glob("*.md"), key=lambda item: item.name.casefold())
        ]
    return {
        **manifest,
        "id": manifest["research_id"],
        "path": _relative(path),
        "documents": documents,
        "document_count": len(documents),
    }


def _prepare_research_documents(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("research archive requires at least one document")
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("each research document must be an object")
        title = _validate_title(raw.get("title"))
        content = str(raw.get("content") or "").strip()
        if not content:
            raise ValueError(f"research document content is empty: {title}")
        base = title[:-3] if title.lower().endswith(".md") else title
        filename = f"{_safe_directory_name(base)}.md"
        key = filename.casefold()
        if key in seen:
            raise ValueError(f"duplicate research document filename: {filename}")
        seen.add(key)
        html_content = str(raw.get("html") or "").strip()
        documents.append({"title": title, "filename": filename, "content": content, "html": html_content})
    return documents


def _normalize_sources(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, dict):
            url = str(raw.get("url") or raw.get("source_url") or "").strip()
            title = str(raw.get("title") or raw.get("source_title") or url).strip()
        else:
            url = str(raw or "").strip()
            title = url
        key = url.casefold()
        if not url or key in seen:
            continue
        seen.add(key)
        result.append({"title": title[:300], "url": url[:2000]})
    return result


def _normalize_quality_review(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_warnings = value.get("warnings") or []
    if not isinstance(raw_warnings, list):
        raw_warnings = [raw_warnings]
    human_acknowledged = value.get("human_acknowledged") is True
    return {
        "automatic_status": str(value.get("automatic_status") or "unknown").strip() or "unknown",
        "automatic_approval_ready": value.get("automatic_approval_ready") is True,
        "reason": str(value.get("reason") or "").strip(),
        "warnings": [str(item).strip() for item in raw_warnings if str(item).strip()],
        "human_acknowledged": human_acknowledged,
        "acknowledged_at": (
            str(value.get("acknowledged_at") or "").strip() or None
            if human_acknowledged
            else None
        ),
    }


def _normalize_citation_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        number = _as_int(raw.get("number"), default=0)
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        evidence: list[dict[str, Any]] = []
        for item in raw.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "evidence_id": str(item.get("evidence_id") or "").strip(),
                    "quote": str(item.get("quote") or "").strip(),
                    "claim": str(item.get("claim") or "").strip(),
                    "used_by_documents": [
                        str(name).strip()
                        for name in item.get("used_by_documents") or []
                        if str(name).strip()
                    ],
                }
            )
        result.append(
            {
                "number": number,
                "title": str(raw.get("title") or raw.get("url") or f"来源 {number}").strip()[:300],
                "url": str(raw.get("url") or "").strip()[:2000],
                "evidence": evidence,
            }
        )
    return sorted(result, key=lambda item: item["number"])


def _document_citations(content: str, document_meta: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    numbers = {int(value) for value in _NUMBERED_CITATION_RE.findall(content)}
    if not numbers:
        return []
    document_title = str(document_meta.get("title") or Path(str(document_meta.get("filename") or "")).stem).strip()
    citation_sources = _normalize_citation_sources(manifest.get("citation_sources"))
    if not citation_sources:
        citation_sources = [
            {"number": index, "title": item.get("title"), "url": item.get("url"), "evidence": []}
            for index, item in enumerate(_normalize_sources(manifest.get("sources")), start=1)
        ]
    citations: list[dict[str, Any]] = []
    for source in citation_sources:
        if source["number"] not in numbers:
            continue
        relevant = []
        for evidence in source.get("evidence") or []:
            used_by = evidence.get("used_by_documents") or []
            if used_by and document_title not in used_by:
                continue
            relevant.append(
                {
                    "evidence_id": evidence.get("evidence_id"),
                    "quote": evidence.get("quote"),
                    "claim": evidence.get("claim"),
                }
            )
        citations.append(
            {
                "number": source["number"],
                "title": source.get("title"),
                "url": source.get("url"),
                "evidence": relevant,
            }
        )
    return citations


def _research_directory_name(researched_at: str, title: str, research_id: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", researched_at)
    date_prefix = match.group(1) if match else datetime.now().astimezone().strftime("%Y-%m-%d")
    base = f"{date_prefix}_{_safe_directory_name(title)}"
    return base if len(base) <= 120 else f"{base[:110].rstrip()}__{research_id[-8:]}"


def _touch_manifest(path: Path) -> None:
    manifest = _read_json(path)
    assert manifest is not None
    manifest["updated_at"] = _now()
    manifest["revision"] = int(manifest.get("revision") or 0) + 1
    _write_json(path, manifest)


def _refresh_theme_overview(theme_path: Path) -> None:
    manifest = _read_manifest(theme_path / THEME_MANIFEST, expected_type="theme", required=True)
    assert manifest is not None
    rows = _research_rows(theme_path)
    managed = _render_managed_overview(rows, scope="theme")
    overview = theme_path / OVERVIEW_FILENAME
    existing = overview.read_text(encoding="utf-8") if overview.exists() else ""
    _write_text(overview, merge_managed_overview(existing, managed, title=str(manifest["title"])))


def _refresh_knowledge_base_overview(kb_path: Path) -> None:
    manifest = _read_manifest(kb_path / KB_MANIFEST, expected_type="knowledge_base", required=True)
    assert manifest is not None
    direct = _research_rows(kb_path)
    themes: list[tuple[str, list[dict[str, Any]]]] = []
    for child in sorted(kb_path.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir():
            continue
        theme = _read_manifest(child / THEME_MANIFEST, expected_type="theme", required=False)
        if theme is not None:
            themes.append((str(theme["title"]), _research_rows(child)))
    all_rows = [*direct, *(row for _title, rows in themes for row in rows)]
    managed = _render_managed_overview(all_rows, scope="knowledge_base", direct=direct, themes=themes)
    overview = kb_path / OVERVIEW_FILENAME
    existing = overview.read_text(encoding="utf-8") if overview.exists() else ""
    _write_text(overview, merge_managed_overview(existing, managed, title=str(manifest["title"])))


def _research_rows(parent: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in parent.iterdir() if parent.exists() else []:
        if not child.is_dir() or child.name.startswith("."):
            continue
        manifest = _read_manifest(child / RESEARCH_MANIFEST, expected_type="research", required=False)
        if manifest is not None:
            rows.append({**manifest, "directory": child.name})
    rows.sort(key=lambda item: str(item.get("researched_at") or ""), reverse=True)
    return rows


def _render_managed_overview(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    direct: list[dict[str, Any]] | None = None,
    themes: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> str:
    lines = ["## 当前覆盖", ""]
    if rows:
        for item in rows[:8]:
            summary = str(item.get("summary") or "暂无摘要").strip()
            lines.append(f"- **{item['title']}**：{summary}")
    else:
        lines.append("暂无调研。")

    lines.extend(["", "## 调研目录", ""])
    if scope == "theme":
        for item in rows:
            lines.append(f"- [{item['title']}]({item['directory']}/00_总览.md)（{str(item.get('researched_at') or '')[:10]}）")
    else:
        direct_rows = direct or []
        theme_rows = themes or []
        for item in direct_rows:
            lines.append(f"- [{item['title']}]({item['directory']}/00_总览.md)（直属调研，{str(item.get('researched_at') or '')[:10]}）")
        for theme_title, items in theme_rows:
            lines.append(f"- **{theme_title}**：{len(items)} 次调研")
            for item in items:
                lines.append(f"  - [{item['title']}]({theme_title}/{item['directory']}/00_总览.md)（{str(item.get('researched_at') or '')[:10]}）")
    if not rows:
        lines.append("暂无调研。")

    lines.extend(["", "## 知识缺口", ""])
    if len(rows) < 2:
        lines.append("当前积累仍较少，建议围绕关键概念、实践路径和最新变化继续补充调研。")
    else:
        lines.append("根据现有调研继续补充时效性验证、相反证据和实践案例。")
    return "\n".join(lines)


def _find_node(node_id: str, *, expected_type: str | None = None) -> dict[str, Any]:
    wanted = str(node_id or "").strip()
    if not wanted:
        raise ValueError("node_id is required")
    for kb_dir in KNOWLEDGE_ROOT.iterdir() if KNOWLEDGE_ROOT.exists() else []:
        if not kb_dir.is_dir() or _is_internal_path(kb_dir):
            continue
        kb = _node_from_path(kb_dir, expected_type="knowledge_base", include_children=False)
        if kb and kb["id"] == wanted:
            return _assert_node_type(kb, expected_type)
        for child in kb_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            node = _node_from_path(child, include_children=False)
            if node and node["id"] == wanted:
                return _assert_node_type(node, expected_type)
            if node and node["type"] == "theme":
                for grandchild in child.iterdir():
                    if not grandchild.is_dir() or grandchild.name.startswith("."):
                        continue
                    research = _node_from_path(grandchild, expected_type="research", include_children=False)
                    if research and research["id"] == wanted:
                        return _assert_node_type(research, expected_type)
    raise ValueError(f"knowledge node not found: {wanted}")


def _find_trashed_node(node_id: str) -> dict[str, Any]:
    wanted = str(node_id or "").strip()
    if not wanted:
        raise ValueError("node_id is required")
    trash_root = KNOWLEDGE_ROOT / _TRASH_DIR
    for type_dir in trash_root.iterdir() if trash_root.exists() else []:
        if not type_dir.is_dir():
            continue
        for item_dir in type_dir.iterdir():
            if not item_dir.is_dir():
                continue
            trash_manifest = _read_json(item_dir / TRASH_MANIFEST, required=False) or {}
            if type_dir.name == "legacy":
                node = {
                    "type": "legacy",
                    "id": trash_manifest.get("node_id"),
                    "title": trash_manifest.get("title") or item_dir.name,
                    "path": item_dir,
                }
            else:
                node = _node_from_path(item_dir, expected_type=type_dir.name, include_children=False)
            if node and node["id"] == wanted:
                node["trash_manifest"] = trash_manifest
                return node
    raise ValueError(f"trashed knowledge node not found: {wanted}")


def _node_from_path(path: Path, *, expected_type: str | None = None, include_children: bool = False) -> dict[str, Any] | None:
    candidates = [expected_type] if expected_type else ["knowledge_base", "theme", "research"]
    for node_type in candidates:
        if node_type not in NODE_MANIFESTS:
            continue
        manifest = _read_manifest(path / NODE_MANIFESTS[node_type], expected_type=node_type, required=False)
        if manifest is None:
            continue
        id_key = {"knowledge_base": "knowledge_base_id", "theme": "theme_id", "research": "research_id"}[node_type]
        return {"type": node_type, "id": manifest[id_key], "path": path, "manifest": manifest, "title": manifest.get("title")}
    return None


def _assert_node_type(node: dict[str, Any], expected_type: str | None) -> dict[str, Any]:
    if expected_type and node["type"] != expected_type:
        raise ValueError(f"expected {expected_type}, got {node['type']}")
    return node


def _legacy_entry(path: Path) -> dict[str, Any]:
    markdown = sorted(path.glob("*.md"))
    return {
        "legacy_topic": path.name,
        "path": _relative(path),
        "document_count": len(markdown),
        "updated_at": _timestamp(path.stat().st_mtime),
    }


def _read_manifest(path: Path, *, expected_type: str, required: bool) -> dict[str, Any] | None:
    value = _read_json(path, required=required)
    if value is None:
        return None
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema at {_relative(path)}")
    if value.get("type") != expected_type:
        raise ValueError(f"manifest type mismatch at {_relative(path)}")
    id_key = {"knowledge_base": "knowledge_base_id", "theme": "theme_id", "research": "research_id"}[expected_type]
    if not str(value.get(id_key) or "").strip():
        raise ValueError(f"manifest missing {id_key}: {_relative(path)}")
    if not str(value.get("title") or "").strip():
        raise ValueError(f"manifest missing title: {_relative(path)}")
    return value


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise ValueError(f"file not found: {_relative(path)}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {_relative(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required at {_relative(path)}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value)


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_preview(preview: dict[str, Any]) -> None:
    _write_json(_preview_path(str(preview["preview_id"])), preview)


def _read_preview(preview_id: str) -> dict[str, Any]:
    safe = str(preview_id or "").strip()
    if not re.fullmatch(r"kp_[a-f0-9]{32}", safe):
        raise ValueError("invalid preview_id")
    value = _read_json(_preview_path(safe))
    assert value is not None
    return value


def _preview_path(preview_id: str) -> Path:
    return KNOWLEDGE_ROOT / _OPERATIONS_DIR / _PREVIEWS_DIR / f"{preview_id}.json"


def _public_preview(preview: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in preview.items() if key not in {"plan", "expected"}}


def _validate_expected(expected: list[dict[str, Any]]) -> None:
    for item in expected:
        path = _absolute(item.get("path"))
        should_exist = bool(item.get("exists"))
        if path.exists() != should_exist:
            raise ValueError(f"preview is stale because path existence changed: {_relative(path)}")
        if should_exist and item.get("sha256") != _digest_path(path):
            raise ValueError(f"preview is stale because content changed: {_relative(path)}")


def _expect_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"expected path does not exist: {_relative(path)}")
    return {"path": _relative(path), "exists": True, "sha256": _digest_path(path)}


def _expect_absent(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "exists": False}


def _digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        return ""
    for item in sorted((candidate for candidate in path.rglob("*") if candidate.is_file()), key=lambda candidate: str(candidate.relative_to(path))):
        if _OPERATIONS_DIR in item.parts or _STAGING_DIR in item.parts:
            continue
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_layout() -> None:
    KNOWLEDGE_ROOT.mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_ROOT / _OPERATIONS_DIR / _PREVIEWS_DIR).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_ROOT / _OPERATIONS_DIR / _BACKUPS_DIR).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_ROOT / _STAGING_DIR).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_ROOT / _TRASH_DIR).mkdir(parents=True, exist_ok=True)


@contextmanager
def _tree_lock() -> Iterator[None]:
    _ensure_layout()
    lock_path = KNOWLEDGE_ROOT / _OPERATIONS_DIR / "tree.lock"
    with _PROCESS_LOCK:
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    if not title:
        raise ValueError("title is required")
    if title in {".", ".."}:
        raise ValueError("invalid title")
    if len(title) > 120:
        raise ValueError("title must be 120 characters or fewer")
    return title


def _safe_directory_name(title: str) -> str:
    value = _INVALID_PATH_CHARS.sub("_", title).strip(" .")
    value = re.sub(r"\s+", " ", value)
    if not value or value in {".", ".."} or value.startswith("."):
        raise ValueError("title cannot be converted to a safe directory name")
    if value in {_TRASH_DIR, _OPERATIONS_DIR, _STAGING_DIR}:
        raise ValueError("title uses a reserved directory name")
    return value[:120].rstrip(" .")


def _directory_name_for_node(node_type: str, title: str, manifest: dict[str, Any]) -> str:
    safe = _safe_directory_name(title)
    if node_type != "research":
        return safe
    researched_at = str(manifest.get("researched_at") or manifest.get("created_at") or "")
    match = re.match(r"(\d{4}-\d{2}-\d{2})", researched_at)
    date_prefix = match.group(1) if match else datetime.now().astimezone().strftime("%Y-%m-%d")
    return f"{date_prefix}_{safe}"


def _clean_tags(value: Any) -> list[str]:
    values = value if isinstance(value, list) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        tag = re.sub(r"\s+", " ", str(item or "")).strip()
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            result.append(tag[:60])
    return result[:30]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def _relative(path: Path) -> str:
    resolved = path.resolve()
    root = KNOWLEDGE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path escapes knowledge root")
    return str(resolved.relative_to(root)).replace("\\", "/") if resolved != root else "."


def _absolute(value: Any) -> Path:
    relative = str(value or "").replace("\\", "/").strip()
    if not relative or relative == ".":
        return KNOWLEDGE_ROOT.resolve()
    candidate = (KNOWLEDGE_ROOT / relative).resolve()
    root = KNOWLEDGE_ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes knowledge root")
    return candidate


def _is_internal_path(path: Path) -> bool:
    return path.name.startswith(".") or path.name in {_TRASH_DIR, _OPERATIONS_DIR, _STAGING_DIR}
