"""Contract tests for the manifest-backed personal knowledge hierarchy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import knowledge_tree_service as service


@pytest.fixture()
def knowledge_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(service, "KNOWLEDGE_ROOT", root)
    return root


def _confirm(action: str, payload: dict) -> dict:
    preview = service.create_preview(action, payload)
    assert preview["state"] == "pending"
    return service.confirm_preview(preview["preview_id"])


def _create_kb(title: str = "Agent 学习") -> dict:
    result = _confirm("create_knowledge_base", {"title": title, "description": "个人 Agent 学习空间"})
    return result["result"]


def _create_theme(knowledge_base_id: str, title: str = "Harness") -> dict:
    result = _confirm("create_theme", {"knowledge_base_id": knowledge_base_id, "title": title})
    return result["result"]


def _seed_research(root: Path, *, kb: dict, theme: dict | None = None, title: str = "Harness vs Loop 调研") -> dict:
    research_id = "research_" + "1" * 32
    parent = root / (theme["path"] if theme else kb["path"])
    path = parent / f"2026-08-27_{title}"
    path.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "type": "research",
        "research_id": research_id,
        "knowledge_base_id": kb["node_id"],
        "theme_id": theme["node_id"] if theme else None,
        "title": title,
        "summary": "一次测试调研",
        "researched_at": "2026-08-27T10:00:00+08:00",
        "status": "active",
        "tags": ["Harness"],
        "documents": [{"filename": "00_总览.md"}],
        "created_at": "2026-08-27T10:00:00+08:00",
        "updated_at": "2026-08-27T10:00:00+08:00",
        "revision": 1,
    }
    (path / service.RESEARCH_MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (path / "00_总览.md").write_text("# Harness vs Loop\n", encoding="utf-8")
    return {"node_id": research_id, "path": str(path.relative_to(root)).replace("\\", "/")}


def test_preview_does_not_mutate_then_confirm_creates_manifest_tree(knowledge_root: Path):
    preview = service.create_preview("create_knowledge_base", {"title": "Agent 学习"})
    assert preview["state"] == "pending"
    assert not (knowledge_root / "Agent 学习").exists()

    committed = service.confirm_preview(preview["preview_id"])
    assert committed["state"] == "committed"
    assert (knowledge_root / "Agent 学习" / service.KB_MANIFEST).exists()
    assert (knowledge_root / "Agent 学习" / service.OVERVIEW_FILENAME).exists()

    tree = service.list_tree()
    assert tree["knowledge_bases"][0]["title"] == "Agent 学习"
    assert tree["knowledge_bases"][0]["themes"] == []


def test_create_theme_and_list_research_items_at_both_supported_levels(knowledge_root: Path):
    kb = _create_kb()
    theme = _create_theme(kb["node_id"])
    _seed_research(knowledge_root, kb=kb, theme=theme)

    direct = _seed_research(knowledge_root, kb=kb, title="Agent 总体趋势调研")
    direct_path = knowledge_root / direct["path"]
    direct_manifest = json.loads((direct_path / service.RESEARCH_MANIFEST).read_text(encoding="utf-8"))
    direct_manifest["research_id"] = "research_" + "2" * 32
    (direct_path / service.RESEARCH_MANIFEST).write_text(json.dumps(direct_manifest, ensure_ascii=False), encoding="utf-8")

    tree = service.list_tree()
    item = tree["knowledge_bases"][0]
    assert item["research_count"] == 2
    assert [entry["title"] for entry in item["research_items"]] == ["Agent 总体趋势调研"]
    assert [entry["title"] for entry in item["themes"]] == ["Harness"]
    assert item["themes"][0]["research_items"][0]["title"] == "Harness vs Loop 调研"


def test_theme_cannot_be_created_under_another_theme(knowledge_root: Path):
    kb = _create_kb()
    theme = _create_theme(kb["node_id"])
    with pytest.raises(ValueError, match="knowledge node not found|expected knowledge_base"):
        service.create_preview("create_theme", {"knowledge_base_id": theme["node_id"], "title": "Nested"})


def test_managed_overview_replacement_preserves_human_content():
    existing = (
        "# Agent 学习\n\n## 我的学习目标\n\n保留这段人工笔记。\n\n"
        f"{service.MANAGED_START}\n旧内容\n{service.MANAGED_END}\n"
    )
    merged = service.merge_managed_overview(existing, "## 当前覆盖\n\n新内容")
    assert "保留这段人工笔记" in merged
    assert "旧内容" not in merged
    assert "新内容" in merged
    assert merged.count(service.MANAGED_START) == 1


def test_rename_preview_becomes_stale_when_source_changes(knowledge_root: Path):
    kb = _create_kb()
    preview = service.create_preview("rename_node", {"node_id": kb["node_id"], "title": "Agent 工程"})
    overview = knowledge_root / kb["path"] / service.OVERVIEW_FILENAME
    overview.write_text(overview.read_text(encoding="utf-8") + "人工新增\n", encoding="utf-8")

    with pytest.raises(ValueError, match="preview is stale"):
        service.confirm_preview(preview["preview_id"])
    assert service.get_preview(preview["preview_id"])["state"] == "stale"
    assert (knowledge_root / "Agent 学习").exists()


def test_move_research_updates_parent_ids_and_keeps_stable_identity(knowledge_root: Path):
    kb = _create_kb()
    theme = _create_theme(kb["node_id"])
    research = _seed_research(knowledge_root, kb=kb)

    committed = _confirm(
        "move_research",
        {"research_id": research["node_id"], "knowledge_base_id": kb["node_id"], "theme_id": theme["node_id"]},
    )
    moved = knowledge_root / committed["result"]["path"]
    manifest = json.loads((moved / service.RESEARCH_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["research_id"] == research["node_id"]
    assert manifest["theme_id"] == theme["node_id"]
    assert not (knowledge_root / research["path"]).exists()


def test_trash_and_restore_round_trip(knowledge_root: Path):
    kb = _create_kb()
    theme = _create_theme(kb["node_id"])

    trashed = _confirm("trash_node", {"node_id": theme["node_id"]})
    assert trashed["result"]["trash_path"].startswith(".trash/theme/")
    assert service.list_tree()["knowledge_bases"][0]["themes"] == []
    assert service.list_trash()["items"][0]["id"] == theme["node_id"]

    restored = _confirm("restore_node", {"node_id": theme["node_id"]})
    assert restored["result"]["path"] == theme["path"]
    assert service.list_tree()["knowledge_bases"][0]["themes"][0]["id"] == theme["node_id"]
    assert service.list_trash()["count"] == 0


def test_legacy_directories_are_reported_without_becoming_new_nodes(knowledge_root: Path):
    legacy = knowledge_root / "old-topic"
    legacy.mkdir(parents=True)
    (legacy / "00_总览.md").write_text("# old\n", encoding="utf-8")

    tree = service.list_tree()
    assert tree["knowledge_bases"] == []
    assert tree["legacy"] == [
        {
            "legacy_topic": "old-topic",
            "path": "old-topic",
            "document_count": 1,
            "updated_at": tree["legacy"][0]["updated_at"],
        }
    ]


def test_legacy_topic_uses_previewed_trash_and_restore(knowledge_root: Path):
    legacy = knowledge_root / "old-topic"
    legacy.mkdir(parents=True)
    (legacy / "00_总览.md").write_text("# old\n", encoding="utf-8")

    trashed = _confirm("trash_legacy", {"legacy_topic": "old-topic"})
    legacy_id = trashed["result"]["node_id"]
    assert not legacy.exists()
    assert service.list_tree()["legacy"] == []
    assert service.list_trash()["items"][0]["id"] == legacy_id

    restored = _confirm("restore_node", {"node_id": legacy_id})
    assert restored["result"]["path"] == "old-topic"
    assert (legacy / "00_总览.md").exists()


def test_archive_research_can_create_full_three_level_tree_atomically(knowledge_root: Path):
    preview = service.create_preview(
        "archive_research",
        {
            "source_job_id": "job_001",
            "new_knowledge_base": {"title": "Agent 学习", "description": "持续学习 Agent"},
            "new_theme": {"title": "Harness"},
            "title": "Harness vs Loop 调研",
            "summary": "比较 Harness 与 Loop 的边界和工程取舍。",
            "researched_at": "2026-08-27T12:00:00+08:00",
            "tags": ["Harness", "Agent"],
            "documents": [
                {
                    "title": "00_总览",
                    "content": "# Harness vs Loop\n\n结论[1]。",
                    "html": "<!DOCTYPE html><html><body>结论<sup>1</sup></body></html>",
                },
                {"title": "01_关键机制", "content": "# 关键机制\n\n机制。"},
            ],
            "html": "<!DOCTYPE html><html><body>完整报告<sup>1</sup></body></html>",
            "sources": [{"title": "Source A", "url": "https://example.com/a"}],
            "citation_sources": [
                {
                    "number": 1,
                    "title": "Source A",
                    "url": "https://example.com/a",
                    "evidence": [
                        {
                            "evidence_id": "ev_001",
                            "quote": "来源原文。",
                            "claim": "结论。",
                            "used_by_documents": ["00_总览"],
                        }
                    ],
                }
            ],
            "evidence_count": 4,
        },
    )
    assert preview["state"] == "pending"
    assert not (knowledge_root / "Agent 学习").exists()
    assert preview["archive"]["knowledge_base"]["created"] is True
    assert preview["archive"]["theme"]["created"] is True

    committed = service.confirm_preview(preview["preview_id"])
    result = committed["result"]
    research_path = knowledge_root / result["path"]
    manifest = json.loads((research_path / service.RESEARCH_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["research_id"] == result["research_id"]
    assert manifest["source_job_id"] == "job_001"
    assert manifest["index_status"]["status"] == "pending"
    assert [item["filename"] for item in manifest["documents"]] == ["00_总览.md", "01_关键机制.md"]
    assert (research_path / "00_总览.md").read_text(encoding="utf-8").startswith("# Harness vs Loop")
    assert manifest["documents"][0]["html_filename"] == "00_总览.html"
    assert (research_path / "00_总览.html").read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert manifest["html_filename"] == service.RESEARCH_HTML_FILENAME
    assert (research_path / service.RESEARCH_HTML_FILENAME).read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    document = service.read_research_document(result["research_id"], "00_总览.md")
    assert document["has_html"] is True
    assert document["citations"][0]["number"] == 1
    assert document["citations"][0]["evidence"][0]["quote"] == "来源原文。"
    html_document = service.read_research_document_html(result["research_id"], "00_总览.md")
    assert "<sup>1</sup>" in html_document["content"]
    research_html = service.read_research_html(result["research_id"])
    assert "完整报告<sup>1</sup>" in research_html["content"]

    tree = service.list_tree()
    kb = tree["knowledge_bases"][0]
    assert kb["title"] == "Agent 学习"
    assert kb["themes"][0]["title"] == "Harness"
    assert kb["themes"][0]["research_items"][0]["id"] == result["research_id"]
    assert "Harness vs Loop 调研" in (knowledge_root / "Agent 学习" / "00_总览.md").read_text(encoding="utf-8")
    assert "Harness vs Loop 调研" in (knowledge_root / "Agent 学习" / "Harness" / "00_总览.md").read_text(encoding="utf-8")


def test_archive_research_preserves_manual_overview_and_appends_distinct_items(knowledge_root: Path):
    kb = _create_kb()
    overview = knowledge_root / kb["path"] / service.OVERVIEW_FILENAME
    overview.write_text("# Agent 学习\n\n## 我的目标\n\n保留人工目标。\n", encoding="utf-8")

    payload = {
        "knowledge_base_id": kb["node_id"],
        "title": "Agent 总体趋势调研",
        "summary": "第一次调研。",
        "documents": [{"title": "00_总览", "content": "# 第一次\n"}],
    }
    first = _confirm("archive_research", payload)["result"]
    second = _confirm("archive_research", {**payload, "summary": "第二次调研。"})["result"]

    assert first["research_id"] != second["research_id"]
    tree = service.list_tree()["knowledge_bases"][0]
    assert tree["research_count"] == 2
    assert "保留人工目标" in overview.read_text(encoding="utf-8")
    assert overview.read_text(encoding="utf-8").count(service.MANAGED_START) == 1


def test_archive_preview_becomes_stale_if_knowledge_base_changes(knowledge_root: Path):
    kb = _create_kb()
    preview = service.create_preview(
        "archive_research",
        {
            "knowledge_base_id": kb["node_id"],
            "title": "Memory 调研",
            "documents": [{"title": "00_总览", "content": "# Memory\n"}],
        },
    )
    overview = knowledge_root / kb["path"] / service.OVERVIEW_FILENAME
    overview.write_text(overview.read_text(encoding="utf-8") + "\n人工更新。\n", encoding="utf-8")

    with pytest.raises(ValueError, match="preview is stale"):
        service.confirm_preview(preview["preview_id"])
    assert service.get_preview(preview["preview_id"])["state"] == "stale"
    assert service.list_tree()["knowledge_bases"][0]["research_count"] == 0


def test_archive_swap_failure_restores_original_tree(knowledge_root: Path, monkeypatch: pytest.MonkeyPatch):
    kb = _create_kb()
    preview = service.create_preview(
        "archive_research",
        {
            "knowledge_base_id": kb["node_id"],
            "title": "会失败的调研",
            "documents": [{"title": "00_总览", "content": "# 不应落库\n"}],
        },
    )
    real_replace = service.os.replace

    def fail_stage_swap(source, target):
        if ".staging" in str(source) and Path(target).name == "Agent 学习":
            raise OSError("simulated swap failure")
        return real_replace(source, target)

    monkeypatch.setattr(service.os, "replace", fail_stage_swap)
    with pytest.raises(OSError, match="simulated swap failure"):
        service.confirm_preview(preview["preview_id"])

    assert (knowledge_root / "Agent 学习" / service.KB_MANIFEST).exists()
    assert service.list_tree()["knowledge_bases"][0]["research_count"] == 0
    assert service.get_preview(preview["preview_id"])["state"] == "failed"


def test_research_status_change_requires_preview_and_can_record_supersession(knowledge_root: Path):
    kb = _create_kb()
    old = _confirm(
        "archive_research",
        {
            "knowledge_base_id": kb["node_id"],
            "title": "旧调研",
            "documents": [{"title": "00_总览", "content": "# 旧调研\n"}],
        },
    )["result"]
    current = _confirm(
        "archive_research",
        {
            "knowledge_base_id": kb["node_id"],
            "title": "新调研",
            "documents": [{"title": "00_总览", "content": "# 新调研\n"}],
        },
    )["result"]

    preview = service.create_preview(
        "update_research_status",
        {
            "research_id": old["research_id"],
            "status": "superseded",
            "superseded_by_research_id": current["research_id"],
            "reason": "新调研覆盖了旧结论",
        },
    )
    before = service.get_research_node(old["research_id"])["manifest"]
    assert before["status"] == "active"

    committed = service.confirm_preview(preview["preview_id"])
    assert committed["result"]["status"] == "superseded"
    after = service.get_research_node(old["research_id"])["manifest"]
    assert after["status"] == "superseded"
    assert after["superseded_by_research_id"] == current["research_id"]
