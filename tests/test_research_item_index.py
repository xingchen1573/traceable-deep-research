"""Contract tests for stable-ID ResearchItem LightRAG workspaces."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.services import index_service, knowledge_tree_service
from tools import lightrag_qa


def _research_dir(tmp_path: Path, research_id: str) -> Path:
    path = tmp_path / "knowledge_base" / "Agent 学习" / "Harness" / "2026-08-27_调研"
    path.mkdir(parents=True)
    (path / "_research.json").write_text(
        json.dumps({"schema_version": 1, "type": "research", "research_id": research_id, "title": "调研"}),
        encoding="utf-8",
    )
    (path / "00_总览.md").write_text("# 调研\n", encoding="utf-8")
    return path


def test_research_lightrag_workspace_uses_stable_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    research_id = "research_" + "a" * 32
    research_path = _research_dir(tmp_path, research_id)
    workspaces = tmp_path / "lightrag_workspaces"
    captured = {}

    async def fake_build(topic, docs, metadata, *, force, workspace_dir=None, workspace_name=None):
        captured.update(
            {
                "topic": topic,
                "docs": docs,
                "metadata": metadata,
                "force": force,
                "workspace_dir": workspace_dir,
                "workspace_name": workspace_name,
            }
        )
        return {
            "status": "indexed",
            "topic": topic,
            "workspace": str(workspace_dir),
            "indexed_documents": len(docs),
            "metadata": metadata,
        }

    monkeypatch.setattr(lightrag_qa, "LIGHTRAG_WORKSPACES", workspaces)
    monkeypatch.setattr(lightrag_qa, "_build_lightrag_index_async", fake_build)
    monkeypatch.setattr(lightrag_qa, "_run_async", asyncio.run)

    result = lightrag_qa.build_research_lightrag_index(research_id, research_path)
    assert result["research_id"] == research_id
    assert captured["workspace_dir"] == workspaces / "research" / research_id
    assert captured["workspace_name"] == f"research_{research_id[-12:]}"
    assert captured["metadata"]["scope_type"] == "research"
    assert captured["metadata"]["scope_id"] == research_id


def test_research_index_job_failure_marks_index_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(knowledge_tree_service, "KNOWLEDGE_ROOT", root)
    kb = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview("create_knowledge_base", {"title": "Agent 学习"})["preview_id"]
    )["result"]
    archived = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview(
            "archive_research",
            {
                "knowledge_base_id": kb["node_id"],
                "title": "索引失败测试",
                "documents": [{"title": "00_总览", "content": "# 内容仍应保留\n"}],
            },
        )["preview_id"]
    )["result"]
    research_id = archived["research_id"]
    captured = {}

    def fake_create(*, job_type, input_data, target, steps):
        captured["target"] = target
        return {"job_id": "job_index_failure", "job_type": job_type, "status": "queued", "input": input_data}

    monkeypatch.setattr(index_service.job_service, "create", fake_create)
    monkeypatch.setattr(index_service.job_service, "update_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(index_service, "current_job_id", lambda: "job_index_failure")
    monkeypatch.setattr(
        index_service,
        "build_research_lightrag_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )

    job = index_service.create_research_index_job(research_id)
    assert job["job_id"] == "job_index_failure"
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        captured["target"]({})

    node = knowledge_tree_service.get_research_node(research_id)
    assert node["manifest"]["index_status"]["status"] == "failed"
    assert "embedding unavailable" in node["manifest"]["index_status"]["error"]
    assert (Path(node["absolute_path"]) / "00_总览.md").read_text(encoding="utf-8").startswith("# 内容仍应保留")


def test_delete_research_index_removes_only_rebuildable_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "knowledge_base"
    workspaces = tmp_path / "lightrag_workspaces"
    monkeypatch.setattr(knowledge_tree_service, "KNOWLEDGE_ROOT", root)
    monkeypatch.setattr(index_service, "LIGHTRAG_WORKSPACES", workspaces)
    kb = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview("create_knowledge_base", {"title": "Agent 学习"})["preview_id"]
    )["result"]
    archived = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview(
            "archive_research",
            {
                "knowledge_base_id": kb["node_id"],
                "title": "可重建索引",
                "documents": [{"title": "00_总览", "content": "# 正文不能删除\n"}],
            },
        )["preview_id"]
    )["result"]
    research_id = archived["research_id"]
    workspace = workspaces / "research" / research_id
    workspace.mkdir(parents=True)
    (workspace / "index_metadata.json").write_text("{}", encoding="utf-8")

    result = index_service.delete_research_index(research_id)

    assert result["status"] == "deleted"
    assert not workspace.exists()
    node = knowledge_tree_service.get_research_node(research_id)
    assert node["manifest"]["index_status"]["status"] == "pending"
    assert (Path(node["absolute_path"]) / "00_总览.md").exists()
