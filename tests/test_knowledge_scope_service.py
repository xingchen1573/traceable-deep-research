"""Contracts for KnowledgeScope membership and cross-index aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services import knowledge_scope_service as scope_service
from backend.services import knowledge_tree_service as tree_service
from tools.lightrag_qa import IndexState


@pytest.fixture()
def scope_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(tree_service, "KNOWLEDGE_ROOT", root)
    kb = tree_service.confirm_preview(
        tree_service.create_preview("create_knowledge_base", {"title": "Agent 学习"})["preview_id"]
    )["result"]
    theme = tree_service.confirm_preview(
        tree_service.create_preview("create_theme", {"knowledge_base_id": kb["node_id"], "title": "Harness"})[
            "preview_id"
        ]
    )["result"]

    def archive(title: str, *, theme_id: str | None = None, status: str = "active", date: str = "2026-08-27"):
        payload = {
            "knowledge_base_id": kb["node_id"],
            "theme_id": theme_id,
            "title": title,
            "summary": f"{title} 的摘要",
            "researched_at": f"{date}T10:00:00+08:00",
            "documents": [{"title": "00_总览", "content": f"# {title}\n"}],
        }
        result = tree_service.confirm_preview(
            tree_service.create_preview("archive_research", payload)["preview_id"]
        )["result"]
        manifest_path = root / result["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest["index_status"] = {"status": "ready"}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return result

    direct = archive("Agent 总体趋势", date="2026-08-26")
    stale = archive("Harness 运行机制", theme_id=theme["node_id"], status="possibly_stale")
    superseded = archive("旧版 Harness", theme_id=theme["node_id"], status="superseded", date="2026-07-01")
    return root, kb, theme, direct, stale, superseded


def test_scope_membership_and_history_policy(scope_tree):
    _root, kb, theme, direct, stale, superseded = scope_tree
    kb_scope = scope_service.resolve_scope("knowledge_base", kb["node_id"])
    assert {item["id"] for item in kb_scope["research_items"]} == {direct["research_id"], stale["research_id"]}
    assert kb_scope["excluded_research_items"] == [
        {"research_id": superseded["research_id"], "title": "旧版 Harness", "status": "superseded"}
    ]

    theme_scope = scope_service.resolve_scope("theme", theme["node_id"], include_history=True)
    assert {item["id"] for item in theme_scope["research_items"]} == {
        stale["research_id"],
        superseded["research_id"],
    }
    weights = {item["id"]: item["scope_weight"] for item in theme_scope["research_items"]}
    assert weights[stale["research_id"]] == 0.65
    assert weights[superseded["research_id"]] == 0.35


def test_scope_query_deduplicates_citations_and_warns_for_stale(scope_tree, monkeypatch: pytest.MonkeyPatch):
    _root, kb, _theme, direct, stale, _superseded = scope_tree

    def query_one(question, research_id, research_path, *, mode, include_retrieval_trace):
        assert Path(research_path).exists()
        return {
            "status": "answered",
            "research_id": research_id,
            "answer": f"{research_id} 对问题的回答",
            "citations": [
                {"path": "shared.md", "content": "Harness 使用约束与反馈闭环。"},
                {"path": f"{research_id}.md", "content": f"{research_id} 独有证据。"},
            ],
        }

    monkeypatch.setattr(scope_service, "query_research_lightrag", query_one)
    result = scope_service.query_scope("knowledge_base", kb["node_id"], "Harness 如何形成反馈闭环？")
    assert result["status"] == "answered"
    assert set(result["queried_research_ids"]) == {direct["research_id"], stale["research_id"]}
    assert len(result["citations"]) == 3
    assert any("可能已过时" in warning for warning in result["warnings"])
    assert "2 次调研" in result["answer"]


def test_scope_query_reports_pending_indexes(scope_tree):
    root, _kb, _theme, direct, _stale, _superseded = scope_tree
    manifest_path = root / direct["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_status"] = {"status": "pending"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    result = scope_service.query_scope("research", direct["research_id"], "Agent 趋势是什么？")
    assert result["status"] == "insufficient_index"
    assert result["queried_research_ids"] == []
    assert result["skipped_research_items"][0]["reason"] == "index_pending"


def test_scope_query_reports_failed_queries_instead_of_empty_answer(scope_tree, monkeypatch: pytest.MonkeyPatch):
    _root, kb, _theme, _direct, _stale, _superseded = scope_tree

    monkeypatch.setattr(
        scope_service,
        "query_research_lightrag",
        lambda *_args, **_kwargs: {
            "status": "error",
            "answer": "",
            "citations": [],
            "gaps": ["LightRAG 查询失败：Connection error."],
        },
    )

    result = scope_service.query_scope("knowledge_base", kb["node_id"], "Agent 如何工作？")

    assert result["status"] == "query_failed"
    assert result["answer"] == "当前范围的调研索引已就绪，但问答查询暂时失败，请稍后重试。"
    assert result["queried_research_ids"] == []
    query_failures = [item for item in result["skipped_research_items"] if "reason" in item]
    assert len(query_failures) == 2
    assert all("Connection error" in item["reason"] for item in query_failures)
    assert "查询均失败" in result["gaps"][0]


def test_scope_query_keeps_successful_answers_and_reports_partial_query_failure(
    scope_tree, monkeypatch: pytest.MonkeyPatch
):
    _root, kb, _theme, direct, stale, _superseded = scope_tree

    def query_one(_question, research_id, _research_path, **_kwargs):
        if research_id == direct["research_id"]:
            return {
                "status": "answered",
                "answer": "Agent 通过反馈循环持续执行任务。",
                "citations": [{"path": "00_总览.md", "content": "判断、行动、反馈。"}],
            }
        return {"status": "error", "answer": "", "citations": [], "gaps": ["模型连接失败。"]}

    monkeypatch.setattr(scope_service, "query_research_lightrag", query_one)
    result = scope_service.query_scope("knowledge_base", kb["node_id"], "Agent 如何工作？")

    assert result["status"] == "answered"
    assert result["answer"] == "Agent 通过反馈循环持续执行任务。"
    assert result["queried_research_ids"] == [direct["research_id"]]
    assert result["citations"][0]["research_id"] == direct["research_id"]
    query_failures = [item for item in result["skipped_research_items"] if "reason" in item]
    assert query_failures == [
        {
            "research_id": stale["research_id"],
            "title": "Harness 运行机制",
            "status": "possibly_stale",
            "reason": "query_error: 模型连接失败。",
        }
    ]
    assert "未参与本次回答" in result["gaps"][0]


def test_scope_index_graph_and_vectors_aggregate_research_workspaces(scope_tree, tmp_path, monkeypatch):
    _root, kb, _theme, direct, stale, _superseded = scope_tree
    workspaces = tmp_path / "lightrag_workspaces"
    monkeypatch.setattr(scope_service, "LIGHTRAG_WORKSPACES", workspaces)

    metadata_by_id = {}
    for index, research in enumerate((direct, stale), start=1):
        research_id = research["research_id"]
        workspace_name = f"research_{research_id[-12:]}"
        metadata_by_id[research_id] = {
            "embedding_model": "test-embedding",
            "embedding_dim": 8,
            "chunk_size": 500,
            "chunk_overlap_size": 60,
            "indexed_at": f"2026-08-2{index} 10:00:00",
            "lightrag_workspace": workspace_name,
        }
        storage = workspaces / "research" / research_id / workspace_name
        storage.mkdir(parents=True)
        other = "B" if index == 1 else "C"
        (storage / "graph_chunk_entity_relation.graphml").write_text(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="undirected">
    <node id="A"><data key="d0">A</data><data key="d1">concept</data><data key="d2">A-{index}</data></node>
    <node id="{other}"><data key="d0">{other}</data><data key="d1">concept</data></node>
    <edge source="A" target="{other}"><data key="d7">1</data><data key="d8">related</data></edge>
  </graph>
</graphml>''',
            encoding="utf-8",
        )
        for filename, count in (
            ("vdb_chunks.json", index),
            ("vdb_entities.json", index + 1),
            ("vdb_relationships.json", index + 2),
        ):
            (storage / filename).write_text(json.dumps({"data": [{}] * count}), encoding="utf-8")

    monkeypatch.setattr(
        scope_service,
        "research_lightrag_index_state",
        lambda research_id, _path: IndexState(True, "ready", metadata_by_id[research_id]),
    )

    status = scope_service.scope_index_status("knowledge_base", kb["node_id"])
    assert status["index_status"] == "ready"
    assert status["diagnosis"]["ready_count"] == 2

    vectors = scope_service.scope_vectors("knowledge_base", kb["node_id"])
    assert vectors["chunks"] == 3
    assert vectors["entities"] == 5
    assert vectors["relationships"] == 7

    graph = scope_service.scope_graph("knowledge_base", kb["node_id"])
    assert graph["nodes"] == 3
    assert graph["edges"] == 2
    assert next(node for node in graph["graph_nodes"] if node["id"] == "A")["degree"] == 2

    detail = scope_service.scope_graph_node_detail("knowledge_base", kb["node_id"], "A")
    assert {edge["entity_id"] for edge in detail["edges"]} == {"B", "C"}
    assert detail["description"] == "A-1<SEP>A-2"
