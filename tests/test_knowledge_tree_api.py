"""HTTP contract tests for the manifest-backed knowledge tree API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.services import knowledge_tree_service


client = TestClient(app)


@pytest.fixture()
def knowledge_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(knowledge_tree_service, "KNOWLEDGE_ROOT", root)
    return root


def test_preview_confirm_and_list_tree_over_http(knowledge_root: Path):
    preview_response = client.post(
        "/api/v1/knowledge-tree/previews",
        json={"action": "create_knowledge_base", "payload": {"title": "Agent 学习"}},
    )
    assert preview_response.status_code == 200, preview_response.text
    preview = preview_response.json()["data"]
    assert preview["state"] == "pending"
    assert "plan" not in preview
    assert not (knowledge_root / "Agent 学习").exists()

    confirm_response = client.post(f"/api/v1/knowledge-tree/previews/{preview['preview_id']}/confirm")
    assert confirm_response.status_code == 200, confirm_response.text
    committed = confirm_response.json()["data"]
    assert committed["state"] == "committed"

    tree_response = client.get("/api/v1/knowledge-tree")
    assert tree_response.status_code == 200, tree_response.text
    tree = tree_response.json()["data"]
    assert tree["knowledge_bases"][0]["title"] == "Agent 学习"
    assert tree["legacy"] == []


def test_unknown_knowledge_node_returns_structured_404(knowledge_root: Path):
    response = client.post(
        "/api/v1/knowledge-tree/previews",
        json={"action": "rename_node", "payload": {"node_id": "kb_missing", "title": "不存在"}},
    )
    assert response.status_code == 404
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "knowledge_node_not_found"


def test_preview_action_is_schema_restricted(knowledge_root: Path):
    response = client.post(
        "/api/v1/knowledge-tree/previews",
        json={"action": "delete_everything", "payload": {}},
    )
    assert response.status_code == 422


def test_read_overview_and_research_document_over_http(knowledge_root: Path):
    kb_preview = client.post(
        "/api/v1/knowledge-tree/previews",
        json={"action": "create_knowledge_base", "payload": {"title": "Agent 学习"}},
    ).json()["data"]
    kb = client.post(f"/api/v1/knowledge-tree/previews/{kb_preview['preview_id']}/confirm").json()["data"]["result"]
    archive_preview = client.post(
        "/api/v1/knowledge-tree/previews",
        json={
            "action": "archive_research",
            "payload": {
                "knowledge_base_id": kb["node_id"],
                "title": "Agent 总体趋势调研",
                "documents": [{"title": "00_总览", "content": "# Agent 总体趋势\n\n正文。"}],
            },
        },
    ).json()["data"]
    research = client.post(
        f"/api/v1/knowledge-tree/previews/{archive_preview['preview_id']}/confirm"
    ).json()["data"]["result"]

    overview = client.get(f"/api/v1/knowledge-tree/nodes/knowledge_base/{kb['node_id']}/overview")
    assert overview.status_code == 200
    assert "Agent 总体趋势调研" in overview.json()["data"]["content"]

    document = client.get(
        f"/api/v1/knowledge-tree/research/{research['research_id']}/documents/00_%E6%80%BB%E8%A7%88.md"
    )
    assert document.status_code == 200
    assert document.json()["data"]["content"].startswith("# Agent 总体趋势")

    traversal = client.get(
        f"/api/v1/knowledge-tree/research/{research['research_id']}/documents/..%2F_kb.json"
    )
    assert traversal.status_code in {404, 422}
