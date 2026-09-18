"""Stable KnowledgeScope channel binding and push rotation tests."""

from __future__ import annotations

import time

import pytest

from backend.services import channel_service, knowledge_tree_service, push_service
from backend.services import knowledge_scope_service
from tools import knowledge_qa
from tools._registry import set_current_thread_id


def _build_scope_tree(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "knowledge_base"
    monkeypatch.setattr(knowledge_tree_service, "KNOWLEDGE_ROOT", root)
    kb = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview("create_knowledge_base", {"title": "Agent 学习"})["preview_id"]
    )["result"]
    theme = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview("create_theme", {"knowledge_base_id": kb["node_id"], "title": "Harness"})[
            "preview_id"
        ]
    )["result"]
    research = knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview(
            "archive_research",
            {
                "knowledge_base_id": kb["node_id"],
                "theme_id": theme["node_id"],
                "title": "Harness vs Loop",
                "documents": [
                    {"title": "00_总览", "content": "# 总览\n"},
                    {"title": "01_关键机制", "content": "# 机制\n\n约束与反馈。\n"},
                ],
            },
        )["preview_id"]
    )["result"]
    return root, kb, theme, research


def test_scope_binding_replaces_legacy_topic_and_is_stable(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _root, kb, theme, _research = _build_scope_tree(tmp_path, monkeypatch)
    channel = channel_service.ensure_channel(platform="local", external_id="scope-binding-stable")
    channel_service.bind_topic(channel["channel_id"], "legacy-topic")

    bound = channel_service.bind_scope(channel["channel_id"], "theme", theme["node_id"])
    assert bound["topic_binding"] is None
    assert bound["scope_type"] == "theme"
    assert bound["scope_id"] == theme["node_id"]
    assert channel_service.get_scope_binding(channel["channel_id"]) == {
        "scope_type": "theme",
        "scope_id": theme["node_id"],
        "legacy": False,
    }

    knowledge_tree_service.confirm_preview(
        knowledge_tree_service.create_preview("rename_node", {"node_id": theme["node_id"], "title": "Harness 工程"})[
            "preview_id"
        ]
    )
    assert channel_service.get_scope_binding(channel["channel_id"])["scope_id"] == theme["node_id"]
    assert kb["node_id"]


def test_scope_push_uses_research_id_document_identity(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _root, kb, _theme, research = _build_scope_tree(tmp_path, monkeypatch)
    channel = channel_service.ensure_channel(platform="feishu", external_id="oc_scope_push")
    channel_service.bind_scope(channel["channel_id"], "knowledge_base", kb["node_id"])
    channel_service.upsert_push_config(channel["channel_id"], enabled=True, frequency="daily")

    monkeypatch.setattr(
        channel_service,
        "list_push_candidates",
        lambda: [{**channel_service.get_channel(channel["channel_id"]), **channel_service.get_push_config(channel["channel_id"])}],
    )
    monkeypatch.setattr(
        push_service,
        "generate_knowledge_card",
        lambda document, topic: {"headline": f"{topic} 核心结论", "evidence": [document.stem]},
    )

    sent = []

    class FakeFeishu:
        def send_reply(self, *, chat_id, text):
            sent.append({"chat_id": chat_id, "text": text})
            return {"status": "recorded"}

    monkeypatch.setattr(push_service, "_feishu_service", lambda: FakeFeishu())
    now = time.mktime((2026, 8, 27, 9, 15, 0, 0, 0, -1))
    channel_service._set_next_push_at(channel["channel_id"], now - 1)
    result = push_service.run_push_cycle(now_ts=now)[0]
    assert result["status"] == "pushed"
    assert result["document"] == "01_关键机制.md"
    assert "Agent 学习" in sent[0]["text"]
    config = channel_service.get_push_config(channel["channel_id"])
    assert config["last_pushed_doc"] == f"{research['research_id']}/01_关键机制.md"


def test_channel_default_qa_uses_stable_scope_binding(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _root, kb, _theme, _research = _build_scope_tree(tmp_path, monkeypatch)
    channel = channel_service.ensure_channel(platform="local", external_id="scope-query-default")
    channel_service.bind_scope(channel["channel_id"], "knowledge_base", kb["node_id"])
    set_current_thread_id(channel_service.thread_id_for(channel["channel_id"], "kb_qa"))
    calls = []

    def query_scope(scope_type, scope_id, question, *, mode):
        calls.append((scope_type, scope_id, question, mode))
        return {
            "status": "answered",
            "scope": {"type": scope_type, "id": scope_id},
            "answer": "稳定 scope 回答",
            "citations": [],
            "gaps": [],
        }

    monkeypatch.setattr(knowledge_scope_service, "query_scope", query_scope)
    result = knowledge_qa.query_knowledge_base.invoke({"question": "Harness 是什么？"})
    assert "稳定 scope 回答" in result
    assert calls == [("knowledge_base", kb["node_id"], "Harness 是什么？", "mix")]
