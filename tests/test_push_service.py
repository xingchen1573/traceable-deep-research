"""Tests for backend/services/push_service.py (docs/VISION.md 第5节 "Cron 定时推送").

Config storage (channel_service) and the conversational config tools are covered separately in
tests/test_channel_push_config.py -- this file covers the push logic itself: document
selection per content_strategy, the pure due-check, and run_push_cycle's end-to-end
orchestration (with the model call and Feishu send mocked out).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
from pathlib import Path

import pytest

import backend.services.push_service as push_service
from backend.services import channel_service

TOPIC = "harness-agent-v03-e2e-1781143504"
DOC_1 = "01_核心概念与架构设计.md"
DOC_2 = "02_实操使用指南.md"


@pytest.fixture(autouse=True)
def isolated_legacy_topic(tmp_path, monkeypatch):
    """Legacy compatibility tests must not depend on production knowledge_base data."""
    root = tmp_path / "knowledge_base"
    topic = root / TOPIC
    topic.mkdir(parents=True)
    (topic / "00_总览.md").write_text("# 测试总览\n", encoding="utf-8")
    (topic / DOC_1).write_text("# 核心概念\n", encoding="utf-8")
    (topic / DOC_2).write_text("# 实操指南\n", encoding="utf-8")
    for path in topic.glob("*.md"):
        os.utime(path, (1, 1))
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", root)


# ---------------------------------------------------------------------------
# select_next_document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "00_主题导读.md",
        "01_研究总览与范围.md",
        "02_Product_Overview_Notes.md",
        "03_Summary_and_scope.md",
        "04_摘要与边界.md",
        "05_证据与参考.md",
        "06_Knowledge_QA.md",
        "07_入库建议.md",
        "08_关键_Evidence.md",
        "09_结构化_Sources.md",
        "10_REFERENCES.md",
        "10_sources.md",
        "10_evidence.md",
        "11_来源索引.md",
        "12_证据索引.md",
    ],
)
def test_auxiliary_document_classifier_excludes_normalized_role_titles(filename):
    assert push_service._is_auxiliary_document(Path(filename)) is True


@pytest.mark.parametrize(
    "filename",
    [
        "01_证据原生流水线.md",
        "02_过程指标：轨迹、工具调用与证据绑定.md",
        "03_来源可靠性评估.md",
        "04_Source_Routing_Patterns.md",
        "05_关键_Evidence_如何参与决策.md",
        "06_结构化_Sources_消费策略.md",
        "07_Agent_Summary_Statistics.md",
    ],
)
def test_auxiliary_document_classifier_preserves_substantive_titles(filename):
    assert push_service._is_auxiliary_document(Path(filename)) is False


def test_rotation_pool_excludes_overview_document():
    pool = push_service._rotation_pool(TOPIC)
    names = {p.name for p in pool}
    assert "00_总览.md" not in names
    assert names == {DOC_1, DOC_2}


def test_rotation_first_ever_push_picks_first_document_in_pool():
    doc = push_service.select_next_document(TOPIC, content_strategy="rotation", last_pushed_doc=None, last_pushed_at=None)
    assert doc.name == DOC_1


def test_rotation_continues_after_last_pushed_document():
    far_future = time.time() + 10_000
    doc = push_service.select_next_document(TOPIC, content_strategy="rotation", last_pushed_doc=DOC_1, last_pushed_at=far_future)
    assert doc.name == DOC_2


def test_rotation_wraps_around_to_start():
    far_future = time.time() + 10_000
    doc = push_service.select_next_document(TOPIC, content_strategy="rotation", last_pushed_doc=DOC_2, last_pushed_at=far_future)
    assert doc.name == DOC_1


def test_rotation_new_document_jumps_the_queue_without_disrupting_continuation(tmp_path, monkeypatch):
    """VISION.md 第5节: 新增插队优先推送，推完回到正常轮询顺序. Simulate this with a topic dir
    holding two old docs and one freshly-touched (newer mtime) doc."""
    topic_dir = tmp_path / "kb" / "rotation-topic"
    topic_dir.mkdir(parents=True)
    old_a = topic_dir / "01_old_a.md"
    old_b = topic_dir / "02_old_b.md"
    old_a.write_text("A", encoding="utf-8")
    old_b.write_text("B", encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")

    last_pushed_at = time.time()
    time.sleep(0.01)
    new_doc = topic_dir / "03_new.md"
    new_doc.write_text("N", encoding="utf-8")

    # New doc jumps ahead of normal rotation order, even though 01_old_a would be "next".
    picked = push_service.select_next_document(
        "rotation-topic", content_strategy="rotation", last_pushed_doc="01_old_a.md", last_pushed_at=last_pushed_at
    )
    assert picked.name == "03_new.md"

    # After recording that push, rotation continues from where it left off (01_old_a -> 02_old_b),
    # unaffected by the new-doc detour.
    picked_after = push_service.select_next_document(
        "rotation-topic", content_strategy="rotation", last_pushed_doc="01_old_a.md", last_pushed_at=time.time()
    )
    assert picked_after.name == "02_old_b.md"


def test_new_only_returns_none_when_nothing_new():
    far_future = time.time() + 10_000
    doc = push_service.select_next_document(TOPIC, content_strategy="new_only", last_pushed_doc=DOC_1, last_pushed_at=far_future)
    assert doc is None


def test_new_only_returns_oldest_new_document_when_present():
    doc = push_service.select_next_document(TOPIC, content_strategy="new_only", last_pushed_doc=None, last_pushed_at=0)
    assert doc is not None


def test_random_strategy_returns_a_pool_member():
    doc = push_service.select_next_document(TOPIC, content_strategy="random", last_pushed_doc=None, last_pushed_at=None)
    assert doc.name in {DOC_1, DOC_2}


def test_empty_or_missing_topic_returns_none():
    assert push_service.select_next_document("no-such-topic-xyz", content_strategy="rotation", last_pushed_doc=None, last_pushed_at=None) is None


def test_legacy_rotation_pool_excludes_auxiliary_docs_without_hiding_substantive_evidence_titles(tmp_path, monkeypatch):
    topic_dir = tmp_path / "kb" / "filtered-topic"
    topic_dir.mkdir(parents=True)
    for filename in (
        "00_总览.md",
        "01_证据原生流水线.md",
        "02_Source_Routing_Patterns.md",
        "03_证据与参考.md",
        "04_Knowledge_QA.md",
        "05_入库建议.md",
    ):
        (topic_dir / filename).write_text(filename, encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")

    assert {path.name for path in push_service._rotation_pool("filtered-topic")} == {
        "01_证据原生流水线.md",
        "02_Source_Routing_Patterns.md",
    }


def test_legacy_rotation_pool_with_only_auxiliary_docs_is_empty(tmp_path, monkeypatch):
    topic_dir = tmp_path / "kb" / "aux-only-topic"
    topic_dir.mkdir(parents=True)
    for filename in ("00_总览.md", "05_证据与参考.md", "06_Knowledge_QA.md", "07_入库建议.md"):
        (topic_dir / filename).write_text(filename, encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")

    assert push_service._rotation_pool("aux-only-topic") == []
    assert (
        push_service.select_next_document(
            "aux-only-topic", content_strategy="rotation", last_pushed_doc="05_证据与参考.md", last_pushed_at=0
        )
        is None
    )


# ---------------------------------------------------------------------------
# _rotation_pool / select_next_document -- theme-level aggregation across subtopics
# (docs/VISION.md 第5节 主题级推送聚合)
# ---------------------------------------------------------------------------


def _make_theme(tmp_path, monkeypatch):
    """theme-x/{sub-a,sub-b} each with 2 docs + their own overview, plus a theme-root overview --
    the theme-root overview must never enter the pool (the theme branch never globs topic_dir
    itself, only subtopic dirs)."""
    theme_dir = tmp_path / "kb" / "theme-x"
    sub_a = theme_dir / "sub-a"
    sub_b = theme_dir / "sub-b"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    (theme_dir / "00_主题总览.md").write_text("theme overview", encoding="utf-8")
    (sub_a / "00_总览.md").write_text("a overview", encoding="utf-8")
    (sub_a / "01_a1.md").write_text("a1", encoding="utf-8")
    (sub_a / "02_a2.md").write_text("a2", encoding="utf-8")
    (sub_b / "00_总览.md").write_text("b overview", encoding="utf-8")
    (sub_b / "01_b1.md").write_text("b1", encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")
    return theme_dir, sub_a, sub_b


def test_rotation_pool_aggregates_across_theme_subtopics_and_excludes_all_overviews(tmp_path, monkeypatch):
    _make_theme(tmp_path, monkeypatch)
    pool = push_service._rotation_pool("theme-x")
    identities = {p.relative_to(push_service.KNOWLEDGE_BASE / "theme-x").as_posix() for p in pool}
    assert identities == {"sub-a/01_a1.md", "sub-a/02_a2.md", "sub-b/01_b1.md"}


def test_theme_rotation_continues_across_subtopics_using_qualified_identity(tmp_path, monkeypatch):
    _make_theme(tmp_path, monkeypatch)
    far_future = time.time() + 10_000
    doc = push_service.select_next_document(
        "theme-x", content_strategy="rotation", last_pushed_doc="sub-a/02_a2.md", last_pushed_at=far_future
    )
    # pool is sorted by relative-path string: sub-a/01_a1.md, sub-a/02_a2.md, sub-b/01_b1.md
    assert doc.relative_to(push_service.KNOWLEDGE_BASE / "theme-x").as_posix() == "sub-b/01_b1.md"


def test_theme_rotation_wraps_around_to_first_subtopic(tmp_path, monkeypatch):
    _make_theme(tmp_path, monkeypatch)
    far_future = time.time() + 10_000
    doc = push_service.select_next_document(
        "theme-x", content_strategy="rotation", last_pushed_doc="sub-b/01_b1.md", last_pushed_at=far_future
    )
    assert doc.relative_to(push_service.KNOWLEDGE_BASE / "theme-x").as_posix() == "sub-a/01_a1.md"


def test_theme_rotation_new_doc_in_one_subtopic_jumps_the_queue(tmp_path, monkeypatch):
    theme_dir, sub_a, sub_b = _make_theme(tmp_path, monkeypatch)
    last_pushed_at = time.time()
    time.sleep(0.01)
    new_doc = sub_b / "02_b_new.md"
    new_doc.write_text("new", encoding="utf-8")

    picked = push_service.select_next_document(
        "theme-x", content_strategy="rotation", last_pushed_doc="sub-a/01_a1.md", last_pushed_at=last_pushed_at
    )
    assert picked.relative_to(theme_dir).as_posix() == "sub-b/02_b_new.md"


def test_theme_rotation_identity_disambiguates_same_named_files_across_subtopics(tmp_path, monkeypatch):
    """Collision-safety: two subtopics each happen to have a file literally named "01_x.md" --
    bare-filename identity would be ambiguous (names.index would always resolve to whichever
    pool entry has that name first); the qualified "subtopic/filename.md" identity must resolve
    rotation continuation to the correct one."""
    theme_dir = tmp_path / "kb" / "theme-y"
    sub_a = theme_dir / "sub-a"
    sub_b = theme_dir / "sub-b"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    (sub_a / "01_x.md").write_text("a", encoding="utf-8")
    (sub_b / "01_x.md").write_text("b", encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")

    far_future = time.time() + 10_000
    doc = push_service.select_next_document(
        "theme-y", content_strategy="rotation", last_pushed_doc="sub-a/01_x.md", last_pushed_at=far_future
    )
    assert doc.relative_to(theme_dir).as_posix() == "sub-b/01_x.md"


def test_leaf_topic_rotation_pool_unaffected_by_theme_branch(tmp_path, monkeypatch):
    """A flat leaf topic (no subdirectories with .md) must still take the original single-
    directory glob path, not the theme branch."""
    flat_dir = tmp_path / "kb" / "flat-topic"
    flat_dir.mkdir(parents=True)
    (flat_dir / "00_总览.md").write_text("overview", encoding="utf-8")
    (flat_dir / "01_doc.md").write_text("doc", encoding="utf-8")
    monkeypatch.setattr(push_service, "KNOWLEDGE_BASE", tmp_path / "kb")

    pool = push_service._rotation_pool("flat-topic")
    assert {p.name for p in pool} == {"01_doc.md"}


# ---------------------------------------------------------------------------
# _scope_rotation_pool / select_next_scope_document -- stable KnowledgeScope
# ---------------------------------------------------------------------------


def _install_scope_pool(tmp_path, monkeypatch, research_documents):
    root = tmp_path / "scope-kb"
    research_items = []
    for index, filenames in enumerate(research_documents, start=1):
        research_id = f"research_{index}"
        relative_path = f"theme/{research_id}"
        research_dir = root / relative_path
        research_dir.mkdir(parents=True)
        for filename in filenames:
            (research_dir / filename).write_text(filename, encoding="utf-8")
        research_items.append(
            {
                "id": research_id,
                "path": relative_path,
                "title": f"Research {index}",
                "researched_at": f"2026-09-{index:02d}T00:00:00Z",
                "documents": [{"filename": filename} for filename in filenames],
            }
        )

    class FakeScopeService:
        @staticmethod
        def resolve_scope(scope_type, scope_id):
            assert (scope_type, scope_id) == ("theme", "theme_current")
            return {"title": "多 Agent 与协作", "research_items": research_items}

    class FakeTreeService:
        KNOWLEDGE_ROOT = root

    monkeypatch.setattr(push_service, "_knowledge_scope_service", lambda: FakeScopeService)
    monkeypatch.setattr(push_service, "_knowledge_tree_service", lambda: FakeTreeService)
    return research_items


def _current_style_scope_documents():
    return [
        [
            "00_总览.md",
            "01_核心原理.md",
            "02_主要架构模式.md",
            "03_工程实现.md",
            "04_失败模式与评测.md",
            "05_证据与参考.md",
            "06_Knowledge_QA.md",
            "07_入库建议.md",
        ],
        ["01_Handoff 接受时的活动 Agent 切换规则.md"],
        ["01_状态与一致性模型.md", "02_冲突检测与合并.md", "03_部分失败恢复.md", "04_验证与测试.md"],
    ]


def test_scope_rotation_pool_filters_current_style_twelve_candidates_to_nine(tmp_path, monkeypatch):
    documents = _current_style_scope_documents()
    _install_scope_pool(tmp_path, monkeypatch, documents)
    assert sum(1 for group in documents for filename in group if not filename.startswith("00_")) == 12

    title, pool = push_service._scope_rotation_pool("theme", "theme_current")

    assert title == "多 Agent 与协作"
    assert len(pool) == 9
    assert {item["path"].name for item in pool}.isdisjoint(
        {"05_证据与参考.md", "06_Knowledge_QA.md", "07_入库建议.md"}
    )


def test_scope_rotation_pool_with_only_auxiliary_docs_is_empty(tmp_path, monkeypatch):
    _install_scope_pool(
        tmp_path,
        monkeypatch,
        [["00_总览.md", "05_证据与参考.md", "06_Knowledge_QA.md", "07_入库建议.md"]],
    )

    title, pool = push_service._scope_rotation_pool("theme", "theme_current")
    selected_title, selected = push_service.select_next_scope_document(
        "theme",
        "theme_current",
        content_strategy="rotation",
        last_pushed_doc="research_1/05_证据与参考.md",
        last_pushed_at=0,
    )

    assert title == selected_title == "多 Agent 与协作"
    assert pool == []
    assert selected is None


def test_scope_rotation_with_stale_auxiliary_cursor_falls_back_to_first_learning_doc(tmp_path, monkeypatch):
    _install_scope_pool(tmp_path, monkeypatch, _current_style_scope_documents())
    _title, pool = push_service._scope_rotation_pool("theme", "theme_current")

    _selected_title, selected = push_service.select_next_scope_document(
        "theme",
        "theme_current",
        content_strategy="rotation",
        last_pushed_doc="research_1/07_入库建议.md",
        last_pushed_at=time.time() + 10_000,
    )

    assert selected is not None
    assert selected["identity"] == pool[0]["identity"]
    assert push_service._is_auxiliary_document(selected["path"]) is False


# ---------------------------------------------------------------------------
# _is_due -- pure function, no time mocking needed
# ---------------------------------------------------------------------------


def _local_ts(hour: int, minute: int = 0) -> float:
    return time.mktime((2026, 7, 22, hour, minute, 0, 0, 0, -1))


def test_is_due_false_before_persisted_schedule():
    now = _local_ts(14)
    assert push_service._is_due(now_ts=now, next_push_at=now + 1) is False


def test_is_due_true_at_or_after_persisted_schedule():
    now = _local_ts(9, 15)
    assert push_service._is_due(now_ts=now, next_push_at=now) is True
    assert push_service._is_due(now_ts=now, next_push_at=now - 3600) is True


def test_is_due_false_when_schedule_is_missing():
    assert push_service._is_due(now_ts=_local_ts(9, 15), next_push_at=None) is False


# ---------------------------------------------------------------------------
# render_push_message
# ---------------------------------------------------------------------------


def test_render_push_message_includes_headline_topic_and_evidence():
    text = push_service.render_push_message({"headline": "H", "evidence": ["E1", "E2"]}, topic=TOPIC, document_title="T")
    assert "H" in text
    assert "E1" in text and "E2" in text
    assert TOPIC in text
    assert "T" in text


def test_render_push_message_tolerates_missing_evidence():
    text = push_service.render_push_message({"headline": "H", "evidence": []}, topic=TOPIC, document_title="T")
    assert "H" in text
    assert "依据" not in text


# ---------------------------------------------------------------------------
# run_push_cycle -- end to end, model call + Feishu send mocked
# ---------------------------------------------------------------------------


def _stub_card(monkeypatch, headline: str = "H", evidence=None):
    monkeypatch.setattr(
        push_service, "generate_knowledge_card", lambda doc, topic: {"headline": headline, "evidence": evidence or ["E"]}
    )


def _isolate_candidates(monkeypatch, *channel_ids) -> None:
    """The test-session DB is shared across every test file (see tests/conftest.py) -- earlier
    tests may have left other enabled+bound channels behind, and run_push_cycle scans ALL
    candidates. Scope a test to just the channel(s) it created, using the real query's own
    output filtered down, so tests stay correct regardless of what ran before them."""
    real = channel_service.list_push_candidates
    wanted = set(channel_ids)
    monkeypatch.setattr(channel_service, "list_push_candidates", lambda: [c for c in real() if c["channel_id"] in wanted])


class _FakeFeishu:
    def __init__(self):
        self.sent: list[dict] = []

    def send_reply(self, *, chat_id, text):
        self.sent.append({"chat_id": chat_id, "text": text})
        return {"status": "recorded"}


def test_run_push_cycle_pushes_due_channel_and_records(monkeypatch):
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cycle_e2e")
    channel_service.bind_topic(ch["channel_id"], TOPIC)
    channel_service.upsert_push_config(ch["channel_id"], enabled=True, frequency="daily")

    _stub_card(monkeypatch)
    fake_feishu = _FakeFeishu()
    monkeypatch.setattr(push_service, "_feishu_service", lambda: fake_feishu)
    _isolate_candidates(monkeypatch, ch["channel_id"])

    now = _local_ts(9, 15)
    channel_service._set_next_push_at(ch["channel_id"], now - 1)
    results = push_service.run_push_cycle(now_ts=now)

    row = next(r for r in results if r["channel_id"] == ch["channel_id"])
    assert row["status"] == "pushed"
    assert row["document"] == DOC_1
    assert fake_feishu.sent == [{"chat_id": "oc_push_cycle_e2e", "text": fake_feishu.sent[0]["text"]}]
    assert "H" in fake_feishu.sent[0]["text"]

    cfg = channel_service.get_push_config(ch["channel_id"])
    assert cfg["last_pushed_at"] == now
    assert cfg["last_pushed_doc"] == DOC_1


def test_run_push_cycle_skips_channels_not_due(monkeypatch):
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cycle_not_due")
    channel_service.bind_topic(ch["channel_id"], TOPIC)
    now = _local_ts(9, 15)
    channel_service.upsert_push_config(ch["channel_id"], enabled=True, frequency="daily")
    channel_service.record_push(ch["channel_id"], document=DOC_1, pushed_at=now - 3600)  # 1h ago, daily needs 20h

    _stub_card(monkeypatch)
    fake_feishu = _FakeFeishu()
    monkeypatch.setattr(push_service, "_feishu_service", lambda: fake_feishu)
    _isolate_candidates(monkeypatch, ch["channel_id"])

    results = push_service.run_push_cycle(now_ts=now)
    row = next(r for r in results if r["channel_id"] == ch["channel_id"])
    assert row["status"] == "not_due"
    assert fake_feishu.sent == []


def test_run_push_cycle_new_only_with_no_new_content_is_skipped_not_errored(monkeypatch):
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cycle_new_only")
    channel_service.bind_topic(ch["channel_id"], TOPIC)
    now = _local_ts(9, 15)
    channel_service.upsert_push_config(ch["channel_id"], enabled=True, frequency="daily", content_strategy="new_only")
    channel_service.record_push(ch["channel_id"], document=DOC_1, pushed_at=now - 100_000)  # long ago, past the docs' mtimes
    channel_service._set_next_push_at(ch["channel_id"], now - 1)

    _stub_card(monkeypatch)
    fake_feishu = _FakeFeishu()
    monkeypatch.setattr(push_service, "_feishu_service", lambda: fake_feishu)
    _isolate_candidates(monkeypatch, ch["channel_id"])

    results = push_service.run_push_cycle(now_ts=now)
    row = next(r for r in results if r["channel_id"] == ch["channel_id"])
    assert row["status"] == "no_content"
    assert fake_feishu.sent == []
    cfg = channel_service.get_push_config(ch["channel_id"])
    assert cfg["next_push_at"] > now
    assert push_service._is_due(now_ts=now, next_push_at=cfg["next_push_at"]) is False


def test_run_push_cycle_one_channel_error_does_not_sink_others(monkeypatch):
    failing = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cycle_fail")
    channel_service.bind_topic(failing["channel_id"], TOPIC)
    channel_service.upsert_push_config(failing["channel_id"], enabled=True, frequency="daily")

    healthy = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cycle_healthy")
    channel_service.bind_topic(healthy["channel_id"], TOPIC)
    channel_service.upsert_push_config(healthy["channel_id"], enabled=True, frequency="daily")

    call_count = {"n": 0}

    def flaky_card(doc, topic):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return {"headline": "H", "evidence": []}

    monkeypatch.setattr(push_service, "generate_knowledge_card", flaky_card)
    fake_feishu = _FakeFeishu()
    monkeypatch.setattr(push_service, "_feishu_service", lambda: fake_feishu)
    _isolate_candidates(monkeypatch, failing["channel_id"], healthy["channel_id"])

    now = _local_ts(9, 15)
    channel_service._set_next_push_at(failing["channel_id"], now - 1)
    channel_service._set_next_push_at(healthy["channel_id"], now - 1)
    results = push_service.run_push_cycle(now_ts=now)

    statuses = {r["channel_id"]: r["status"] for r in results}
    assert "error" in statuses.values()
    assert "pushed" in statuses.values()
    assert len(fake_feishu.sent) == 1  # only the healthy channel actually got a message
    assert channel_service.get_push_config(failing["channel_id"])["next_push_at"] == now - 1
    assert channel_service.get_push_config(healthy["channel_id"])["next_push_at"] > now


def test_run_push_cycle_ignores_non_feishu_channels(monkeypatch):
    ch = channel_service.ensure_channel(platform="local", external_id="local_push_channel")
    channel_service.bind_topic(ch["channel_id"], TOPIC)
    channel_service.upsert_push_config(ch["channel_id"], enabled=True, frequency="daily")

    _stub_card(monkeypatch)
    fake_feishu = _FakeFeishu()
    monkeypatch.setattr(push_service, "_feishu_service", lambda: fake_feishu)
    _isolate_candidates(monkeypatch, ch["channel_id"])

    results = push_service.run_push_cycle(now_ts=_local_ts(9, 15))
    assert all(r["channel_id"] != ch["channel_id"] for r in results)
    assert fake_feishu.sent == []


def test_start_push_scheduler_is_a_noop_by_default(monkeypatch):
    """PUSH_SCHEDULER_ENABLED defaults off -- must not spin up a background thread in dev/test."""
    monkeypatch.delenv("PUSH_SCHEDULER_ENABLED", raising=False)
    started = {"called": False}
    monkeypatch.setattr(
        push_service.threading,
        "Thread",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Thread should not be constructed")),
    )
    push_service.start_push_scheduler()  # must return without touching threading.Thread at all


def test_scheduler_start_and_stop_are_idempotent_under_concurrent_calls(monkeypatch):
    push_service.stop_push_scheduler(timeout=1)
    monkeypatch.setenv("PUSH_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("PUSH_TICK_INTERVAL_SECONDS", "3600")
    entered = threading.Event()
    call_count = 0
    count_lock = threading.Lock()

    def fake_cycle():
        nonlocal call_count
        with count_lock:
            call_count += 1
        entered.set()
        return []

    monkeypatch.setattr(push_service, "run_push_cycle", fake_cycle)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(push_service.start_push_scheduler) for _ in range(8)]
            for future in futures:
                future.result(timeout=2)

        assert entered.wait(timeout=2)
        assert call_count == 1
        assert push_service.scheduler_status()["running"] is True
    finally:
        push_service.stop_push_scheduler(timeout=2)
        push_service.stop_push_scheduler(timeout=2)

    assert push_service.scheduler_status()["running"] is False


def test_run_push_cycle_skips_overlapping_tick(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    processed: list[str] = []
    channel = {"channel_id": "feishu:overlap", "platform": "feishu"}
    monkeypatch.setattr(channel_service, "list_push_candidates", lambda: [channel])

    def blocking_process(item, *, now_ts):
        processed.append(item["channel_id"])
        entered.set()
        assert release.wait(timeout=2)
        return {"channel_id": item["channel_id"], "status": "pushed"}

    monkeypatch.setattr(push_service, "_process_channel", blocking_process)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(push_service.run_push_cycle, now_ts=1.0)
        assert entered.wait(timeout=2)
        overlapping = pool.submit(push_service.run_push_cycle, now_ts=1.0)
        assert overlapping.result(timeout=2) == []
        release.set()
        assert first.result(timeout=2) == [{"channel_id": "feishu:overlap", "status": "pushed"}]

    assert processed == ["feishu:overlap"]
