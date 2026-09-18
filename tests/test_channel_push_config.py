"""Tests for docs/VISION.md 第5节 "Cron 定时推送" -- the push-config storage layer and the two
conversational config tools (get/set_channel_push_config_tool).

Push logic itself (document selection, card generation, run_push_cycle orchestration) is
covered separately in tests/test_push_service.py.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backend.services import channel_service
import tools.knowledge_manager as knowledge_manager
import tools._registry as registry
from tools._registry import set_current_thread_id

TOPIC = "harness-agent-v03-e2e-1781143504"


@pytest.fixture(autouse=True)
def isolated_legacy_topic(tmp_path, monkeypatch):
    topic = tmp_path / TOPIC
    topic.mkdir(parents=True)
    (topic / "00_总览.md").write_text("# 测试总览\n", encoding="utf-8")
    (topic / "01_内容.md").write_text("# 内容\n", encoding="utf-8")
    monkeypatch.setattr(knowledge_manager, "KNOWLEDGE_BASE", tmp_path)
    monkeypatch.setattr(registry, "KNOWLEDGE_BASE", tmp_path)


# ---------------------------------------------------------------------------
# backend/services/channel_service.py -- push config storage
# ---------------------------------------------------------------------------


def test_new_channel_has_default_push_config():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_default")
    cfg = channel_service.get_push_config(ch["channel_id"])
    assert cfg == {
        "enabled": False,
        "frequency": "daily",
        "content_strategy": "rotation",
        "push_time": "09:00",
        "timezone": "Asia/Shanghai",
        "next_push_at": None,
        "last_pushed_at": None,
        "last_pushed_doc": None,
    }


def test_get_push_config_on_unregistered_channel_returns_default_not_raise():
    """Like get_topic_binding, this must be lenient -- the scheduler tick calls it across many
    channels every cycle and can't afford to raise for one that was never configured."""
    cfg = channel_service.get_push_config("feishu:oc_never_registered_push_cfg")
    assert cfg["enabled"] is False


def test_upsert_push_config_partial_update_and_round_trip():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_upsert")
    channel_id = ch["channel_id"]
    channel_service.bind_topic(channel_id, TOPIC)

    cfg = channel_service.upsert_push_config(
        channel_id,
        enabled=True,
        frequency="weekly_3x",
        push_time="18:30",
        timezone="America/New_York",
    )
    assert cfg["enabled"] is True
    assert cfg["frequency"] == "weekly_3x"
    assert cfg["content_strategy"] == "rotation"  # untouched field keeps its default
    assert cfg["push_time"] == "18:30"
    assert cfg["timezone"] == "America/New_York"
    assert cfg["next_push_at"] is not None

    cfg2 = channel_service.upsert_push_config(channel_id, content_strategy="random")
    assert cfg2["enabled"] is True  # untouched by this call
    assert cfg2["frequency"] == "weekly_3x"  # untouched by this call
    assert cfg2["content_strategy"] == "random"
    assert cfg2["push_time"] == "18:30"
    assert cfg2["timezone"] == "America/New_York"


def test_upsert_push_config_rejects_invalid_enums():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_bad_enum")
    try:
        channel_service.upsert_push_config(ch["channel_id"], frequency="hourly")
        raise AssertionError("expected ValueError for invalid frequency")
    except ValueError:
        pass
    try:
        channel_service.upsert_push_config(ch["channel_id"], content_strategy="chronological")
        raise AssertionError("expected ValueError for invalid content_strategy")
    except ValueError:
        pass


@pytest.mark.parametrize("push_time", ["9:00", "24:00", "09:60", "noon"])
def test_upsert_push_config_rejects_invalid_push_time(push_time):
    ch = channel_service.ensure_channel(platform="feishu", external_id=f"oc_push_bad_time_{push_time}")
    with pytest.raises(ValueError, match="HH:MM"):
        channel_service.upsert_push_config(ch["channel_id"], push_time=push_time)


def test_upsert_push_config_rejects_unknown_timezone():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_bad_timezone")
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        channel_service.upsert_push_config(ch["channel_id"], timezone="Mars/Olympus")


def test_next_push_uses_channel_wall_clock_and_timezone():
    after = datetime(2026, 9, 14, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    next_push = channel_service.calculate_next_push_at(
        after,
        frequency="daily",
        push_time="09:30",
        timezone="Asia/Shanghai",
        respect_frequency_gap=False,
    )
    assert datetime.fromtimestamp(next_push, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M") == "2026-09-14 09:30"

    after_slot = datetime(2026, 9, 14, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    following = channel_service.calculate_next_push_at(
        after_slot,
        frequency="daily",
        push_time="09:30",
        timezone="Asia/Shanghai",
        respect_frequency_gap=False,
    )
    assert datetime.fromtimestamp(following, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M") == "2026-09-15 09:30"


def test_next_push_handles_new_york_dst_gap_and_overlap_once_per_local_date():
    timezone = ZoneInfo("America/New_York")

    # 02:30 does not exist on the spring-forward date. zoneinfo normalizes that slot to
    # 03:30, retaining the configured minute and keeping the occurrence on the same date.
    spring_after = datetime(2026, 3, 8, 0, 0, tzinfo=timezone).timestamp()
    spring_push = channel_service.calculate_next_push_at(
        spring_after,
        frequency="daily",
        push_time="02:30",
        timezone="America/New_York",
        respect_frequency_gap=False,
    )
    assert datetime.fromtimestamp(spring_push, timezone).isoformat() == "2026-03-08T03:30:00-04:00"

    # 01:30 occurs twice on the fall-back date. Use the first occurrence, then move to the
    # next local date instead of delivering the same daily slot twice.
    fall_after = datetime(2026, 11, 1, 0, 0, tzinfo=timezone).timestamp()
    first_fall_push = channel_service.calculate_next_push_at(
        fall_after,
        frequency="daily",
        push_time="01:30",
        timezone="America/New_York",
        respect_frequency_gap=False,
    )
    first_local = datetime.fromtimestamp(first_fall_push, timezone)
    assert first_local.isoformat() == "2026-11-01T01:30:00-04:00"
    assert first_local.fold == 0

    following_push = channel_service.calculate_next_push_at(
        first_fall_push,
        frequency="daily",
        push_time="01:30",
        timezone="America/New_York",
        respect_frequency_gap=False,
    )
    assert datetime.fromtimestamp(following_push, timezone).isoformat() == "2026-11-02T01:30:00-05:00"


def test_legacy_push_config_schema_migrates_existing_rows_idempotently():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE channel_push_config (
            channel_id       TEXT PRIMARY KEY,
            enabled          INTEGER NOT NULL DEFAULT 0,
            frequency        TEXT NOT NULL DEFAULT 'daily',
            content_strategy TEXT NOT NULL DEFAULT 'rotation',
            last_pushed_at   REAL,
            last_pushed_doc  TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );
        INSERT INTO channel_push_config
            (channel_id, enabled, frequency, content_strategy, created_at, updated_at)
        VALUES ('feishu:legacy_push', 1, 'weekly_3x', 'rotation', 'old', 'old');
        """
    )

    channel_service._ensure_schema(conn)
    channel_service._ensure_schema(conn)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(channel_push_config)")}
    assert {"push_time", "timezone", "next_push_at"} <= columns
    row = conn.execute(
        "SELECT push_time, timezone, next_push_at FROM channel_push_config WHERE channel_id = ?",
        ("feishu:legacy_push",),
    ).fetchone()
    assert dict(row) == {"push_time": "09:00", "timezone": "Asia/Shanghai", "next_push_at": None}


def test_upsert_push_config_on_unknown_channel_raises():
    try:
        channel_service.upsert_push_config("feishu:oc_never_registered_upsert", enabled=True)
        raise AssertionError("expected ValueError for unregistered channel")
    except ValueError:
        pass


def test_record_push_updates_last_pushed_fields():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_record")
    channel_id = ch["channel_id"]
    now = time.time()
    channel_service.record_push(channel_id, document="01_x.md", pushed_at=now)
    cfg = channel_service.get_push_config(channel_id)
    assert cfg["last_pushed_doc"] == "01_x.md"
    assert cfg["last_pushed_at"] == now


def test_record_push_advances_to_next_wall_clock_slot_after_frequency_gap():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_push_record_schedule")
    channel_id = ch["channel_id"]
    channel_service.upsert_push_config(
        channel_id,
        enabled=True,
        frequency="daily",
        push_time="09:00",
        timezone="Asia/Shanghai",
    )
    pushed_at = datetime(2026, 9, 14, 9, 15, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()

    channel_service.record_push(channel_id, document="01_x.md", pushed_at=pushed_at)

    cfg = channel_service.get_push_config(channel_id)
    next_local = datetime.fromtimestamp(cfg["next_push_at"], ZoneInfo("Asia/Shanghai"))
    assert next_local.strftime("%Y-%m-%d %H:%M") == "2026-09-15 09:00"
    assert cfg["next_push_at"] >= pushed_at + channel_service.FREQUENCY_MIN_GAP_HOURS["daily"] * 3600


def test_list_push_candidates_requires_both_binding_and_enabled():
    unbound = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cand_unbound")
    channel_service.upsert_push_config(unbound["channel_id"], enabled=True)  # no topic bound

    disabled = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cand_disabled")
    channel_service.bind_topic(disabled["channel_id"], TOPIC)
    channel_service.upsert_push_config(disabled["channel_id"], enabled=False)

    eligible = channel_service.ensure_channel(platform="feishu", external_id="oc_push_cand_eligible")
    channel_service.bind_topic(eligible["channel_id"], TOPIC)
    channel_service.upsert_push_config(eligible["channel_id"], enabled=True)

    ids = {c["channel_id"] for c in channel_service.list_push_candidates()}
    assert unbound["channel_id"] not in ids
    assert disabled["channel_id"] not in ids
    assert eligible["channel_id"] in ids


# ---------------------------------------------------------------------------
# tools/knowledge_manager.py -- conversational config tools
# ---------------------------------------------------------------------------


def test_get_push_config_tool_reflects_current_channel_state():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_km_push_get")
    set_current_thread_id(f"channel:{ch['channel_id']}:main")
    output = json.loads(knowledge_manager.get_channel_push_config_tool.invoke({}))
    assert output["channel_id"] == ch["channel_id"]
    assert output["push_config"]["enabled"] is False


def test_set_push_config_tool_refuses_enable_without_topic_binding():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_km_push_no_bind")
    set_current_thread_id(f"channel:{ch['channel_id']}:main")
    output = json.loads(knowledge_manager.set_channel_push_config_tool.invoke({"enabled": True}))
    assert output["status"] == "topic_not_bound"
    assert channel_service.get_push_config(ch["channel_id"])["enabled"] is False


def test_set_push_config_tool_succeeds_after_binding():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_km_push_bound")
    set_current_thread_id(f"channel:{ch['channel_id']}:main")
    knowledge_manager.bind_channel_to_topic_tool.invoke({"topic": TOPIC})

    output = json.loads(
        knowledge_manager.set_channel_push_config_tool.invoke(
            {"enabled": True, "frequency": "weekly_1x", "push_time": "18:30", "timezone": "Asia/Tokyo"}
        )
    )
    assert output["status"] == "updated"
    assert output["push_config"]["enabled"] is True
    assert output["push_config"]["frequency"] == "weekly_1x"
    assert output["push_config"]["push_time"] == "18:30"
    assert output["push_config"]["timezone"] == "Asia/Tokyo"


def test_set_push_config_tool_on_unregistered_channel_returns_soft_error():
    set_current_thread_id("bare-cli-thread-no-channel-row-for-push")
    output = json.loads(knowledge_manager.set_channel_push_config_tool.invoke({"enabled": True}))
    assert output["status"] == "channel_not_registered"


def test_set_push_config_tool_rejects_invalid_frequency():
    ch = channel_service.ensure_channel(platform="feishu", external_id="oc_km_push_bad_freq")
    set_current_thread_id(f"channel:{ch['channel_id']}:main")
    try:
        knowledge_manager.set_channel_push_config_tool.invoke({"frequency": "hourly"})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_knowledge_manager_agent_holds_the_push_config_tools(monkeypatch):
    import main
    import workers

    monkeypatch.setattr(main, "load_env_files", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("RESEARCH_UNIFIED_MODEL_CHANNEL_ENABLED", raising=False)
    monkeypatch.delenv("RESEARCH_PREFERRED_PROVIDER_ENABLED", raising=False)
    monkeypatch.delenv("MIMO_FALLBACK_ENABLED", raising=False)
    agent = workers.build_knowledge_manager_agent()
    names = {t.name for t in agent.nodes["tools"].bound.tools_by_name.values()}
    assert {"get_channel_push_config_tool", "set_channel_push_config_tool"} <= names
