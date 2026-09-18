"""Tests for the Feishu WS hang watchdog (backend/services/feishu_ws_service.py).

Covers the fix for the incident where lark-oapi's own reconnect logic died
silently after a ping_timeout that followed a Feishu-side 502 -- the process
stayed alive but the WS connection was dead, with no automatic recovery for
2+ hours (see docs/evaluations/ for the real incident writeup). The watchdog
detects "no WS activity for too long" and exits the process so Docker's
`restart: unless-stopped` can recover it.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import time

import pytest


def _reload_feishu_ws_service():
    """Reload the module so ``_ACTIVE_WATCHDOG`` module-level state doesn't
    leak between tests."""
    from backend.services import feishu_ws_service

    return importlib.reload(feishu_ws_service)


@pytest.fixture(autouse=True)
def _isolated_module():
    module = _reload_feishu_ws_service()
    yield module
    _reload_feishu_ws_service()


class _FakeClock:
    """Deterministic stand-in for time.monotonic()/time.time()."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_watchdog_touch_resets_staleness(_isolated_module):
    module = _isolated_module
    clock = _FakeClock()
    watchdog = module.HangWatchdog(
        timeout_seconds=60,
        check_interval_seconds=1000,  # background thread never fires in this test
        heartbeat_path=None,
        clock=clock,
        wall_clock=clock,
    )
    watchdog.touch()
    clock.advance(30)
    assert watchdog.seconds_since_last_activity() == pytest.approx(30)
    watchdog.touch()
    assert watchdog.seconds_since_last_activity() == pytest.approx(0)


def test_watchdog_fires_on_timeout_via_background_thread(_isolated_module):
    """The core regression guard: if nothing calls touch() for longer than
    timeout_seconds, the watchdog's background thread must call on_timeout
    (in production, this is os._exit(1))."""
    module = _isolated_module
    clock = _FakeClock()
    fired = threading_event = __import__("threading").Event()
    captured = {}

    def on_timeout(stale_for_seconds: float) -> None:
        captured["stale_for_seconds"] = stale_for_seconds
        fired.set()

    watchdog = module.HangWatchdog(
        timeout_seconds=0.05,
        check_interval_seconds=0.01,
        heartbeat_path=None,
        on_timeout=on_timeout,
        clock=time.monotonic,  # real clock so the background thread's sleep loop progresses
        wall_clock=time.time,
    )
    watchdog.start()
    assert fired.wait(timeout=2.0), "watchdog did not fire on_timeout within 2s"
    assert captured["stale_for_seconds"] >= 0.05


def test_watchdog_does_not_fire_while_touched_periodically(_isolated_module):
    """A watchdog that keeps receiving touch() calls faster than its timeout
    must never fire -- guards against false positives on a healthy connection."""
    module = _isolated_module
    fired = __import__("threading").Event()

    watchdog = module.HangWatchdog(
        timeout_seconds=0.2,
        check_interval_seconds=0.02,
        heartbeat_path=None,
        on_timeout=lambda stale_for_seconds: fired.set(),
        clock=time.monotonic,
        wall_clock=time.time,
    )
    watchdog.start()
    for _ in range(10):
        time.sleep(0.05)
        watchdog.touch()
    watchdog.stop()
    assert not fired.is_set()


def test_watchdog_does_not_kill_long_running_message_handler(_isolated_module):
    module = _isolated_module
    fired = __import__("threading").Event()
    watchdog = module.HangWatchdog(
        timeout_seconds=0.05,
        check_interval_seconds=0.01,
        heartbeat_path=None,
        on_timeout=lambda stale_for_seconds: fired.set(),
        clock=time.monotonic,
        wall_clock=time.time,
    )
    watchdog.start()
    watchdog.begin_processing()
    time.sleep(0.15)
    assert not fired.is_set()

    watchdog.end_processing()
    assert fired.wait(timeout=1.0), "watchdog should resume normal timeout checks after handling"


def test_watchdog_writes_heartbeat_file(tmp_path, _isolated_module):
    module = _isolated_module
    heartbeat_path = tmp_path / "heartbeat.txt"
    clock = _FakeClock()
    watchdog = module.HangWatchdog(
        timeout_seconds=60,
        check_interval_seconds=1000,
        heartbeat_path=heartbeat_path,
        clock=clock,
        wall_clock=clock,
    )
    watchdog.touch()
    assert heartbeat_path.exists()
    first_content = heartbeat_path.read_text(encoding="utf-8")
    assert first_content == str(clock.value)


def test_watchdog_heartbeat_file_write_is_debounced(tmp_path, _isolated_module):
    """Bursts of inbound frames must not turn into a heartbeat-file write storm."""
    module = _isolated_module
    heartbeat_path = tmp_path / "heartbeat.txt"
    clock = _FakeClock()
    watchdog = module.HangWatchdog(
        timeout_seconds=60,
        check_interval_seconds=1000,
        heartbeat_path=heartbeat_path,
        clock=clock,
        wall_clock=clock,
    )
    watchdog.touch()
    first_mtime = heartbeat_path.stat().st_mtime_ns
    clock.advance(1)  # inside the 5s debounce window
    watchdog.touch()
    assert heartbeat_path.stat().st_mtime_ns == first_mtime

    clock.advance(10)  # past the debounce window
    watchdog.touch()
    assert heartbeat_path.read_text(encoding="utf-8") == str(clock.value)


def test_install_hang_watchdog_wires_reconnect_hooks_and_touches_on_call(_isolated_module):
    module = _isolated_module

    class FakeClient:
        def __init__(self) -> None:
            self.on_reconnecting = lambda: None
            self.on_reconnected = lambda: None

        async def _handle_message(self, msg):
            return f"handled:{msg}"

    client = FakeClient()
    watchdog = module.install_hang_watchdog(
        client, timeout_seconds=999, check_interval_seconds=999, heartbeat_path=None
    )
    watchdog._last_activity = 0.0  # force staleness so we can observe touch() resetting it

    client.on_reconnecting()
    assert watchdog.seconds_since_last_activity() < 1.0

    watchdog._last_activity = 0.0
    client.on_reconnected()
    assert watchdog.seconds_since_last_activity() < 1.0

    watchdog._last_activity = 0.0
    result = asyncio.run(client._handle_message("frame"))
    assert result == "handled:frame"
    assert watchdog.seconds_since_last_activity() < 1.0


def test_install_hang_watchdog_preserves_original_reconnect_callbacks(_isolated_module):
    """The watchdog must not swallow any existing on_reconnecting/on_reconnected
    callback a caller had already wired (e.g. future observability hooks)."""
    module = _isolated_module
    calls = []

    class FakeClient:
        def __init__(self) -> None:
            self.on_reconnecting = lambda: calls.append("reconnecting")
            self.on_reconnected = lambda: calls.append("reconnected")

    client = FakeClient()
    module.install_hang_watchdog(
        client, timeout_seconds=999, check_interval_seconds=999, heartbeat_path=None
    )
    client.on_reconnecting()
    client.on_reconnected()
    assert calls == ["reconnecting", "reconnected"]


def test_handle_ws_message_event_touches_active_watchdog(monkeypatch, _isolated_module):
    """Even if the _handle_message hook is somehow missing/bypassed, an
    actual user message reaching handle_ws_message_event must still count
    as WS activity -- this is the redundant safety net."""
    module = _isolated_module

    class FakeClient:
        def __init__(self) -> None:
            self.on_reconnecting = lambda: None
            self.on_reconnected = lambda: None
            # No _handle_message attribute -- simulates a future lark-oapi
            # version that renamed/removed the internal method.

    client = FakeClient()
    watchdog = module.install_hang_watchdog(
        client, timeout_seconds=999, check_interval_seconds=999, heartbeat_path=None
    )
    watchdog._last_activity = 0.0

    monkeypatch.setattr(
        module,
        "feishu_service",
        type("_F", (), {"handle_event": staticmethod(lambda payload, validate_token=False: {"status": "ok"})})(),
    )
    module.handle_ws_message_event({"event": {"message": {"message_id": "om_1", "chat_id": "oc_1"}}})
    assert watchdog.seconds_since_last_activity() < 1.0


def test_install_hang_watchdog_reads_env_overrides(monkeypatch, _isolated_module):
    module = _isolated_module
    monkeypatch.setenv("FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS", "123")
    monkeypatch.setenv("FEISHU_WS_WATCHDOG_CHECK_INTERVAL_SECONDS", "7")

    class FakeClient:
        pass

    watchdog = module.install_hang_watchdog(FakeClient(), heartbeat_path=None)
    assert watchdog._timeout_seconds == 123.0
    assert watchdog._check_interval_seconds == 7.0
    assert watchdog._heartbeat_path is None


def test_install_hang_watchdog_uses_configured_default_heartbeat_path(
    tmp_path, monkeypatch, _isolated_module
):
    heartbeat_path = tmp_path / "configured-heartbeat.txt"
    monkeypatch.setenv("FEISHU_WS_HEARTBEAT_PATH", str(heartbeat_path))
    module = _reload_feishu_ws_service()

    class FakeClient:
        pass

    watchdog = module.install_hang_watchdog(FakeClient())
    assert watchdog._heartbeat_path == heartbeat_path


def test_healthcheck_script_reports_unhealthy_when_heartbeat_missing(tmp_path, monkeypatch):
    import scripts.feishu_ws_healthcheck as healthcheck

    monkeypatch.setattr(healthcheck, "HEARTBEAT_PATH", tmp_path / "missing.txt")
    assert healthcheck.main() == 1


def test_healthcheck_script_uses_configured_heartbeat_path(tmp_path, monkeypatch):
    heartbeat_path = tmp_path / "configured-heartbeat.txt"
    heartbeat_path.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setenv("FEISHU_WS_HEARTBEAT_PATH", str(heartbeat_path))
    import scripts.feishu_ws_healthcheck as healthcheck

    healthcheck = importlib.reload(healthcheck)
    assert healthcheck.HEARTBEAT_PATH == heartbeat_path
    assert healthcheck.main() == 0


def test_healthcheck_script_reports_healthy_when_heartbeat_fresh(tmp_path, monkeypatch):
    import scripts.feishu_ws_healthcheck as healthcheck

    heartbeat_path = tmp_path / "heartbeat.txt"
    heartbeat_path.write_text(str(time.time()), encoding="utf-8")
    monkeypatch.setattr(healthcheck, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setenv("FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS", "600")
    assert healthcheck.main() == 0


def test_healthcheck_script_reports_unhealthy_when_heartbeat_stale(tmp_path, monkeypatch):
    import scripts.feishu_ws_healthcheck as healthcheck

    heartbeat_path = tmp_path / "heartbeat.txt"
    heartbeat_path.write_text(str(time.time()), encoding="utf-8")
    stale_time = time.time() - 700
    os.utime(heartbeat_path, (stale_time, stale_time))
    monkeypatch.setattr(healthcheck, "HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setenv("FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS", "600")
    assert healthcheck.main() == 1
