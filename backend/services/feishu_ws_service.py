"""Feishu long-connection worker support.

This module keeps the SDK boundary thin: SDK events are normalized into the
same payload shape used by the HTTP webhook adapter, then routed through the
shared Channel Runtime.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    from ...observability import emit_event
except ImportError:  # pragma: no cover
    from observability import emit_event

from . import feishu_service

ROOT = Path(__file__).resolve().parent.parent.parent
_configured_heartbeat_path = os.getenv("FEISHU_WS_HEARTBEAT_PATH", "").strip()
DEFAULT_HEARTBEAT_PATH = (
    Path(_configured_heartbeat_path).expanduser()
    if _configured_heartbeat_path
    else ROOT / "data" / "feishu_ws_heartbeat.txt"
)
_USE_DEFAULT_HEARTBEAT_PATH = object()
# lark-oapi's WS client defaults ping_interval to 120s (server can push a
# different value via ClientConfig, but we can't know that ahead of connect).
# 600s = 5x that default, leaving headroom for a couple of missed keepalive
# cycles before treating the connection as hung. See
# docs/evaluations/feishu-ws-watchdog-e2e-<date>.md for the incident this
# guards against: SDK reconnect silently died after a ping_timeout that
# followed a Feishu-side 502, and nothing noticed for 2+ hours.
DEFAULT_WATCHDOG_TIMEOUT_SECONDS = 600.0
DEFAULT_WATCHDOG_CHECK_INTERVAL_SECONDS = 30.0


def status() -> dict[str, Any]:
    """Return safe long-connection readiness details."""
    creds = feishu_service.credentials()
    return {
        "connection_mode": "long_connection",
        "sdk_installed": _sdk_installed(),
        "app_id_configured": bool(creds["app_id"]),
        "app_secret_configured": bool(creds["app_secret"]),
        "verification_token_configured": bool(creds["verification_token"]),
        "encrypt_key_configured": bool(creds["encrypt_key"]),
        "real_send_enabled": os.getenv("FEISHU_ENABLE_REAL_SEND", "").lower() in {"1", "true", "yes"},
    }


def handle_ws_message_event(data: Any) -> dict[str, Any]:
    """Route a Feishu long-connection message event through Channel Runtime."""
    watchdog = _ACTIVE_WATCHDOG
    if watchdog is not None:
        watchdog.begin_processing()
    try:
        payload = normalize_ws_message_event(data)
        event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        message_id = str(message.get("message_id") or message.get("root_id") or "").strip()
        chat_id = str(message.get("chat_id") or "").strip()
        message_type = str(message.get("message_type") or "").strip()
        print(f"[Feishu WS] received message_id={message_id or '-'} chat_id={chat_id or '-'} type={message_type or '-'}")
        result = feishu_service.handle_event(payload, validate_token=False)
        reply = result.get("reply") if isinstance(result.get("reply"), dict) else {}
        print(
            "[Feishu WS] handled "
            f"status={result.get('status')} reason={result.get('reason') or '-'} "
            f"channel_id={result.get('channel_id') or '-'} reply_status={reply.get('status') or '-'}"
        )
        emit_event(
            "feishu.ws.event.handled",
            "feishu",
            "Handled Feishu long-connection message event.",
            session_id=str(result.get("channel_id") or ""),
            data={"status": result.get("status"), "chat_id": result.get("chat_id")},
        )
        return result
    finally:
        if watchdog is not None:
            watchdog.end_processing()


def normalize_ws_message_event(data: Any) -> dict[str, Any]:
    """Convert lark-oapi event models or plain dicts into webhook-like payloads."""
    plain = _to_plain_dict(data)
    event = plain.get("event") if isinstance(plain.get("event"), dict) else plain
    return {
        "schema": plain.get("schema") or "2.0",
        "header": plain.get("header") if isinstance(plain.get("header"), dict) else {"event_type": "im.message.receive_v1"},
        "event": event,
    }


def create_ws_client(lark_module: Any | None = None) -> Any:
    """Create a lark-oapi websocket client.

    The optional module argument exists for tests; production uses ``lark_oapi``.
    """
    creds = feishu_service.credentials()
    if not creds["app_id"] or not creds["app_secret"]:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required for Feishu long connection")
    lark = lark_module or _import_lark_oapi()
    builder = lark.EventDispatcherHandler.builder(creds["verification_token"], creds["encrypt_key"])
    handler = builder.register_p2_im_message_receive_v1(lambda data: handle_ws_message_event(data)).build()
    return lark.ws.Client(
        creds["app_id"],
        creds["app_secret"],
        event_handler=handler,
        log_level=_sdk_log_level(lark),
    )


class HangWatchdog:
    """Detects a Feishu WS worker that is alive (process running) but deaf.

    lark-oapi's own reconnect logic can die silently: an exception raised
    inside its internal ``_reconnect()``/``_disconnect()`` propagates out of
    an ``asyncio.create_task(...)``-scheduled coroutine that nothing ever
    awaits, so it is only ever surfaced (if at all) by asyncio's default
    "Task exception was never retrieved" handler -- trivially missed in
    container logs. After that happens the receive-loop task is simply gone
    and no further reconnection attempts are made, but the OS process and
    Docker container both look perfectly healthy (see
    docs/evaluations/ for the real incident this guards against: 2+ hours of
    total silence after a ping_timeout that followed a Feishu-side 502).

    Rather than trying to patch every possible way the SDK's reconnect can
    fail, this watchdog detects the *symptom*: no WS activity (an inbound
    frame -- data or the SDK's own periodic control-frame heartbeat -- or a
    reconnect lifecycle event) for longer than ``timeout_seconds``. When that
    fires, it exits the process so Docker's ``restart: unless-stopped`` can
    recover it, which is simpler and more robust than trying to resume the
    half-dead asyncio state in place.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        check_interval_seconds: float,
        heartbeat_path: Path | None,
        on_timeout: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._check_interval_seconds = check_interval_seconds
        self._heartbeat_path = heartbeat_path
        self._on_timeout = on_timeout or _default_on_timeout
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._last_activity = clock()
        self._last_heartbeat_write = 0.0
        self._processing_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def touch(self) -> None:
        """Record WS activity now. Safe to call from any thread."""
        with self._lock:
            self._last_activity = self._clock()
        self._maybe_write_heartbeat_file()

    def begin_processing(self) -> None:
        """Keep the connection watchdog healthy while a synchronous event is being handled."""
        with self._lock:
            self._processing_count += 1
            self._last_activity = self._clock()
        self._maybe_write_heartbeat_file()

    def end_processing(self) -> None:
        with self._lock:
            self._processing_count = max(0, self._processing_count - 1)
            self._last_activity = self._clock()
        self._maybe_write_heartbeat_file()

    def _touch_if_processing(self) -> bool:
        with self._lock:
            if self._processing_count <= 0:
                return False
            self._last_activity = self._clock()
        self._maybe_write_heartbeat_file()
        return True

    def seconds_since_last_activity(self) -> float:
        with self._lock:
            return self._clock() - self._last_activity

    def start(self) -> None:
        self.touch()
        self._thread = threading.Thread(target=self._run, name="feishu-ws-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self._check_interval_seconds):
            # The lark SDK invokes the event handler synchronously. A legitimate research request
            # can therefore occupy that callback longer than the connection timeout and prevent
            # raw ping frames from reaching our hook. Busy handling is not a dead WS connection.
            if self._touch_if_processing():
                continue
            stale_for = self.seconds_since_last_activity()
            if stale_for >= self._timeout_seconds:
                self._on_timeout(stale_for)
                return

    def _maybe_write_heartbeat_file(self) -> None:
        if self._heartbeat_path is None:
            return
        now = self._wall_clock()
        # Debounce: bursts of inbound frames shouldn't turn into a write storm.
        if now - self._last_heartbeat_write < 5.0:
            return
        self._last_heartbeat_write = now
        try:
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            self._heartbeat_path.write_text(str(now), encoding="utf-8")
        except OSError:
            pass  # best-effort; a heartbeat file write failure must not crash the worker


def _default_on_timeout(stale_for_seconds: float) -> None:
    message = (
        f"No Feishu WS activity for {stale_for_seconds:.0f}s -- treating the "
        "long connection as hung and exiting so Docker restarts the worker."
    )
    emit_event(
        "feishu.ws.watchdog.hang_detected",
        "feishu",
        message,
        level="error",
        data={"stale_for_seconds": stale_for_seconds},
    )
    print(f"[Feishu WS] watchdog: {message}", flush=True)
    os._exit(1)


_ACTIVE_WATCHDOG: HangWatchdog | None = None


def _touch_active_watchdog() -> None:
    watchdog = _ACTIVE_WATCHDOG
    if watchdog is not None:
        watchdog.touch()


def install_hang_watchdog(
    client: Any,
    *,
    timeout_seconds: float | None = None,
    check_interval_seconds: float | None = None,
    heartbeat_path: Path | None | object = _USE_DEFAULT_HEARTBEAT_PATH,
) -> HangWatchdog:
    """Wire a :class:`HangWatchdog` to ``client``'s reconnect hooks and raw
    frame handling, and register it as the module's active watchdog so
    :func:`handle_ws_message_event` also feeds it.

    ``timeout_seconds`` defaults to ``FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS`` env
    (600s / 5x lark-oapi's default 120s ping interval, leaving headroom for a
    couple of missed keepalive cycles before treating the connection as hung).
    Omitting ``heartbeat_path`` uses ``FEISHU_WS_HEARTBEAT_PATH`` or the local
    runtime default. Passing ``None`` explicitly disables heartbeat-file writes.
    """
    resolved_heartbeat_path = (
        DEFAULT_HEARTBEAT_PATH
        if heartbeat_path is _USE_DEFAULT_HEARTBEAT_PATH
        else heartbeat_path
    )
    watchdog = HangWatchdog(
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS", DEFAULT_WATCHDOG_TIMEOUT_SECONDS))
        ),
        check_interval_seconds=(
            check_interval_seconds
            if check_interval_seconds is not None
            else float(os.getenv("FEISHU_WS_WATCHDOG_CHECK_INTERVAL_SECONDS", DEFAULT_WATCHDOG_CHECK_INTERVAL_SECONDS))
        ),
        heartbeat_path=(
            resolved_heartbeat_path
            if isinstance(resolved_heartbeat_path, Path)
            else None
        ),
    )

    global _ACTIVE_WATCHDOG
    _ACTIVE_WATCHDOG = watchdog

    # Reconnect lifecycle hooks: the SDK's own signal that the connection was
    # lost and either recovered or is retrying. Wrapped (not replaced) in
    # case a future caller ever wires its own hook here too.
    original_on_reconnecting = getattr(client, "on_reconnecting", None)
    if callable(original_on_reconnecting):
        def _on_reconnecting(_original: Callable[[], None] = original_on_reconnecting) -> None:
            watchdog.touch()
            _original()

        client.on_reconnecting = _on_reconnecting

    original_on_reconnected = getattr(client, "on_reconnected", None)
    if callable(original_on_reconnected):
        def _on_reconnected(_original: Callable[[], None] = original_on_reconnected) -> None:
            watchdog.touch()
            _original()

        client.on_reconnected = _on_reconnected

    # Raw frame handling: the only reliable "the socket is actually alive"
    # signal, since it fires on every inbound frame -- lark-oapi's own
    # periodic control-frame ping/pong included -- not just on user messages,
    # which can be hours apart in a quiet channel.
    original_handle_message = getattr(client, "_handle_message", None)
    if callable(original_handle_message):
        async def _handle_message_with_touch(msg: Any, _original: Callable[[Any], Any] = original_handle_message) -> Any:
            watchdog.touch()
            return await _original(msg)

        client._handle_message = _handle_message_with_touch
    else:
        emit_event(
            "feishu.ws.watchdog.hook_missing",
            "feishu",
            "lark-oapi Client has no _handle_message method to hook -- the watchdog will only "
            "track reconnect events and inbound user messages, not raw WS/keepalive frame "
            "traffic. This likely means the installed lark-oapi version changed its internals; "
            "re-check this hook after upgrading lark-oapi.",
            level="warning",
        )

    return watchdog


def start_ws_worker() -> None:
    """Start the blocking Feishu long-connection worker."""
    client = create_ws_client()
    watchdog = install_hang_watchdog(client)
    emit_event(
        "feishu.ws.worker.start",
        "feishu",
        "Starting Feishu long-connection worker.",
        data=status(),
    )
    watchdog.start()
    client.start()


def _sdk_installed() -> bool:
    try:
        _import_lark_oapi()
        return True
    except ModuleNotFoundError:
        return False


def _import_lark_oapi() -> Any:
    try:
        import lark_oapi as lark  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lark-oapi is required for Feishu long connection. Install requirements.txt first."
        ) from exc
    return lark


def _sdk_log_level(lark: Any) -> Any:
    level = os.getenv("FEISHU_WS_LOG_LEVEL", "INFO").upper()
    log_level = getattr(lark, "LogLevel", None)
    if log_level is None:
        return None
    return getattr(log_level, level, getattr(log_level, "INFO", None))


def _to_plain_dict(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data

    try:
        import lark_oapi as lark  # type: ignore

        raw = lark.JSON.marshal(data)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    plain = _object_to_dict(data)
    return plain if isinstance(plain, dict) else {}


def _object_to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _object_to_dict(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_object_to_dict(item) for item in value]
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {str(k): _object_to_dict(v) for k, v in attrs.items() if not str(k).startswith("_")}
    return str(value)
