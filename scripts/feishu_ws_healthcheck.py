"""Docker HEALTHCHECK probe for the Feishu long-connection worker.

Checks that backend/services/feishu_ws_service.py's HangWatchdog wrote a
heartbeat within the last FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS (default 600s) --
this is the only externally-observable signal that the process is not just
running (Docker already knows that) but actually receiving WS traffic.

Exit 0 = healthy, exit 1 = unhealthy (missing/stale heartbeat file).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_configured_heartbeat_path = os.getenv("FEISHU_WS_HEARTBEAT_PATH", "").strip()
HEARTBEAT_PATH = (
    Path(_configured_heartbeat_path).expanduser()
    if _configured_heartbeat_path
    else ROOT / "data" / "feishu_ws_heartbeat.txt"
)


def main() -> int:
    timeout_seconds = float(os.getenv("FEISHU_WS_WATCHDOG_TIMEOUT_SECONDS", "600"))
    if not HEARTBEAT_PATH.exists():
        print(f"unhealthy: heartbeat file missing ({HEARTBEAT_PATH})")
        return 1
    age = time.time() - HEARTBEAT_PATH.stat().st_mtime
    if age > timeout_seconds:
        print(f"unhealthy: heartbeat is {age:.0f}s old (limit {timeout_seconds:.0f}s)")
        return 1
    print(f"healthy: heartbeat is {age:.0f}s old")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
