"""Run the local Feishu long-connection worker.

This mode does not require a public callback URL. The process connects out to
Feishu, receives message events, and routes them into the shared Channel Runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    lock_file = _acquire_single_instance_lock()
    if lock_file is None:
        print("Another Feishu long-connection worker is already running. Stop it before starting a new one.")
        return 3

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)

    from backend.services import feishu_ws_service
    from backend.services.runtime_state_service import start_runtime_monitor

    readiness = feishu_ws_service.status()
    print("Feishu long-connection worker readiness:")
    for key, value in readiness.items():
        print(f"- {key}: {value}")
    if not readiness["sdk_installed"]:
        print("lark-oapi is not installed. Run: ./.venv/bin/pip install -r requirements.txt")
        return 2
    if not readiness["app_id_configured"] or not readiness["app_secret_configured"]:
        print("FEISHU_APP_ID and FEISHU_APP_SECRET are required.")
        return 2

    runtime_monitor = start_runtime_monitor(role="feishu-worker", reconcile=False)
    print("Starting worker. Press Ctrl+C to stop.")
    try:
        feishu_ws_service.start_ws_worker()
    except KeyboardInterrupt:
        print("\nFeishu long-connection worker stopped.")
    finally:
        runtime_monitor.stop()
    return 0


def _acquire_single_instance_lock():
    lock_path = ROOT / "data" / "feishu_ws_worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w+")
    try:
        if sys.platform == "win32":
            lock_file.write(str(lock_path) + "\n")
            lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.write(str(lock_path) + "\n")
            lock_file.flush()
    except OSError:
        lock_file.close()
        return None
    return lock_file


if __name__ == "__main__":
    raise SystemExit(main())
