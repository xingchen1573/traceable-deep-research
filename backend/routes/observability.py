"""Observability-related API routes."""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse, StreamingResponse

try:
    from ...observability import (
        replay_events,
        subscribe_events,
        unsubscribe_events,
    )
except ImportError:  # pragma: no cover
    from observability import (
        replay_events,
        subscribe_events,
        unsubscribe_events,
    )

router = APIRouter()


@router.get("/observability/stream")
async def observability_stream(
    session_id: str | None = Query(default=None),
    run_id: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    replay: int = Query(default=20, ge=0, le=200),
) -> StreamingResponse:
    """SSE endpoint for real-time observability events.

    run_id/job_id narrow the stream to one run's or one job's events — used by
    the "进行中" run/job detail timeline to append new steps live (see
    docs/modules/observability-redesign-v0.1.md 3.9). replay only honors
    session_id (today's log only); run_id/job_id filtering applies to the
    live tail, not the replay backlog.
    """
    def _matches(payload: dict[str, Any]) -> bool:
        if run_id and str(payload.get("run_id")) != run_id:
            return False
        if job_id and str(payload.get("job_id")) != job_id:
            return False
        return True

    async def event_generator() -> Any:
        for payload in replay_events(limit=replay, session_id=session_id):
            if not _matches(payload):
                continue
            yield f"event: observability\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        subscriber_id, event_queue = subscribe_events()
        try:
            while True:
                try:
                    payload = await asyncio.to_thread(event_queue.get, True, 15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if session_id and str(payload.get("session_id")) != session_id:
                    continue
                if not _matches(payload):
                    continue
                yield f"event: observability\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            unsubscribe_events(subscriber_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/debug/observability")
def debug_observability() -> HTMLResponse:
    """Debug dashboard for observability events."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Harness Observability</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }
    .bar { padding: 12px 16px; background: #1f2937; position: sticky; top: 0; display: flex; gap: 8px; align-items: center; }
    input, button { border-radius: 6px; border: 1px solid #374151; background: #0f172a; color: #e5e7eb; padding: 8px 10px; }
    button { cursor: pointer; background: #2563eb; border-color: #2563eb; }
    #events { padding: 16px; display: grid; gap: 10px; }
    .event { border: 1px solid #374151; border-radius: 8px; padding: 10px 12px; background: #0b1220; }
    .event.dream { border-color: #0ea5e9; }
    .event.hitl { border-color: #f59e0b; }
    .event.guard { border-color: #ef4444; }
    .event.worker { border-color: #22c55e; }
    .meta { font-size: 12px; opacity: 0.8; margin-bottom: 6px; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; font-size: 12px; }
  </style>
</head>
<body>
  <div class="bar">
    <label>Session</label>
    <input id="session" placeholder="optional thread/session id" />
    <label>Replay</label>
    <input id="replay" type="number" value="20" min="0" max="200" />
    <button id="connect">Connect</button>
    <span id="status">idle</span>
  </div>
  <div id="events"></div>
  <script>
    const eventsEl = document.getElementById('events');
    const statusEl = document.getElementById('status');
    let source = null;
    function cssClass(eventType) {
      if (eventType.startsWith('dream.')) return 'dream';
      if (eventType.startsWith('hitl.') || eventType.startsWith('chat.interrupted')) return 'hitl';
      if (eventType.startsWith('guard.')) return 'guard';
      if (eventType.startsWith('worker.')) return 'worker';
      return '';
    }
    function renderEvent(payload) {
      const card = document.createElement('div');
      card.className = 'event ' + cssClass(payload.event_type || '');
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = [payload.timestamp, payload.session_id, payload.run_id || '-', payload.event_type, payload.level].join(' | ');
      const msg = document.createElement('div');
      msg.textContent = payload.message || '';
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(payload.data || {}, null, 2);
      card.appendChild(meta);
      card.appendChild(msg);
      card.appendChild(pre);
      eventsEl.prepend(card);
      while (eventsEl.childElementCount > 200) eventsEl.removeChild(eventsEl.lastChild);
    }
    function connect() {
      if (source) source.close();
      eventsEl.innerHTML = '';
      const session = encodeURIComponent(document.getElementById('session').value.trim());
      const replay = encodeURIComponent(document.getElementById('replay').value || '20');
      const qs = `?replay=${replay}${session ? `&session_id=${session}` : ''}`;
      source = new EventSource('/observability/stream' + qs);
      statusEl.textContent = 'connecting';
      source.onopen = () => { statusEl.textContent = 'connected'; };
      source.onerror = () => { statusEl.textContent = 'reconnecting'; };
      source.addEventListener('observability', (event) => {
        try { renderEvent(JSON.parse(event.data)); } catch {}
      });
    }
    document.getElementById('connect').addEventListener('click', connect);
  </script>
</body>
</html>"""
    return HTMLResponse(html)
