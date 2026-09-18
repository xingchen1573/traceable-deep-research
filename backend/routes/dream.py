"""Dream-related API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

try:
    from ... import memory as memory_store
    from ...main import final_message_text
    from ...observability import emit_event, new_run_id, set_observation_context
except ImportError:  # pragma: no cover
    import memory as memory_store
    from main import final_message_text
    from observability import emit_event, new_run_id, set_observation_context

from ..dependencies import get_review_agent
from ..helpers import (
    error_response,
    json_from_text,
    response_json,
)
from ..models import DreamReviewRequest

router = APIRouter()


def _apply_review_result(review_json: dict[str, Any]) -> tuple[list[str], bool, str]:
    """Extract review results from the review agent's JSON output."""
    applied_topics = review_json.get("applied_topics") or []
    if not isinstance(applied_topics, list):
        applied_topics = []
    normalized_topics = [str(item) for item in applied_topics if str(item).strip()]
    refreshed_index = bool(review_json.get("refreshed_index"))
    review_summary = str(review_json.get("review_summary") or "Review agent completed.")
    return normalized_topics[:20], refreshed_index, review_summary[:500]


def _topic_snapshot() -> dict[str, str]:
    """Take a snapshot of all topic contents for comparison."""
    snapshot: dict[str, str] = {}
    for record in memory_store.list_memory_topics():
        snapshot[record.topic_id] = memory_store.load_memory_topic(record.topic_id)
    return snapshot


def _changed_topics(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Find topics that changed between two snapshots."""
    topic_ids = sorted(set(before) | set(after))
    return [topic_id for topic_id in topic_ids if before.get(topic_id) != after.get(topic_id)]


def _mark_dream_applying() -> dict[str, Any]:
    """Mark dream state as applying."""
    state = memory_store.read_dream_state()
    state["status"] = "applying"
    return memory_store.write_dream_state(state)


def _review_prompt(decision: str, note: str | None = None) -> str:
    """Build the prompt for the review agent."""
    reviewer_note = (note or "").strip()
    return (
        "Review and apply the current dream report to formal memory.\n\n"
        f"Reviewer decision: {decision}\n"
        f"Reviewer note: {reviewer_note or 'None'}\n\n"
        "Workflow:\n"
        "1. Read the dream report.\n"
        "2. Read MEMORY.md and current topics.\n"
        "3. Update or create only the durable topics justified by the report.\n"
        "4. Do not delete topics.\n"
        "5. Refresh MEMORY.md if any formal memory changed.\n"
        "6. Return only the required JSON summary.\n"
    )


@router.get("/dream/status")
def dream_status() -> dict[str, Any]:
    """Get the current dream state."""
    request_id = new_run_id()
    return response_json(memory_store.read_dream_state(), request_id=request_id)


@router.get("/dream/report")
def dream_report() -> dict[str, Any]:
    """Get the current dream report content."""
    from pathlib import Path

    request_id = new_run_id()
    root = Path(__file__).resolve().parent.parent.parent
    report_path = root / "memory" / "dream_report.md"
    return response_json(
        {
            "path": "memory/dream_report.md",
            "exists": report_path.exists(),
            "content": report_path.read_text(encoding="utf-8") if report_path.exists() else "",
            "dream_state": memory_store.read_dream_state(),
        },
        request_id=request_id,
    )


@router.post("/dream/run")
def dream_run() -> dict[str, Any]:
    """Run the AutoDream process."""
    state = memory_store.read_dream_state()
    request_id = new_run_id()
    if state.get("status") == "pending_review":
        emit_event(
            "dream.skipped.pending_review",
            "server",
            "Skipped AutoDream because a report is pending review.",
            run_id=request_id,
            level="warning",
            data={"report_path": state.get("report_path")},
        )
        return response_json(
            {"status": "skipped", "reason": "pending_review", "dream_state": state},
            request_id=request_id,
        )
    set_observation_context(session_id="dream-maintenance", run_id=request_id)
    try:
        from ..dependencies import build_dream_model

        report = memory_store.run_autodream(dream_model=build_dream_model())
        return response_json(
            {"status": "generated", "report_path": report, "dream_state": memory_store.read_dream_state()},
            request_id=request_id,
        )
    except Exception as exc:
        return error_response(
            exc,
            default_message="The dream run failed.",
            session_id="dream-maintenance",
            event_type="dream.run.failed",
            request_id=request_id,
        )
    finally:
        set_observation_context(session_id="dream-maintenance", run_id=None)


@router.post("/dream/tick")
def dream_tick() -> dict[str, Any]:
    """Check and run AutoDream if conditions are met."""
    should_run, reason, latest_log = memory_store.should_run_autodream_tick()
    request_id = new_run_id()
    if not should_run:
        emit_event(
            "dream.tick.skipped",
            "server",
            "Skipped dream tick.",
            run_id=request_id,
            level="info",
            data={"reason": reason, "latest_log": latest_log},
        )
        return response_json(
            {
                "status": "skipped",
                "reason": reason,
                "latest_log": latest_log,
                "dream_state": memory_store.read_dream_state(),
            },
            request_id=request_id,
        )
    set_observation_context(session_id="dream-maintenance", run_id=request_id)
    try:
        from ..dependencies import build_dream_model

        report = memory_store.run_autodream(dream_model=build_dream_model())
        return response_json(
            {
                "status": "generated",
                "reason": "ready",
                "latest_log": latest_log,
                "report_path": report,
                "dream_state": memory_store.read_dream_state(),
            },
            request_id=request_id,
        )
    except Exception as exc:
        return error_response(
            exc,
            default_message="The dream tick failed.",
            session_id="dream-maintenance",
            event_type="dream.tick.failed",
            request_id=request_id,
        )
    finally:
        set_observation_context(session_id="dream-maintenance", run_id=None)


@router.post("/dream/review")
def dream_review(request: DreamReviewRequest) -> dict[str, Any]:
    """Review and apply a dream report."""
    decision = request.decision.strip().lower()
    if decision not in {"approve", "edit", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be approve, edit, or reject")
    state = memory_store.read_dream_state()
    request_id = new_run_id()
    if state.get("status") != "pending_review":
        raise HTTPException(status_code=400, detail="No pending dream review exists")
    if decision == "reject":
        updated = memory_store.close_dream_review(decision=decision, note=request.note)
        return response_json(
            {
                "status": "closed",
                "dream_state": updated,
                "applied_topics": [],
                "refreshed_index": False,
                "review_summary": "Dream review was rejected; no formal memory changes were applied.",
            },
            request_id=request_id,
        )

    previous_state = dict(state)
    review_agent = get_review_agent()
    topics_before = _topic_snapshot()
    index_before = memory_store.read_memory_index()
    set_observation_context(session_id="dream-review", run_id=request_id)
    emit_event(
        "dream.apply.started",
        "server",
        "Dream review apply started.",
        session_id="dream-review",
        run_id=request_id,
        data={"decision": decision, "has_note": bool(request.note)},
    )
    _mark_dream_applying()
    try:
        result = review_agent.invoke({"messages": [{"role": "user", "content": _review_prompt(decision, request.note)}]})
        review_json = json_from_text(final_message_text(result))
        _, _, review_summary = _apply_review_result(review_json)
        topics_after = _topic_snapshot()
        index_after = memory_store.read_memory_index()
        applied_topics = _changed_topics(topics_before, topics_after)
        refreshed_index = index_before != index_after
        updated = memory_store.close_dream_review(decision=decision, note=request.note)
        emit_event(
            "dream.apply.finished",
            "server",
            "Dream review apply finished.",
            session_id="dream-review",
            run_id=request_id,
            data={"applied_topics": applied_topics, "refreshed_index": refreshed_index},
        )
        return response_json(
            {
                "status": "closed",
                "dream_state": updated,
                "applied_topics": applied_topics,
                "refreshed_index": refreshed_index,
                "review_summary": review_summary,
            },
            request_id=request_id,
        )
    except Exception as exc:
        memory_store.write_dream_state(previous_state)
        return error_response(
            exc,
            default_message="The dream review apply failed.",
            session_id="dream-review",
            event_type="dream.apply.failed",
            request_id=request_id,
        )
    finally:
        set_observation_context(session_id="dream-review", run_id=None)
