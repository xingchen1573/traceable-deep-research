"""Skills-related tools."""

from __future__ import annotations

from langchain_core.tools import tool

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from ._registry import SKILLS_DIR, ensure_demo_files, skill_summaries


@tool
def list_skills() -> str:
    """List available skill names and short summaries."""
    return skill_summaries()


@tool
def load_skill(skill_name: str) -> str:
    """Load the full body of a skill by name."""
    ensure_demo_files()
    safe = "".join(ch for ch in skill_name if ch.isalnum() or ch in "-_")
    path = SKILLS_DIR / f"{safe}.md"
    if not path.exists():
        emit_event(
            "skill.loaded",
            "tools",
            "Skill load failed.",
            level="warning",
            data={"skill_name": safe, "status": "not_found"},
        )
        return f"Skill not found: {skill_name}"
    content = path.read_text(encoding="utf-8")
    emit_event(
        "skill.loaded",
        "tools",
        "Loaded full skill body.",
        data={"skill_name": safe, "chars": len(content)},
    )
    return content
