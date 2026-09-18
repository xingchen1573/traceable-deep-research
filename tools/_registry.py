"""Internal registry and helper functions for tools."""

from __future__ import annotations

import re
from pathlib import Path

try:
    from .. import memory as memory_store
except ImportError:  # pragma: no cover - script mode fallback
    import memory as memory_store


ROOT = Path(__file__).resolve().parent.parent
DEMO_WORKSPACE = ROOT / "demo_workspace"
SKILLS_DIR = ROOT / "skills"
KNOWLEDGE_BASE = ROOT / "knowledge_base"
LIGHTRAG_WORKSPACES = ROOT / "lightrag_workspaces"

_SURFACED_REGISTRY: memory_store.SurfacedMemoryRegistry | None = None
_CURRENT_THREAD_ID = "default"

# Matches the thread_id shape produced by backend/services/channel_service.py::thread_id_for,
# e.g. "channel:feishu:oc_xxx:main" -> real channel_id "feishu:oc_xxx". CLI and bare-web
# thread_ids don't match this and are used verbatim as the conversation's tagging key.
CHANNEL_THREAD_RE = re.compile(r"^channel:(.+):(main|kb_qa)$")


def conversation_channel_id(thread_id: str | None = None) -> str:
    """Derive the job-tagging channel_id for the current (or given) conversation thread.

    For channel-routed conversations (Web channel / Feishu), this recovers the real
    channel_id (e.g. "feishu:oc_xxx") embedded in the thread_id. For CLI and bare Web chat
    (no channel concept), the thread_id itself is used verbatim as a stable per-conversation
    key. Shared by tools/research_jobs.py (job tagging) and tools/knowledge_manager.py
    (channel <-> knowledge-base-topic binding).
    """
    tid = thread_id if thread_id is not None else current_thread_id()
    match = CHANNEL_THREAD_RE.match(tid)
    if match:
        return match.group(1)
    return tid


def configure_surfaced_registry(registry: memory_store.SurfacedMemoryRegistry) -> None:
    """Share the middleware surfaced-memory registry with memory tools."""
    global _SURFACED_REGISTRY
    _SURFACED_REGISTRY = registry


def set_current_thread_id(thread_id: str) -> None:
    """Set the CLI session id used when tools mark surfaced memory."""
    global _CURRENT_THREAD_ID
    _CURRENT_THREAD_ID = thread_id or "default"


def mark_memory_surfaced(topic_id: str) -> None:
    """Mark a memory topic as surfaced in the current session."""
    if _SURFACED_REGISTRY is not None:
        _SURFACED_REGISTRY.mark(_CURRENT_THREAD_ID, topic_id)


def current_thread_id() -> str:
    """Get the current thread id."""
    return _CURRENT_THREAD_ID


def ensure_demo_files() -> None:
    """Seed a tiny local workspace and skills library."""
    DEMO_WORKSPACE.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    ensure_knowledge_base()

    sample = DEMO_WORKSPACE / "sample_project.md"
    if not sample.exists():
        sample.write_text(
            "# Sample Project\n\n"
            "This tiny workspace exists so Deep Research can safely read and edit files.\n"
            "It is deliberately separate from the real repository.\n",
            encoding="utf-8",
        )

    repo_skill = SKILLS_DIR / "repo_reader.md"
    if not repo_skill.exists():
        repo_skill.write_text(
            "---\n"
            "name: repo_reader\n"
            "summary: Inspect local files conservatively and cite paths.\n"
            "---\n\n"
            "# Repo Reader Skill\n\n"
            "- Read only the files needed for the question.\n"
            "- Prefer concise findings with file-path evidence.\n"
            "- Treat memory as a hint and verify local facts from files.\n",
            encoding="utf-8",
        )

    web_skill = SKILLS_DIR / "web_reader.md"
    if not web_skill.exists():
        web_skill.write_text(
            "---\n"
            "name: web_reader\n"
            "summary: Fetch explicit URLs and summarize source-aware facts.\n"
            "---\n\n"
            "# Web Reader Skill\n\n"
            "- Use explicit URLs only.\n"
            "- Keep source uncertainty visible.\n"
            "- Do not flood the main context with raw page content.\n",
            encoding="utf-8",
        )


def ensure_knowledge_base() -> None:
    """Create the knowledge_base directory if it doesn't exist."""
    KNOWLEDGE_BASE.mkdir(parents=True, exist_ok=True)


def ensure_lightrag_workspaces() -> None:
    """Create the LightRAG workspace root if it doesn't exist."""
    LIGHTRAG_WORKSPACES.mkdir(parents=True, exist_ok=True)


def _normalize_relative(path: str, *, strip_prefix: str | None = None) -> str:
    """Normalize a user/model-supplied relative path.

    Converts backslashes, drops leading "./" segments, and optionally strips a
    redundant leading directory prefix (e.g. a model passing "demo_workspace/x"
    when the base is already demo_workspace).
    """
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    if strip_prefix:
        prefix = strip_prefix.rstrip("/") + "/"
        while normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def safe_workspace_path(path: str) -> Path:
    """Resolve a path within the demo workspace, raising if it escapes."""
    DEMO_WORKSPACE.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_relative(path, strip_prefix=DEMO_WORKSPACE.name)
    full = (DEMO_WORKSPACE / normalized).resolve()
    workspace = DEMO_WORKSPACE.resolve()
    if full != workspace and workspace not in full.parents:
        raise ValueError("Path escapes demo_workspace")
    return full


def safe_read_path(path: str) -> Path:
    """Resolve a path within the MVP directory, raising if it escapes."""
    normalized = _normalize_relative(path)
    full = (ROOT / normalized).resolve()
    if full != ROOT.resolve() and ROOT.resolve() not in full.parents:
        raise ValueError("Path escapes repository root")
    return full


def validate_topic_shape(topic: str) -> list[str]:
    """Validate a (slash-stripped) topic string as 1 or 2 path segments.

    Shared by tools/knowledge.py (write path) and tools/knowledge_qa.py /
    tools/knowledge_manager.py (read paths) to enforce the same "theme/subtopic" nesting limit
    (docs/VISION.md 第1节) everywhere a topic string is parsed. Callers pass an already
    slash-stripped string; this only checks segment count/emptiness, not existence.
    """
    raw_segments = topic.split("/")
    segments = [seg for seg in raw_segments if seg]
    if not segments or len(segments) != len(raw_segments) or len(segments) > 2:
        raise ValueError("topic must be a single directory name or 'theme/subtopic' (at most one '/')")
    return segments


def safe_knowledge_path(path: str) -> Path:
    """Resolve a path within the knowledge_base directory, raising if it escapes."""
    KNOWLEDGE_BASE.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_relative(path, strip_prefix=KNOWLEDGE_BASE.name)
    full = (KNOWLEDGE_BASE / normalized).resolve()
    base = KNOWLEDGE_BASE.resolve()
    if full != base and base not in full.parents:
        raise ValueError("Path escapes knowledge_base")
    return full


def discover_topic_dirs(base: Path) -> list[tuple[str, Path]]:
    """List (topic, dir) pairs directly under `base`, one level deeper for theme containers.

    docs/VISION.md 第1节: a knowledge_base child directory is either a flat topic (has .md files
    directly inside -- the legacy shape, e.g. the 3 pre-existing knowledge bases) or a theme
    container (has no .md of its own but has subdirectories that do -- each such subdirectory is
    reported as "theme/subtopic"). A theme container's own root-level docs (e.g. a theme-level
    "00_主题总览.md") are intentionally NOT reported as a separate flat topic here, to avoid
    double-counting the same directory both as a topic and as a container of topics.
    """
    if not base.exists():
        return []
    results: list[tuple[str, Path]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        subtopic_dirs = [
            sub for sub in sorted(child.iterdir()) if sub.is_dir() and list(sub.glob("*.md"))
        ]
        if subtopic_dirs:
            results.extend((f"{child.name}/{sub.name}", sub) for sub in subtopic_dirs)
            continue
        if list(child.glob("*.md")):
            results.append((child.name, child))
    return results


def skill_summaries() -> str:
    """Return short summaries of all available skills."""
    ensure_demo_files()
    lines = []
    for path in sorted(SKILLS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        name = path.stem
        summary = "No summary."
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                for line in parts[1].splitlines():
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("summary:"):
                        summary = line.split(":", 1)[1].strip()
        lines.append(f"- {name}: {summary}")
    return "\n".join(lines) if lines else "- No skills available."
