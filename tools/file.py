"""File-related tools."""

from __future__ import annotations

from langchain_core.tools import tool

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from ._registry import ROOT, ensure_demo_files, safe_read_path, safe_workspace_path


@tool
def list_files(root: str = ".") -> str:
    """列出当前仓库根目录下的文件，供 repo-worker 只读探索使用。"""
    try:
        base = safe_read_path(root)
    except ValueError as exc:
        return f"Denied: {exc}"
    if not base.exists():
        return f"Not found: {root}"
    paths = []
    for item in sorted(base.rglob("*") if base.is_dir() else [base]):
        if item.is_file() and "__pycache__" not in item.parts:
            paths.append(str(item.relative_to(ROOT)))
    return "\n".join(paths[:200]) or "No files found."


@tool
def read_file(path: str) -> str:
    """读取当前仓库根目录范围内的文件内容（只读）。

    用于在回答或落库前确认已有文件内容。返回文件文本（最多 12000 字符）。
    修改任何已有文件前，应先用本工具确认其当前内容。
    """
    return _read_file(path)


@tool
def write_file(path: str, content: str, mode: str = "write") -> str:
    """写入 demo_workspace 范围内的文件（需 HITL 审批）。

    mode:
    - "write": 覆盖写入，content 必须是完整的新内容
    - "append": 在文件末尾追加 content

    使用规则：
    - 修改已有文件前，必须先用 read_file 确认文件内容
    - "write" 会覆盖原有内容，请确保 content 包含完整新内容
    """
    if mode == "write":
        return _write_file(path, content, overwrite=True)
    elif mode == "append":
        return _write_file(path, content, overwrite=False)
    else:
        return f"Denied: 不支持的 mode '{mode}'，请使用 write/append"


def _read_file(path: str) -> str:
    """读取当前仓库根目录范围内的文件。"""
    try:
        full = safe_read_path(path)
    except ValueError as exc:
        return f"Denied: {exc}"
    if not full.exists() or not full.is_file():
        return f"Not found: {path}"
    text = full.read_text(encoding="utf-8", errors="replace")
    return text[:12000]


def _write_file(path: str, content: str, overwrite: bool = True) -> str:
    """写入 demo_workspace 范围内的文件。"""
    try:
        full = safe_workspace_path(path)
    except ValueError as exc:
        emit_event(
            "tool.call.failed",
            "tools",
            "write_file denied path outside demo workspace.",
            level="warning",
            data={"tool": "write_file", "path": path},
        )
        return f"Denied: {exc}"

    full.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        full.write_text(content, encoding="utf-8")
        action = "wrote"
    else:
        with open(full, "a", encoding="utf-8") as f:
            f.write(content)
        action = "appended to"

    emit_event(
        "tool.call.finished",
        "tools",
        f"write_file {action} a demo workspace file.",
        data={
            "tool": "write_file",
            "path": str(full.relative_to(ROOT)),
            "chars": len(content),
            "operation": "write" if overwrite else "append",
        },
    )
    return f"{action.capitalize()} {full.relative_to(ROOT)} ({len(content)} chars)."
