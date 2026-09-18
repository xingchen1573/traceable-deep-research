"""Feishu Docx delivery adapter for research review artifacts."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from . import feishu_service


DEFAULT_DOC_URL_BASE = "https://www.feishu.cn/docx"
MAX_BLOCK_TEXT_CHARS = 1800


def create_research_review_doc(
    *,
    review_path: str | Path,
    title: str,
    folder_token: str | None = None,
) -> dict[str, Any]:
    """Create a Feishu Docx document and append the Markdown review content."""
    path = Path(review_path)
    markdown = path.read_text(encoding="utf-8")
    document = create_document(title=title, folder_token=folder_token)
    document_id = str(document["document_id"])
    try:
        append_markdown(document_id=document_id, markdown=markdown)
    except Exception:
        append_markdown(document_id=document_id, markdown=markdown, text_only=True)
    return {
        "type": "feishu_doc",
        "document_id": document_id,
        "url": _document_url(document_id),
        "title": title,
        "source_path": str(path),
    }


def create_document(*, title: str, folder_token: str | None = None) -> dict[str, Any]:
    lark = _lark()
    client = _client(lark)
    from lark_oapi.api.docx.v1.model.create_document_request import CreateDocumentRequest
    from lark_oapi.api.docx.v1.model.create_document_request_body import CreateDocumentRequestBody

    body_builder = CreateDocumentRequestBody.builder().title(_trim_title(title))
    if folder_token:
        body_builder = body_builder.folder_token(folder_token)
    request = CreateDocumentRequest.builder().request_body(body_builder.build()).build()
    response = client.docx.v1.document.create(request)
    data = _response_data(response, "Feishu create document failed")
    document = data.get("document") if isinstance(data, dict) else None
    document_id = str((document or {}).get("document_id") or "").strip()
    if not document_id:
        raise RuntimeError("Feishu create document response did not include document_id")
    return {
        "document_id": document_id,
        "revision_id": (document or {}).get("revision_id"),
        "title": (document or {}).get("title") or title,
    }


def append_markdown(*, document_id: str, markdown: str, text_only: bool = False) -> dict[str, Any]:
    blocks = markdown_to_docx_blocks(markdown, text_only=text_only)
    if not blocks:
        return {"document_id": document_id, "created_blocks": 0}
    created = 0
    batch_size = _block_batch_size()
    for index in range(0, len(blocks), batch_size):
        batch = blocks[index : index + batch_size]
        _append_blocks(document_id=document_id, blocks=batch, index=index)
        created += len(batch)
    return {"document_id": document_id, "created_blocks": created}


def markdown_to_docx_blocks(markdown: str, *, text_only: bool = False) -> list[Any]:
    blocks: list[Any] = []
    in_code = False
    code_buffer: list[str] = []
    table_buffer: list[str] = []
    lines = str(markdown or "").replace("\r\n", "\n").split("\n")
    for raw in lines:
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if table_buffer:
                blocks.extend(_line_blocks("\n".join(table_buffer), kind="code" if not text_only else "text"))
                table_buffer = []
            if in_code and code_buffer:
                blocks.extend(_line_blocks("\n".join(code_buffer), kind="code" if not text_only else "text"))
                code_buffer = []
            in_code = not in_code
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if _is_markdown_table_line(line):
            table_buffer.append(line.strip())
            continue
        if table_buffer:
            blocks.extend(_line_blocks("\n".join(table_buffer), kind="code" if not text_only else "text"))
            table_buffer = []
        if not line.strip():
            continue
        kind, content = _markdown_line_kind(line, text_only=text_only)
        blocks.extend(_line_blocks(content, kind=kind))
    if table_buffer:
        blocks.extend(_line_blocks("\n".join(table_buffer), kind="code" if not text_only else "text"))
    if code_buffer:
        blocks.extend(_line_blocks("\n".join(code_buffer), kind="code" if not text_only else "text"))
    return blocks


def _append_blocks(*, document_id: str, blocks: list[Any], index: int) -> None:
    lark = _lark()
    client = _client(lark)
    from lark_oapi.api.docx.v1.model.create_document_block_children_request import CreateDocumentBlockChildrenRequest
    from lark_oapi.api.docx.v1.model.create_document_block_children_request_body import CreateDocumentBlockChildrenRequestBody

    body = CreateDocumentBlockChildrenRequestBody.builder().index(index).children(blocks).build()
    request = (
        CreateDocumentBlockChildrenRequest.builder()
        .document_id(document_id)
        .block_id(document_id)
        .document_revision_id(-1)
        .request_body(body)
        .build()
    )
    response = client.docx.v1.document_block_children.create(request)
    _response_data(response, "Feishu append document blocks failed")


def _line_blocks(content: str, *, kind: str) -> list[Any]:
    chunks = _chunk_text(content, MAX_BLOCK_TEXT_CHARS)
    return [_build_block(chunk, kind=kind) for chunk in chunks if chunk.strip()]


def _build_block(content: str, *, kind: str) -> Any:
    from lark_oapi.api.docx.v1.model.block import Block

    text = _text(content)
    builder = Block.builder()
    if kind == "heading1":
        return builder.block_type(3).heading1(text).build()
    if kind == "heading2":
        return builder.block_type(4).heading2(text).build()
    if kind == "heading3":
        return builder.block_type(5).heading3(text).build()
    if kind == "bullet":
        return builder.block_type(12).bullet(text).build()
    if kind == "ordered":
        return builder.block_type(13).ordered(text).build()
    if kind == "code":
        return builder.block_type(14).code(text).build()
    if kind == "quote":
        return builder.block_type(15).quote(text).build()
    return builder.block_type(2).text(text).build()


def _text(content: str) -> Any:
    from lark_oapi.api.docx.v1.model.text import Text
    from lark_oapi.api.docx.v1.model.text_element import TextElement
    from lark_oapi.api.docx.v1.model.text_run import TextRun

    run = TextRun.builder().content(content).build()
    element = TextElement.builder().text_run(run).build()
    return Text.builder().elements([element]).build()


def _markdown_line_kind(line: str, *, text_only: bool) -> tuple[str, str]:
    clean = line.strip()
    if text_only:
        return "text", _clean_inline_markdown(clean)
    heading = re.match(r"^(#{1,6})\s+(.+)$", clean)
    if heading:
        level = min(len(heading.group(1)), 3)
        return f"heading{level}", _clean_inline_markdown(heading.group(2))
    bullet = re.match(r"^[-*+]\s+(.+)$", clean)
    if bullet:
        return "bullet", _clean_inline_markdown(bullet.group(1))
    ordered = re.match(r"^\d+[.)]\s+(.+)$", clean)
    if ordered:
        return "ordered", _clean_inline_markdown(ordered.group(1))
    quote = re.match(r"^>\s*(.+)$", clean)
    if quote:
        return "quote", _clean_inline_markdown(quote.group(1))
    return "text", _clean_inline_markdown(clean)


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    return stripped.count("|") >= 2


def _clean_inline_markdown(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"\1", value)
    value = re.sub(r"__(.+?)__", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    return value.strip()


def _chunk_text(text: str, limit: int) -> list[str]:
    value = str(text or "").strip()
    if len(value) <= limit:
        return [value]
    chunks: list[str] = []
    while value:
        chunks.append(value[:limit])
        value = value[limit:]
    return chunks


def _response_data(response: Any, error_prefix: str) -> dict[str, Any]:
    raw = getattr(response, "raw", None)
    payload: dict[str, Any] = {}
    if raw is not None:
        content = getattr(raw, "content", b"")
        if isinstance(content, bytes):
            payload = json.loads(content.decode("utf-8") or "{}")
    success = bool(response.success()) if hasattr(response, "success") else int(payload.get("code") or 0) == 0
    if not success or int(payload.get("code") or 0) != 0:
        code = getattr(response, "code", None) or payload.get("code")
        msg = getattr(response, "msg", None) or payload.get("msg") or payload
        raise RuntimeError(f"{error_prefix}: code={code}, msg={msg}")
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _client(lark: Any) -> Any:
    creds = feishu_service.credentials()
    if not creds["app_id"] or not creds["app_secret"]:
        raise RuntimeError("FEISHU_APP_ID and FEISHU_APP_SECRET are required for Feishu Doc delivery")
    return (
        lark.Client.builder()
        .app_id(creds["app_id"])
        .app_secret(creds["app_secret"])
        .log_level(lark.LogLevel.ERROR)
        .build()
    )


def _lark() -> Any:
    try:
        import lark_oapi as lark  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("lark_oapi is required for Feishu Doc delivery") from exc
    return lark


def _document_url(document_id: str) -> str:
    base = os.getenv("FEISHU_DOC_URL_BASE", DEFAULT_DOC_URL_BASE).strip() or DEFAULT_DOC_URL_BASE
    return f"{base.rstrip('/')}/{document_id}"


def _trim_title(title: str) -> str:
    value = re.sub(r"\s+", " ", str(title or "")).strip()
    return value[:120] or f"Research Review {int(time.time())}"


def _block_batch_size() -> int:
    try:
        value = int(os.getenv("FEISHU_DOC_BLOCK_BATCH_SIZE", "40"))
    except ValueError:
        return 40
    return max(1, min(value, 50))
