"""Knowledge-base service functions."""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

try:
    from ...tools._registry import KNOWLEDGE_BASE, LIGHTRAG_WORKSPACES, safe_knowledge_path, set_current_thread_id
    from ...tools.knowledge import save_knowledge
    from ...tools.knowledge_manager import (
        check_knowledge_base_index,
        inspect_knowledge_base,
        list_knowledge_base_status,
        summarize_knowledge_base,
    )
    from ...tools.knowledge_qa import (
        get_active_knowledge_base_topic,
        query_knowledge_base,
        set_active_knowledge_base,
    )
except ImportError:  # pragma: no cover
    from tools._registry import KNOWLEDGE_BASE, LIGHTRAG_WORKSPACES, safe_knowledge_path, set_current_thread_id
    from tools.knowledge import save_knowledge
    from tools.knowledge_manager import (
        check_knowledge_base_index,
        inspect_knowledge_base,
        list_knowledge_base_status,
        summarize_knowledge_base,
    )
    from tools.knowledge_qa import (
        get_active_knowledge_base_topic,
        query_knowledge_base,
        set_active_knowledge_base,
    )

from .common import invoke_tool, now_text, parse_tool_json


# GraphML file name emitted by LightRAG inside each workspace storage dir.
_GRAPH_FILE = "graph_chunk_entity_relation.graphml"
# Default cap on how many nodes the light /graph endpoint returns. The current
# real sample (harness-agent-v03-e2e-1781143504) has 111 nodes, well under this;
# the cap is a forward-looking safety bound for much larger knowledge bases.
_DEFAULT_NODE_LIMIT = 600
# How many top-degree entities to surface in the `top_entities` field.
_TOP_ENTITY_COUNT = 10
# GraphML namespace + attribute-key ids (see the .graphml <key> declarations).
_GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
_SEP = "<SEP>"


def list_knowledge_bases(*, include_index: bool = True, include_stats: bool = True) -> dict[str, Any]:
    payload = invoke_tool(list_knowledge_base_status, {})
    topics = payload.get("topics") or []
    if not include_index:
        for topic in topics:
            topic.pop("index_status", None)
            topic.pop("index_reason", None)
            topic.pop("index_diagnosis", None)
    if not include_stats:
        for topic in topics:
            topic.pop("graph_stats", None)
            topic.pop("vector_stats", None)
    return {
        "active_topic": payload.get("active_topic"),
        "topic_count": payload.get("topic_count", len(topics)),
        "topics": topics,
    }


def get_knowledge_base(topic: str) -> dict[str, Any]:
    payload = invoke_tool(inspect_knowledge_base, {"topic": topic})
    return payload.get("knowledge_base") or payload


def get_summary(topic: str) -> dict[str, Any]:
    payload = invoke_tool(summarize_knowledge_base, {"topic": topic})
    return {
        "topic": payload.get("topic", topic),
        "summary": payload.get("summary", ""),
        "knowledge_base": payload.get("knowledge_base"),
    }


def list_documents(topic: str) -> dict[str, Any]:
    topic_path = _validate_topic_path(topic)
    documents = [_document_info(path) for path in sorted(topic_path.glob("*.md"))]
    return {"topic": topic, "documents": documents}


def read_document(topic: str, document_id: str) -> dict[str, Any]:
    topic_path = _validate_topic_path(topic)
    stem = unquote(document_id).removesuffix(".md")
    path = topic_path / f"{stem}.md"
    if not path.exists() or not path.is_file():
        raise ValueError(f"document not found: {document_id}")
    content = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(content)
    return {
        "topic": topic,
        "document_id": quote(path.stem, safe=""),
        "title": frontmatter.get("title") or _first_heading(body) or path.stem,
        "filename": path.name,
        "path": str(path.relative_to(KNOWLEDGE_BASE.parent)).replace("\\", "/"),
        "content": content,
        "frontmatter": frontmatter,
    }


def create_document(topic: str, *, title: str, content: str, sources: list[str] | None, keywords: list[str] | None) -> dict[str, Any]:
    _validate_topic_path(topic)
    raw = save_knowledge.invoke(
        {
            "title": title,
            "content": content,
            "topic": topic,
            "sources": sources or [],
            "keywords": keywords or [],
        }
    )
    return {
        "topic": topic,
        "message": str(raw),
        "document": read_document(topic, quote(title, safe="")),
    }


def update_metadata(topic: str, payload: dict[str, Any]) -> dict[str, Any]:
    topic_path = _validate_topic_path(topic)
    path = topic_path / "kb_metadata.json"
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    extra = payload.pop("extra", None) or {}
    current.update({key: value for key, value in payload.items() if value is not None})
    current.update(extra)
    current["topic"] = topic
    current["updated_at"] = now_text()
    current.setdefault("created_at", current["updated_at"])
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def query(topic: str, *, question: str, backend: str, mode: str) -> dict[str, Any]:
    return invoke_tool(
        query_knowledge_base,
        {"question": question, "topic": topic, "backend": backend, "mode": mode},
    )


def set_active(topic: str, thread_id: str | None = None) -> dict[str, Any]:
    if thread_id:
        set_current_thread_id(thread_id)
    return invoke_tool(set_active_knowledge_base, {"topic": topic})


def get_active(thread_id: str | None = None) -> dict[str, Any]:
    if thread_id:
        set_current_thread_id(thread_id)
    return {"active_topic": get_active_knowledge_base_topic()}


def index_status(topic: str) -> dict[str, Any]:
    diagnosis_payload = invoke_tool(check_knowledge_base_index, {"topic": topic})
    kb = get_knowledge_base(topic)
    diagnosis = diagnosis_payload.get("diagnosis") or {}
    return {
        "topic": topic,
        "index_status": diagnosis.get("index_status") or kb.get("index_status"),
        "index_reason": diagnosis.get("reason") or kb.get("index_reason"),
        "diagnosis": diagnosis,
        "embedding_model": kb.get("embedding_model"),
        "embedding_dim": kb.get("embedding_dim"),
        "indexed_at": kb.get("indexed_at"),
        "workspace": kb.get("lightrag_workspace"),
    }


def graph(topic: str, *, max_nodes: int | None = None) -> dict[str, Any]:
    kb = get_knowledge_base(topic)
    workspace = kb.get("lightrag_workspace")
    graph_path = _graph_storage_path(topic, workspace)
    parsed = _parse_graphml(graph_path)

    limit = _resolve_node_limit(max_nodes)
    nodes = parsed["nodes"]
    edges = parsed["edges"]
    total_nodes = len(nodes)
    total_edges = len(edges)

    ranked = sorted(nodes.values(), key=lambda item: (-item["degree"], item["id"]))
    truncated = total_nodes > limit
    kept = ranked[:limit] if truncated else ranked
    kept_ids = {item["id"] for item in kept}

    light_nodes = [{"id": item["id"], "type": item["type"], "degree": item["degree"]} for item in kept]
    light_edges = [
        {"source": edge["source"], "target": edge["target"], "weight": edge["weight"]}
        for edge in edges
        if edge["source"] in kept_ids and edge["target"] in kept_ids
    ]
    top_entities = [
        {"entity": item["id"], "type": item["type"], "count": item["degree"]}
        for item in ranked[:_TOP_ENTITY_COUNT]
    ]

    return {
        "topic": topic,
        # Backwards-compatible aggregate counts (kept for existing callers/stat cards).
        "nodes": (kb.get("graph_stats") or {}).get("nodes", total_nodes),
        "edges": (kb.get("graph_stats") or {}).get("edges", total_edges),
        "top_entities": top_entities,
        "workspace": workspace,
        "graph_file": _GRAPH_FILE,
        # New lightweight graph payload for interactive visualization.
        "graph_nodes": light_nodes,
        "graph_edges": light_edges,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "truncated": truncated,
        "node_limit": limit,
    }


def graph_node_detail(topic: str, entity_id: str) -> dict[str, Any]:
    kb = get_knowledge_base(topic)
    workspace = kb.get("lightrag_workspace")
    graph_path = _graph_storage_path(topic, workspace)
    parsed = _parse_graphml(graph_path)

    target = unquote(entity_id)
    node = parsed["nodes"].get(target)
    if node is None:
        raise ValueError(f"graph node not found: {entity_id}")

    neighbors: list[dict[str, Any]] = []
    for edge in parsed["edges"]:
        if edge["source"] == target:
            other = edge["target"]
        elif edge["target"] == target:
            other = edge["source"]
        else:
            continue
        other_node = parsed["nodes"].get(other) or {}
        neighbors.append(
            {
                "entity_id": other,
                "entity_type": other_node.get("type", ""),
                "weight": edge["weight"],
                "keywords": edge["keywords"],
                "description": edge["description"],
                "neighbor_preview": _truncate_text(other_node.get("description", ""), 200),
            }
        )
    neighbors.sort(key=lambda item: -item["weight"])

    return {
        "topic": topic,
        "entity_id": node["id"],
        "entity_type": node["type"],
        "degree": node["degree"],
        "description": node["description"],
        "source_id": node["source_id"],
        "file_path": node["file_path"],
        "edges": neighbors,
    }


def vectors(topic: str) -> dict[str, Any]:
    kb = get_knowledge_base(topic)
    stats = kb.get("vector_stats") or {}
    return {
        "topic": topic,
        "chunks": stats.get("chunks", 0),
        "entities": stats.get("entities", 0),
        "relationships": stats.get("relationships", 0),
        "embedding_model": kb.get("embedding_model"),
        "embedding_dim": kb.get("embedding_dim"),
        "chunk_size": kb.get("chunk_size"),
        "chunk_overlap_size": kb.get("chunk_overlap_size"),
    }


def _validate_topic_path(topic: str) -> Path:
    safe = str(topic or "").strip().replace("\\", "/").strip("/")
    if not safe or "/" in safe or ".." in safe:
        raise ValueError("topic must be a single knowledge_base directory name")
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"knowledge base topic not found: {safe}")
    return path


def _resolve_node_limit(max_nodes: int | None) -> int:
    if max_nodes is None:
        return _DEFAULT_NODE_LIMIT
    try:
        value = int(max_nodes)
    except (TypeError, ValueError):
        return _DEFAULT_NODE_LIMIT
    return value if value > 0 else _DEFAULT_NODE_LIMIT


def _graph_storage_path(topic: str, workspace: str | None) -> Path:
    """Resolve the .graphml path: LIGHTRAG_WORKSPACES/<topic>/<workspace>/<graph_file>."""
    _validate_topic_path(topic)
    ws = str(workspace or "").strip().replace("\\", "/").strip("/")
    if not ws or "/" in ws or ".." in ws:
        return LIGHTRAG_WORKSPACES / topic / _GRAPH_FILE
    return LIGHTRAG_WORKSPACES / topic / ws / _GRAPH_FILE


def _parse_graphml(path: Path) -> dict[str, Any]:
    """Parse a LightRAG GraphML file into node/edge dicts (empty when absent/invalid)."""
    empty: dict[str, Any] = {"nodes": {}, "edges": []}
    if not path.exists() or not path.is_file():
        return empty
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return empty

    graph_el = root.find("g:graph", _GRAPHML_NS)
    if graph_el is None:
        return empty

    nodes: dict[str, dict[str, Any]] = {}
    for node_el in graph_el.findall("g:node", _GRAPHML_NS):
        data = _node_data(node_el)
        entity_id = data.get("d0") or node_el.get("id") or ""
        if not entity_id:
            continue
        nodes[entity_id] = {
            "id": entity_id,
            "type": data.get("d1", "") or "",
            "description": data.get("d2", "") or "",
            "source_id": data.get("d3", "") or "",
            "file_path": data.get("d4", "") or "",
            "degree": 0,
        }

    edges: list[dict[str, Any]] = []
    for edge_el in graph_el.findall("g:edge", _GRAPHML_NS):
        source = edge_el.get("source") or ""
        target = edge_el.get("target") or ""
        if not source or not target:
            continue
        data = _node_data(edge_el)
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": _to_float(data.get("d7")),
                "description": data.get("d8", "") or "",
                "keywords": data.get("d9", "") or "",
            }
        )
        if source in nodes:
            nodes[source]["degree"] += 1
        if target in nodes:
            nodes[target]["degree"] += 1

    return {"nodes": nodes, "edges": edges}


def _node_data(element: ET.Element) -> dict[str, str]:
    """Collect <data key="dN">value</data> children into a {key: text} map."""
    result: dict[str, str] = {}
    for data_el in element.findall("g:data", _GRAPHML_NS):
        key = data_el.get("key")
        if key:
            result[key] = data_el.text or ""
    return result


def _to_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _truncate_text(text: str, limit: int) -> str:
    cleaned = (text or "").replace(_SEP, " / ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _document_info(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(content)
    return {
        "document_id": quote(path.stem, safe=""),
        "title": frontmatter.get("title") or _first_heading(body) or path.stem,
        "filename": path.name,
        "path": str(path.relative_to(KNOWLEDGE_BASE.parent)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        "frontmatter": frontmatter,
    }


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    frontmatter: dict[str, Any] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, parts[2]


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""

