"""KnowledgeScope resolution and cross-ResearchItem query aggregation."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

try:
    from ...tools._registry import LIGHTRAG_WORKSPACES
    from ...tools.lightrag_qa import query_research_lightrag, research_lightrag_index_state
except ImportError:  # pragma: no cover
    from tools._registry import LIGHTRAG_WORKSPACES
    from tools.lightrag_qa import query_research_lightrag, research_lightrag_index_state

from . import knowledge_base_service, knowledge_tree_service


ScopeType = Literal["knowledge_base", "theme", "research"]
QUERYABLE_STATUSES = {"active", "possibly_stale", "superseded"}
STATUS_WEIGHTS = {"active": 1.0, "possibly_stale": 0.65, "superseded": 0.35}
MAX_QUERY_CANDIDATES = 12
MAX_PARALLEL_QUERIES = 4


def resolve_scope(scope_type: str, scope_id: str, *, include_history: bool = False) -> dict[str, Any]:
    normalized_type = _validate_scope_type(scope_type)
    wanted = str(scope_id or "").strip()
    tree = knowledge_tree_service.list_tree(include_legacy=False)
    scope: dict[str, Any] | None = None
    research_items: list[dict[str, Any]] = []

    for kb in tree.get("knowledge_bases") or []:
        if normalized_type == "knowledge_base" and kb.get("id") == wanted:
            scope = {"type": normalized_type, "id": wanted, "title": kb.get("title"), "path": kb.get("path")}
            research_items = [*(kb.get("research_items") or [])]
            for theme in kb.get("themes") or []:
                research_items.extend(theme.get("research_items") or [])
            break
        for theme in kb.get("themes") or []:
            if normalized_type == "theme" and theme.get("id") == wanted:
                scope = {"type": normalized_type, "id": wanted, "title": theme.get("title"), "path": theme.get("path")}
                research_items = list(theme.get("research_items") or [])
                break
            for research in theme.get("research_items") or []:
                if normalized_type == "research" and research.get("id") == wanted:
                    scope = {"type": normalized_type, "id": wanted, "title": research.get("title"), "path": research.get("path")}
                    research_items = [research]
                    break
        if scope is not None:
            break
        for research in kb.get("research_items") or []:
            if normalized_type == "research" and research.get("id") == wanted:
                scope = {"type": normalized_type, "id": wanted, "title": research.get("title"), "path": research.get("path")}
                research_items = [research]
                break
        if scope is not None:
            break

    if scope is None:
        raise ValueError(f"knowledge scope not found: {normalized_type}:{wanted}")
    eligible = []
    excluded = []
    for item in research_items:
        status = str(item.get("status") or "active")
        if status not in QUERYABLE_STATUSES or (status == "superseded" and not include_history):
            excluded.append({"research_id": item.get("id"), "title": item.get("title"), "status": status})
            continue
        eligible.append({**item, "scope_weight": STATUS_WEIGHTS[status]})
    return {**scope, "research_items": eligible, "excluded_research_items": excluded, "research_count": len(eligible)}


def query_scope(
    scope_type: str,
    scope_id: str,
    question: str,
    *,
    mode: str = "mix",
    history: list[dict[str, str]] | None = None,
    include_history: bool = False,
    max_candidates: int = MAX_QUERY_CANDIDATES,
) -> dict[str, Any]:
    query = str(question or "").strip()
    if not query:
        raise ValueError("question must not be empty")
    scope = resolve_scope(scope_type, scope_id, include_history=include_history)
    candidates = _prefilter_research(query, scope["research_items"], max_candidates=max_candidates)
    skipped = list(scope["excluded_research_items"])
    ready: list[dict[str, Any]] = []
    for item in candidates:
        index_status = item.get("index_status") if isinstance(item.get("index_status"), dict) else {}
        if index_status.get("status") != "ready":
            skipped.append(
                {
                    "research_id": item.get("id"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                    "reason": f"index_{index_status.get('status') or 'pending'}",
                }
            )
            continue
        ready.append(item)

    outputs: list[dict[str, Any]] = []
    if ready:
        query_options: dict[str, Any] = {"mode": mode, "include_retrieval_trace": True}
        if history:
            query_options["conversation_history"] = history
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_QUERIES, len(ready))) as executor:
            future_map = {
                executor.submit(
                    query_research_lightrag,
                    query,
                    str(item["id"]),
                    str((knowledge_tree_service.KNOWLEDGE_ROOT / str(item["path"])).resolve()),
                    **query_options,
                ): item
                for item in ready
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    output = future.result()
                except Exception as exc:
                    skipped.append(
                        {
                            "research_id": item.get("id"),
                            "title": item.get("title"),
                            "status": item.get("status"),
                            "reason": str(exc) or type(exc).__name__,
                        }
                    )
                    continue
                failure_reason = _query_output_failure_reason(output)
                if failure_reason:
                    skipped.append(
                        {
                            "research_id": item.get("id"),
                            "title": item.get("title"),
                            "status": item.get("status"),
                            "reason": failure_reason,
                        }
                    )
                    continue
                outputs.append({"research": item, "output": output})

    outputs.sort(key=lambda item: _research_result_score(query, item["research"]), reverse=True)
    citations = _merge_citations(query, outputs)
    warnings = [
        f"调研“{item['research'].get('title')}”可能已过时，已降低其排序权重。"
        for item in outputs
        if item["research"].get("status") == "possibly_stale"
    ]
    if outputs:
        answer = _aggregate_answers(scope, outputs)
    elif ready:
        answer = "当前范围的调研索引已就绪，但问答查询暂时失败，请稍后重试。"
    else:
        answer = _aggregate_answers(scope, outputs)
    gaps = []
    if not outputs and ready:
        gaps.append("已就绪的 ResearchItem 查询均失败，请稍后重试。")
    elif not outputs:
        gaps.append("当前范围没有可用的 ResearchItem 索引。")
    if skipped:
        gaps.append(f"有 {len(skipped)} 个调研因状态、索引或查询失败未参与本次回答。")
    if outputs:
        status = "answered"
    elif ready:
        status = "query_failed"
    else:
        status = "insufficient_index"
    return {
        "status": status,
        "scope": {key: scope[key] for key in ("type", "id", "title", "path")},
        "question": query,
        "answer": answer,
        "citations": citations,
        "warnings": warnings,
        "gaps": gaps,
        "queried_research_ids": [item["research"]["id"] for item in outputs],
        "candidate_research_ids": [item["id"] for item in candidates],
        "skipped_research_items": skipped,
        "mode": mode,
    }


def _query_output_failure_reason(output: Any) -> str | None:
    if not isinstance(output, dict):
        return "invalid_query_result"
    status = str(output.get("status") or "error").strip().lower()
    answer = str(output.get("answer") or "").strip()
    if status == "answered" and answer:
        return None
    gaps = output.get("gaps")
    if isinstance(gaps, list):
        detail = next((str(item).strip() for item in gaps if str(item).strip()), "")
        if detail:
            return f"query_{status}: {detail[:300]}"
    return "query_empty_answer" if status == "answered" else f"query_{status}"


def scope_index_status(scope_type: str, scope_id: str) -> dict[str, Any]:
    scope, contexts = _scope_index_contexts(scope_type, scope_id)
    ready = [context for context in contexts if context["ready"]]
    failed = [context for context in contexts if not context["ready"]]
    if not contexts:
        status = "missing"
        reason = "empty_scope"
    elif len(ready) == len(contexts):
        status = "ready"
        reason = "ready"
    elif ready:
        status = "incomplete"
        reason = "partial_research_indexes"
    elif any(str(context["reason"]).startswith(("invalid", "error")) for context in failed):
        status = "error"
        reason = "research_indexes_failed"
    else:
        status = "missing"
        reason = "no_ready_research_indexes"

    models = {str(context["metadata"].get("embedding_model")) for context in ready if context["metadata"].get("embedding_model")}
    dimensions = {int(context["metadata"].get("embedding_dim")) for context in ready if context["metadata"].get("embedding_dim")}
    indexed_at = max(
        (str(context["metadata"].get("indexed_at") or "") for context in ready),
        default="",
    )
    return {
        "topic": _scope_key(scope),
        "scope": _scope_summary(scope),
        "index_status": status,
        "index_reason": reason,
        "diagnosis": {
            "research_count": len(contexts),
            "ready_count": len(ready),
            "unavailable_count": len(failed),
            "unavailable_research": [
                {
                    "research_id": context["research"]["id"],
                    "title": context["research"].get("title"),
                    "reason": context["reason"],
                }
                for context in failed
            ],
        },
        "embedding_model": next(iter(models)) if len(models) == 1 else ("mixed" if models else None),
        "embedding_dim": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "indexed_at": indexed_at or None,
        "workspace": f"{len(ready)} 个 ResearchItem 工作区" if ready else None,
    }


def scope_vectors(scope_type: str, scope_id: str) -> dict[str, Any]:
    scope, contexts = _scope_index_contexts(scope_type, scope_id)
    ready = [context for context in contexts if context["ready"]]
    models = {str(context["metadata"].get("embedding_model")) for context in ready if context["metadata"].get("embedding_model")}
    dimensions = {int(context["metadata"].get("embedding_dim")) for context in ready if context["metadata"].get("embedding_dim")}
    chunk_sizes = {int(context["metadata"].get("chunk_size")) for context in ready if context["metadata"].get("chunk_size")}
    overlaps = {
        int(context["metadata"].get("chunk_overlap_size"))
        for context in ready
        if context["metadata"].get("chunk_overlap_size") is not None
    }
    return {
        "topic": _scope_key(scope),
        "scope": _scope_summary(scope),
        "chunks": sum(_vdb_count(context["storage_dir"] / "vdb_chunks.json") for context in ready),
        "entities": sum(_vdb_count(context["storage_dir"] / "vdb_entities.json") for context in ready),
        "relationships": sum(_vdb_count(context["storage_dir"] / "vdb_relationships.json") for context in ready),
        "embedding_model": next(iter(models)) if len(models) == 1 else ("mixed" if models else None),
        "embedding_dim": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "chunk_size": next(iter(chunk_sizes)) if len(chunk_sizes) == 1 else None,
        "chunk_overlap_size": next(iter(overlaps)) if len(overlaps) == 1 else None,
    }


def scope_graph(scope_type: str, scope_id: str, *, max_nodes: int | None = None) -> dict[str, Any]:
    scope, contexts = _scope_index_contexts(scope_type, scope_id)
    parsed = _merge_scope_graphs(contexts)
    return _graph_payload(scope, parsed, max_nodes=max_nodes)


def scope_graph_node_detail(scope_type: str, scope_id: str, entity_id: str) -> dict[str, Any]:
    scope, contexts = _scope_index_contexts(scope_type, scope_id)
    parsed = _merge_scope_graphs(contexts)
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
                "neighbor_preview": knowledge_base_service._truncate_text(other_node.get("description", ""), 200),
            }
        )
    neighbors.sort(key=lambda item: -item["weight"])
    return {
        "topic": _scope_key(scope),
        "scope": _scope_summary(scope),
        "entity_id": node["id"],
        "entity_type": node["type"],
        "degree": node["degree"],
        "description": node["description"],
        "source_id": node["source_id"],
        "file_path": node["file_path"],
        "edges": neighbors,
    }


def _scope_index_contexts(scope_type: str, scope_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scope = resolve_scope(scope_type, scope_id)
    contexts: list[dict[str, Any]] = []
    for research in scope["research_items"]:
        research_id = str(research["id"])
        research_path = (knowledge_tree_service.KNOWLEDGE_ROOT / str(research["path"])).resolve()
        try:
            state = research_lightrag_index_state(research_id, research_path)
            metadata = state.metadata or {}
            reason = state.reason
            ready = state.available
        except Exception as exc:
            metadata = {}
            reason = str(exc) or type(exc).__name__
            ready = False
        workspace = str(metadata.get("lightrag_workspace") or "").strip().replace("\\", "/").strip("/")
        workspace_root = LIGHTRAG_WORKSPACES / "research" / research_id
        storage_dir = workspace_root / workspace if workspace and "/" not in workspace and ".." not in workspace else workspace_root
        contexts.append(
            {
                "research": research,
                "ready": ready,
                "reason": reason,
                "metadata": metadata,
                "storage_dir": storage_dir,
            }
        )
    return scope, contexts


def _merge_scope_graphs(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for context in contexts:
        if not context["ready"]:
            continue
        parsed = knowledge_base_service._parse_graphml(context["storage_dir"] / "graph_chunk_entity_relation.graphml")
        for entity_id, node in parsed["nodes"].items():
            current = nodes.setdefault(
                entity_id,
                {"id": entity_id, "type": node["type"], "description": "", "source_id": "", "file_path": "", "degree": 0},
            )
            if not current["type"] and node["type"]:
                current["type"] = node["type"]
            for field in ("description", "source_id", "file_path"):
                current[field] = _merge_sep_text(current[field], node[field])
        for edge in parsed["edges"]:
            key = (edge["source"], edge["target"])
            current = edges.setdefault(
                key,
                {"source": edge["source"], "target": edge["target"], "weight": 0.0, "description": "", "keywords": ""},
            )
            current["weight"] += float(edge["weight"] or 0)
            current["description"] = _merge_sep_text(current["description"], edge["description"])
            current["keywords"] = _merge_sep_text(current["keywords"], edge["keywords"])
    merged_edges = list(edges.values())
    for edge in merged_edges:
        if edge["source"] in nodes:
            nodes[edge["source"]]["degree"] += 1
        if edge["target"] in nodes:
            nodes[edge["target"]]["degree"] += 1
    return {"nodes": nodes, "edges": merged_edges}


def _graph_payload(scope: dict[str, Any], parsed: dict[str, Any], *, max_nodes: int | None) -> dict[str, Any]:
    limit = knowledge_base_service._resolve_node_limit(max_nodes)
    nodes = parsed["nodes"]
    edges = parsed["edges"]
    ranked = sorted(nodes.values(), key=lambda item: (-item["degree"], item["id"]))
    truncated = len(ranked) > limit
    kept = ranked[:limit] if truncated else ranked
    kept_ids = {item["id"] for item in kept}
    return {
        "topic": _scope_key(scope),
        "scope": _scope_summary(scope),
        "nodes": len(nodes),
        "edges": len(edges),
        "top_entities": [
            {"entity": item["id"], "type": item["type"], "count": item["degree"]}
            for item in ranked[:10]
        ],
        "workspace": f"{scope['research_count']} 个 ResearchItem 工作区",
        "graph_file": "graph_chunk_entity_relation.graphml",
        "graph_nodes": [{"id": item["id"], "type": item["type"], "degree": item["degree"]} for item in kept],
        "graph_edges": [
            {"source": edge["source"], "target": edge["target"], "weight": edge["weight"]}
            for edge in edges
            if edge["source"] in kept_ids and edge["target"] in kept_ids
        ],
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "truncated": truncated,
        "node_limit": limit,
    }


def _merge_sep_text(current: str, incoming: str) -> str:
    values = [value.strip() for value in f"{current}<SEP>{incoming}".split("<SEP>") if value.strip()]
    return "<SEP>".join(dict.fromkeys(values))


def _vdb_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    data = payload.get("data") if isinstance(payload, dict) else None
    return len(data) if isinstance(data, list) else 0


def _scope_key(scope: dict[str, Any]) -> str:
    return f"{scope['type']}:{scope['id']}"


def _scope_summary(scope: dict[str, Any]) -> dict[str, Any]:
    return {key: scope[key] for key in ("type", "id", "title", "path")}


def scope_content_hash(scope_type: str, scope_id: str, *, include_history: bool = False) -> str:
    scope = resolve_scope(scope_type, scope_id, include_history=include_history)
    digest = hashlib.sha256()
    for item in sorted(scope["research_items"], key=lambda value: str(value.get("id") or "")):
        digest.update(str(item.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.get("status") or "").encode("utf-8"))
        digest.update(b"\0")
        for document in item.get("documents") or []:
            digest.update(str(document.get("filename") or "").encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(document.get("sha256") or "").encode("utf-8"))
            digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def scope_case_key(scope_type: str, scope_id: str) -> str:
    normalized = _validate_scope_type(scope_type)
    wanted = str(scope_id or "").strip()
    if not wanted:
        raise ValueError("scope_id is required")
    return f"scope:{normalized}:{wanted}"


def _validate_scope_type(value: str) -> ScopeType:
    normalized = str(value or "").strip().lower()
    if normalized not in {"knowledge_base", "theme", "research"}:
        raise ValueError("scope_type must be knowledge_base, theme, or research")
    return normalized  # type: ignore[return-value]


def _prefilter_research(question: str, items: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    limit = max(1, min(int(max_candidates or MAX_QUERY_CANDIDATES), MAX_QUERY_CANDIDATES))
    ranked = sorted(items, key=lambda item: _research_result_score(question, item), reverse=True)
    return ranked[:limit]


def _research_result_score(question: str, item: dict[str, Any]) -> float:
    query = re.sub(r"\s+", "", question).casefold()
    text = " ".join(
        [str(item.get("title") or ""), str(item.get("summary") or ""), *[str(tag) for tag in item.get("tags") or []]]
    )
    normalized = re.sub(r"\s+", "", text).casefold()
    lexical = sum(1 for token in re.findall(r"[a-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", query) if token in normalized)
    recency = 0.0
    try:
        researched_at = datetime.fromisoformat(str(item.get("researched_at") or ""))
        recency = researched_at.timestamp() / 10_000_000_000
    except ValueError:
        pass
    return (lexical * 3.0 + recency + 1.0) * float(item.get("scope_weight") or 1.0)


def _merge_citations(question: str, outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    query = re.sub(r"\s+", "", question).casefold()
    for output_item in outputs:
        research = output_item["research"]
        citations = output_item["output"].get("citations") or []
        for rank, citation in enumerate(citations, start=1):
            content = str(citation.get("content") or citation.get("text") or "")
            path = str(citation.get("path") or citation.get("file_path") or "")
            url = str(citation.get("url") or citation.get("source_url") or "")
            key_source = f"{path}\0{url}\0{content[:500]}"
            key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
            lexical = sum(1 for token in re.findall(r"[a-z0-9_+-]{2,}|[\u4e00-\u9fff]{2,}", query) if token in re.sub(r"\s+", "", content).casefold())
            score = float(research.get("scope_weight") or 1.0) * (1.0 / rank + lexical * 0.2)
            candidate = {
                **citation,
                "research_id": research.get("id"),
                "research_title": research.get("title"),
                "research_status": research.get("status"),
                "aggregate_score": round(score, 6),
            }
            if key not in merged or score > float(merged[key].get("aggregate_score") or 0):
                merged[key] = candidate
    return sorted(merged.values(), key=lambda item: float(item.get("aggregate_score") or 0), reverse=True)[:20]


def _aggregate_answers(scope: dict[str, Any], outputs: list[dict[str, Any]]) -> str:
    if not outputs:
        return "当前范围内没有已就绪的调研索引，暂时无法基于知识库回答。"
    if len(outputs) == 1:
        return str(outputs[0]["output"].get("answer") or "").strip()
    lines = [f"基于“{scope['title']}”中的 {len(outputs)} 次调研综合整理："]
    for item in outputs[:5]:
        research = item["research"]
        answer = str(item["output"].get("answer") or "").strip()
        if not answer:
            continue
        status_note = "，可能已过时" if research.get("status") == "possibly_stale" else ""
        lines.extend(["", f"### {research.get('title')}（{str(research.get('researched_at') or '')[:10]}{status_note}）", answer[:3000]])
    return "\n".join(lines)
