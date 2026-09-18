"""Manifest-backed personal knowledge hierarchy routes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ..schemas.knowledge_tree import KnowledgeScopeQueryRequest, KnowledgeTreePreviewRequest
from ..schemas.index import IndexJobCreateRequest
from ..services import index_service, knowledge_scope_service, knowledge_tree_service
from ..services.common import accepted, map_exception, ok


router = APIRouter(prefix="/api/v1/knowledge-tree", tags=["knowledge-tree"])


@router.get("")
def get_knowledge_tree(include_legacy: bool = True):
    try:
        return ok(knowledge_tree_service.list_tree(include_legacy=include_legacy))
    except Exception as exc:
        return map_exception(exc)


@router.get("/trash")
def get_knowledge_tree_trash():
    try:
        return ok(knowledge_tree_service.list_trash())
    except Exception as exc:
        return map_exception(exc)


@router.get("/nodes/{node_type}/{node_id}/overview")
def get_knowledge_node_overview(node_type: str, node_id: str):
    try:
        return ok(knowledge_tree_service.read_node_overview(node_type, node_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/research/{research_id}/documents/{filename}")
def get_research_document(research_id: str, filename: str):
    try:
        return ok(knowledge_tree_service.read_research_document(research_id, filename))
    except Exception as exc:
        return map_exception(exc)


@router.get("/research/{research_id}/documents/{filename}/html", response_class=HTMLResponse)
def get_research_document_html(research_id: str, filename: str):
    try:
        document = knowledge_tree_service.read_research_document_html(research_id, filename)
        return HTMLResponse(content=document["content"])
    except Exception as exc:
        return map_exception(exc)


@router.get("/research/{research_id}/html", response_class=HTMLResponse)
def get_research_html(research_id: str):
    try:
        document = knowledge_tree_service.read_research_html(research_id)
        return HTMLResponse(content=document["content"])
    except Exception as exc:
        return map_exception(exc)


@router.get("/research/{research_id}/reader", response_class=HTMLResponse)
def get_research_reader(research_id: str):
    try:
        document = knowledge_tree_service.read_research_html(research_id)
        return HTMLResponse(content=document["content"])
    except Exception as exc:
        return map_exception(exc)


@router.get("/research/{research_id}/index")
def get_research_index(research_id: str):
    try:
        node = knowledge_tree_service.get_research_node(research_id)
        return ok({"research_id": research_id, "index_status": node["manifest"].get("index_status") or {"status": "pending"}})
    except Exception as exc:
        return map_exception(exc)


@router.get("/scopes/{scope_type}/{scope_id}")
def get_knowledge_scope(scope_type: str, scope_id: str, include_history: bool = False):
    try:
        return ok(knowledge_scope_service.resolve_scope(scope_type, scope_id, include_history=include_history))
    except Exception as exc:
        return map_exception(exc)


@router.get("/scopes/{scope_type}/{scope_id}/index")
def get_knowledge_scope_index(scope_type: str, scope_id: str):
    try:
        return ok(knowledge_scope_service.scope_index_status(scope_type, scope_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/scopes/{scope_type}/{scope_id}/graph")
def get_knowledge_scope_graph(scope_type: str, scope_id: str, max_nodes: int | None = Query(default=None, ge=1)):
    try:
        return ok(knowledge_scope_service.scope_graph(scope_type, scope_id, max_nodes=max_nodes))
    except Exception as exc:
        return map_exception(exc)


@router.get("/scopes/{scope_type}/{scope_id}/graph/nodes/{entity_id}")
def get_knowledge_scope_graph_node(scope_type: str, scope_id: str, entity_id: str):
    try:
        return ok(knowledge_scope_service.scope_graph_node_detail(scope_type, scope_id, entity_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/scopes/{scope_type}/{scope_id}/vectors")
def get_knowledge_scope_vectors(scope_type: str, scope_id: str):
    try:
        return ok(knowledge_scope_service.scope_vectors(scope_type, scope_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/scopes/{scope_type}/{scope_id}/query")
def query_knowledge_scope(scope_type: str, scope_id: str, request: KnowledgeScopeQueryRequest):
    try:
        return ok(
            knowledge_scope_service.query_scope(
                scope_type,
                scope_id,
                request.question,
                mode=request.mode,
                history=[{"role": item.role, "content": item.content} for item in request.history],
                include_history=request.include_history,
                max_candidates=request.max_candidates,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.post("/research/{research_id}/index/jobs")
def create_research_index_job(research_id: str, request: IndexJobCreateRequest):
    try:
        job = index_service.create_research_index_job(research_id, force=request.force)
        return accepted({"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]})
    except Exception as exc:
        return map_exception(exc)


@router.delete("/research/{research_id}/index")
def delete_research_index(research_id: str):
    try:
        return ok(index_service.delete_research_index(research_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/previews")
def create_knowledge_tree_preview(request: KnowledgeTreePreviewRequest):
    try:
        return ok(knowledge_tree_service.create_preview(request.action, request.payload))
    except Exception as exc:
        return map_exception(exc)


@router.get("/previews/{preview_id}")
def get_knowledge_tree_preview(preview_id: str):
    try:
        return ok(knowledge_tree_service.get_preview(preview_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/previews/{preview_id}/confirm")
def confirm_knowledge_tree_preview(preview_id: str):
    try:
        return ok(knowledge_tree_service.confirm_preview(preview_id))
    except Exception as exc:
        return map_exception(exc)
