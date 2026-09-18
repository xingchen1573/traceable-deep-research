"""Knowledge-base API v1 routes."""

from __future__ import annotations

import asyncio
import json
import queue
import threading

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from ..schemas.knowledge_base import (
    ActiveKnowledgeBaseRequest,
    KnowledgeDocumentCreateRequest,
    KnowledgeMetadataUpdateRequest,
    KnowledgeQaHistoryRequest,
    KnowledgeQueryRequest,
)
from ..services import conversation_service, knowledge_base_service
from ..services.common import map_exception, ok

try:
    from ...tools.knowledge_qa import stream_query_knowledge_base
except ImportError:  # pragma: no cover - top-level backend package fallback
    from tools.knowledge_qa import stream_query_knowledge_base

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


@router.get("/history/qa")
def list_qa_history(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    items = conversation_service.list_qa_conversations(limit=limit, offset=offset)
    total = conversation_service.count_qa_conversations()
    return ok(
        {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }
    )


@router.post("/history/qa")
def save_qa_history(request: KnowledgeQaHistoryRequest):
    try:
        return ok(conversation_service.upsert_qa_conversation(request.model_dump(by_alias=True)))
    except AttributeError:
        return ok(conversation_service.upsert_qa_conversation(request.dict(by_alias=True)))


@router.delete("/history/qa/{session_id}")
def delete_qa_history(session_id: str):
    return ok({"id": session_id, "deleted": conversation_service.delete_qa_conversation(session_id)})


@router.get("")
def list_knowledge_bases(include_index: bool = True, include_stats: bool = True):
    try:
        return ok(knowledge_base_service.list_knowledge_bases(include_index=include_index, include_stats=include_stats))
    except Exception as exc:
        return map_exception(exc)


@router.get("/active")
def get_active_knowledge_base(thread_id: str | None = None):
    try:
        return ok(knowledge_base_service.get_active(thread_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/active")
def set_active_knowledge_base(request: ActiveKnowledgeBaseRequest):
    try:
        return ok(knowledge_base_service.set_active(request.topic, request.thread_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}")
def get_knowledge_base(topic: str):
    try:
        return ok(knowledge_base_service.get_knowledge_base(topic))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/summary")
def get_knowledge_base_summary(topic: str):
    try:
        return ok(knowledge_base_service.get_summary(topic))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/documents")
def list_documents(topic: str):
    try:
        return ok(knowledge_base_service.list_documents(topic))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/documents/{document_id}")
def read_document(topic: str, document_id: str):
    try:
        return ok(knowledge_base_service.read_document(topic, document_id))
    except Exception as exc:
        return map_exception(exc)


@router.post("/{topic}/documents")
def create_document(topic: str, request: KnowledgeDocumentCreateRequest):
    try:
        return ok(
            knowledge_base_service.create_document(
                topic,
                title=request.title,
                content=request.content,
                sources=request.sources,
                keywords=request.keywords,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.post("/{topic}/metadata")
def update_metadata(topic: str, request: KnowledgeMetadataUpdateRequest):
    try:
        return ok(knowledge_base_service.update_metadata(topic, request.model_dump()))
    except AttributeError:  # pydantic v1 fallback
        return ok(knowledge_base_service.update_metadata(topic, request.dict()))
    except Exception as exc:
        return map_exception(exc)


@router.post("/{topic}/query")
def query_knowledge_base(topic: str, request: KnowledgeQueryRequest):
    try:
        return ok(
            knowledge_base_service.query(
                topic,
                question=request.question,
                backend=request.backend,
                mode=request.mode,
            )
        )
    except Exception as exc:
        return map_exception(exc)


@router.post("/{topic}/query/stream")
async def stream_knowledge_base_query(topic: str, request: KnowledgeQueryRequest):
    """Stream model-level answer tokens for the knowledge-base QA surface."""

    async def events():
        updates: queue.Queue[tuple[str, Any]] = queue.Queue()

        def run_query() -> None:
            try:
                result = stream_query_knowledge_base(
                    topic,
                    question=request.question,
                    history=[
                        {"role": item.role, "content": item.content}
                        for item in request.history
                    ],
                    backend=request.backend,
                    mode=request.mode,
                    on_token=lambda text: updates.put(("delta", text)),
                )
                updates.put(("done", result))
            except Exception as exc:
                updates.put(("error", {"message": str(exc)[:500]}))

        thread = threading.Thread(target=run_query, daemon=True)
        thread.start()
        yield f"event: start\ndata: {json.dumps({'topic': topic}, ensure_ascii=False)}\n\n"
        while thread.is_alive() or not updates.empty():
            try:
                event, payload = await asyncio.to_thread(updates.get, True, 0.1)
            except queue.Empty:
                continue
            if event == "delta":
                yield f"event: delta\ndata: {json.dumps({'text': payload}, ensure_ascii=False)}\n\n"
            elif event == "done":
                yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{topic}/index")
def get_index(topic: str):
    try:
        return ok(knowledge_base_service.index_status(topic))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/graph")
def get_graph(topic: str, max_nodes: int | None = Query(default=None, ge=1)):
    try:
        return ok(knowledge_base_service.graph(topic, max_nodes=max_nodes))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/graph/nodes/{entity_id}")
def get_graph_node(topic: str, entity_id: str):
    try:
        return ok(knowledge_base_service.graph_node_detail(topic, entity_id))
    except Exception as exc:
        return map_exception(exc)


@router.get("/{topic}/vectors")
def get_vectors(topic: str):
    try:
        return ok(knowledge_base_service.vectors(topic))
    except Exception as exc:
        return map_exception(exc)
