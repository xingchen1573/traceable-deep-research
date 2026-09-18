"""LightRAG backend for local knowledge-base QA."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine

try:
    from ..observability import emit_event
except ImportError:  # pragma: no cover - script mode fallback
    from observability import emit_event

from ._registry import (
    KNOWLEDGE_BASE,
    LIGHTRAG_WORKSPACES,
    ensure_lightrag_workspaces,
    safe_knowledge_path,
    validate_topic_shape,
)

METADATA_FILE = "index_metadata.json"
SUPPORTED_MODES = {"local", "global", "hybrid", "naive", "mix"}
DEFAULT_PROCESS_OPTIONS = ""
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_LOOP_THREAD: threading.Thread | None = None
_ASYNC_LOOP_LOCK = threading.Lock()


@dataclass(frozen=True)
class IndexState:
    available: bool
    reason: str
    metadata: dict[str, Any]


def build_lightrag_index(topic: str, *, force: bool = False) -> dict[str, Any]:
    """Build or refresh the LightRAG index for one knowledge-base topic."""
    _load_env()
    safe_topic = _validate_topic(topic)
    ensure_lightrag_workspaces()
    workspace_dir = _workspace_dir(safe_topic)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    docs = _topic_markdown_documents(safe_topic)
    metadata = _build_metadata(safe_topic, docs)
    old_state = lightrag_index_state(safe_topic)
    if old_state.available and not force and _metadata_matches(old_state.metadata, metadata):
        return {
            "status": "ready",
            "topic": safe_topic,
            "backend": "lightrag",
            "workspace": str(workspace_dir.relative_to(LIGHTRAG_WORKSPACES.parent)),
            "indexed_documents": len(docs),
            "message": "LightRAG index is already up to date.",
            "metadata": old_state.metadata,
        }

    emit_event(
        "knowledge_qa.lightrag_index.started",
        "tools",
        "Started LightRAG knowledge-base indexing.",
        data={"topic": safe_topic, "document_count": len(docs), "force": force},
    )
    try:
        result = _run_async(_build_lightrag_index_async(safe_topic, docs, metadata, force=force))
    except Exception as exc:
        emit_event(
            "knowledge_qa.lightrag_index.failed",
            "tools",
            "LightRAG knowledge-base indexing failed.",
            level="error",
            data={
                "topic": safe_topic,
                "document_count": len(docs),
                "force": force,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
        )
        raise
    emit_event(
        "knowledge_qa.lightrag_indexed",
        "tools",
        "Built LightRAG knowledge-base index.",
        data={"topic": safe_topic, "document_count": len(docs), "force": force},
    )
    return result


def build_research_lightrag_index(research_id: str, research_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Build an independent LightRAG workspace for one immutable ResearchItem."""

    _load_env()
    safe_id = _validate_research_id(research_id)
    path = Path(research_path).resolve()
    manifest_path = path / "_research.json"
    if not path.is_dir() or not manifest_path.exists():
        raise ValueError(f"research item not found: {safe_id}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid research manifest: {safe_id}") from exc
    if manifest.get("research_id") != safe_id:
        raise ValueError("research_id does not match its manifest")
    docs = sorted(path.glob("*.md"))
    if not docs:
        raise ValueError(f"research item has no Markdown documents: {safe_id}")

    ensure_lightrag_workspaces()
    workspace_dir = _research_workspace_dir(safe_id)
    workspace_name = _research_workspace_name(safe_id)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    metadata = _build_research_metadata(safe_id, docs)
    old_state = research_lightrag_index_state(safe_id, path)
    if old_state.available and not force and _metadata_matches(old_state.metadata, metadata):
        return {
            "status": "ready",
            "research_id": safe_id,
            "backend": "lightrag",
            "workspace": str(workspace_dir.relative_to(LIGHTRAG_WORKSPACES.parent)),
            "indexed_documents": len(docs),
            "metadata": old_state.metadata,
        }
    result = _run_async(
        _build_lightrag_index_async(
            safe_id,
            docs,
            metadata,
            force=force,
            workspace_dir=workspace_dir,
            workspace_name=workspace_name,
        )
    )
    result.pop("topic", None)
    result["research_id"] = safe_id
    return result


def query_lightrag(
    question: str,
    topic: str,
    *,
    mode: str = "mix",
    conversation_history: list[dict[str, str]] | None = None,
    include_retrieval_trace: bool = False,
) -> dict[str, Any]:
    """Query an existing LightRAG index."""
    _load_env()
    safe_topic = _validate_topic(topic)
    normalized_mode = _validate_mode(mode)
    state = lightrag_index_state(safe_topic)
    if not state.available:
        raise RuntimeError(f"LightRAG index is not available for topic '{safe_topic}': {state.reason}")

    return _run_async(
        _query_lightrag_async(
            question,
            safe_topic,
            normalized_mode,
            state.metadata,
            conversation_history=conversation_history,
            include_retrieval_trace=include_retrieval_trace,
        )
    )


def query_research_lightrag(
    question: str,
    research_id: str,
    research_path: str | Path,
    *,
    mode: str = "mix",
    conversation_history: list[dict[str, str]] | None = None,
    include_retrieval_trace: bool = False,
) -> dict[str, Any]:
    """Query one stable-ID ResearchItem workspace."""

    _load_env()
    safe_id = _validate_research_id(research_id)
    normalized_mode = _validate_mode(mode)
    state = research_lightrag_index_state(safe_id, research_path)
    if not state.available:
        raise RuntimeError(f"LightRAG index is not available for research '{safe_id}': {state.reason}")
    output = _run_async(
        _query_lightrag_async(
            question,
            safe_id,
            normalized_mode,
            state.metadata,
            conversation_history=conversation_history,
            include_retrieval_trace=include_retrieval_trace,
            workspace_dir=_research_workspace_dir(safe_id),
            workspace_name=_research_workspace_name(safe_id),
        )
    )
    output.pop("topic", None)
    output["research_id"] = safe_id
    return output


def query_lightrag_stream(
    question: str,
    topic: str,
    *,
    mode: str = "mix",
    conversation_history: list[dict[str, str]] | None = None,
    on_token: Any,
) -> dict[str, Any]:
    """Query LightRAG while forwarding upstream LLM tokens to ``on_token``."""
    _load_env()
    safe_topic = _validate_topic(topic)
    normalized_mode = _validate_mode(mode)
    state = lightrag_index_state(safe_topic)
    if not state.available:
        raise RuntimeError(f"LightRAG index is not available for topic '{safe_topic}': {state.reason}")
    return _run_async(
        _query_lightrag_async(
            question,
            safe_topic,
            normalized_mode,
            state.metadata,
            on_token=on_token,
            conversation_history=conversation_history,
        )
    )


def lightrag_index_state(topic: str) -> IndexState:
    """Return whether the topic has a current LightRAG index."""
    _load_env()
    safe_topic = _validate_topic(topic)
    metadata_path = _metadata_path(safe_topic)
    if not metadata_path.exists():
        return IndexState(False, "missing_metadata", {})

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return IndexState(False, f"invalid_metadata: {exc}", {})

    expected = _build_metadata(safe_topic, _topic_markdown_documents(safe_topic))
    if not _metadata_matches(metadata, expected):
        return IndexState(False, "stale_metadata", metadata)

    storage_markers = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "graph_chunk_entity_relation.graphml",
    ]
    storage_dir = _workspace_dir(safe_topic) / _workspace_name(safe_topic)
    missing = [name for name in storage_markers if not (storage_dir / name).exists()]
    if missing:
        return IndexState(False, f"missing_storage_files: {', '.join(missing)}", metadata)

    return IndexState(True, "ready", metadata)


def research_lightrag_index_state(research_id: str, research_path: str | Path) -> IndexState:
    safe_id = _validate_research_id(research_id)
    path = Path(research_path).resolve()
    metadata_path = _research_workspace_dir(safe_id) / METADATA_FILE
    if not metadata_path.exists():
        return IndexState(False, "missing_metadata", {})
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return IndexState(False, f"invalid_metadata: {exc}", {})
    docs = sorted(path.glob("*.md")) if path.is_dir() else []
    expected = _build_research_metadata(safe_id, docs)
    if not _metadata_matches(metadata, expected):
        return IndexState(False, "stale_metadata", metadata)
    storage_dir = _research_workspace_dir(safe_id) / _research_workspace_name(safe_id)
    markers = ["kv_store_full_docs.json", "kv_store_text_chunks.json", "vdb_chunks.json", "graph_chunk_entity_relation.graphml"]
    missing = [name for name in markers if not (storage_dir / name).exists()]
    if missing:
        return IndexState(False, f"missing_storage_files: {', '.join(missing)}", metadata)
    return IndexState(True, "ready", metadata)


def _validate_research_id(value: str) -> str:
    safe = str(value or "").strip()
    if not re.fullmatch(r"research_[a-f0-9]{32}", safe):
        raise ValueError("invalid research_id")
    return safe


def _research_workspace_dir(research_id: str) -> Path:
    return LIGHTRAG_WORKSPACES / "research" / research_id


def _research_workspace_name(research_id: str) -> str:
    return f"research_{research_id[-12:]}"


def _validate_topic(topic: str) -> str:
    safe = str(topic or "").strip().replace("\\", "/").strip("/")
    if not safe or ".." in safe:
        raise ValueError("topic must be a single knowledge_base directory name")
    segments = validate_topic_shape(safe)
    safe = "/".join(segments)
    path = safe_knowledge_path(safe)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"knowledge base topic not found: {safe}")
    if not list(path.glob("*.md")):
        raise ValueError(f"knowledge base topic has no Markdown documents: {safe}")
    return safe


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        env_path = KNOWLEDGE_BASE.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ModuleNotFoundError:
        return


def _validate_mode(mode: str) -> str:
    normalized = str(mode or "mix").strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
    return normalized


def _workspace_dir(topic: str) -> Path:
    return LIGHTRAG_WORKSPACES / topic


def _workspace_name(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", topic).strip("_") or "topic"
    digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]
    return f"kb_{slug[:40]}_{digest}"


def _metadata_path(topic: str) -> Path:
    return _workspace_dir(topic) / METADATA_FILE


def _topic_markdown_documents(topic: str) -> list[Path]:
    return sorted(safe_knowledge_path(topic).glob("*.md"))


def _build_metadata(topic: str, docs: list[Path]) -> dict[str, Any]:
    return {
        "topic": topic,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "1024")),
        "embedding_token_limit": int(os.getenv("EMBEDDING_TOKEN_LIMIT", "512")),
        "embedding_send_dim": os.getenv("EMBEDDING_SEND_DIM", "false").lower() in {"1", "true", "yes"},
        "embedding_batch_num": min(int(os.getenv("EMBEDDING_BATCH_NUM", "16")), 2),
        "embedding_input_chars": 500,
        "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
        "chunk_overlap_size": int(os.getenv("CHUNK_OVERLAP_SIZE", "60")),
        "process_options": _process_options(),
        "documents": _document_metadata(docs),
    }


def _build_research_metadata(research_id: str, docs: list[Path]) -> dict[str, Any]:
    metadata = _build_metadata(research_id, docs)
    metadata.update({"scope_type": "research", "scope_id": research_id})
    return metadata


def _document_metadata(docs: list[Path]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in docs:
        try:
            relative = path.relative_to(KNOWLEDGE_BASE.parent)
            display_path = str(relative).replace("\\", "/")
        except ValueError:
            display_path = str(path)
        values.append(
            {
                "path": display_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    return values


def _metadata_matches(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    comparable_keys = [
        "topic",
        "embedding_model",
        "embedding_dim",
        "embedding_token_limit",
        "embedding_send_dim",
        "embedding_batch_num",
        "embedding_input_chars",
        "chunk_size",
        "chunk_overlap_size",
        "process_options",
        "documents",
        "scope_type",
        "scope_id",
    ]
    return all(current.get(key) == expected.get(key) for key in comparable_keys)


def _process_options() -> str:
    # Empty process_options means full LightRAG processing: chunk vectors plus
    # entity/relation extraction. Set LIGHTRAG_KB_PROCESS_OPTIONS="!" only for
    # a deliberately fast vector-only rebuild.
    return os.getenv("LIGHTRAG_KB_PROCESS_OPTIONS", DEFAULT_PROCESS_OPTIONS).strip()


async def _build_lightrag_index_async(
    topic: str,
    docs: list[Path],
    metadata: dict[str, Any],
    *,
    force: bool,
    workspace_dir: Path | None = None,
    workspace_name: str | None = None,
) -> dict[str, Any]:
    resolved_workspace_dir = workspace_dir or _workspace_dir(topic)
    resolved_workspace_name = workspace_name or _workspace_name(topic)
    rag = _create_rag(
        topic,
        working_dir=resolved_workspace_dir,
        workspace=resolved_workspace_name,
        use_selected_qa_provider=True,
    )
    try:
        await rag.initialize_storages()
        if force:
            await _drop_existing_storage(rag)
        contents = []
        file_paths = []
        ids = []
        for path in docs:
            rel_path = str(path.relative_to(KNOWLEDGE_BASE.parent)).replace("\\", "/")
            contents.append(path.read_text(encoding="utf-8"))
            file_paths.append(rel_path)
            ids.append(hashlib.sha256(rel_path.encode("utf-8")).hexdigest())
        await rag.apipeline_enqueue_documents(
            contents,
            ids=ids,
            file_paths=file_paths,
            process_options=[metadata["process_options"]] * len(contents),
        )
        await rag.apipeline_process_enqueue_documents()
    finally:
        await rag.finalize_storages()

    storage_dir = resolved_workspace_dir / resolved_workspace_name
    required_files = ["kv_store_text_chunks.json", "vdb_chunks.json", "graph_chunk_entity_relation.graphml"]
    missing = [name for name in required_files if not (storage_dir / name).exists()]
    if missing:
        raise RuntimeError(f"LightRAG index incomplete; missing files: {', '.join(missing)}")

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "lightrag_workspace": resolved_workspace_name,
        }
    )
    (resolved_workspace_dir / METADATA_FILE).write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "indexed",
        "topic": topic,
        "backend": "lightrag",
        "workspace": str(resolved_workspace_dir.relative_to(LIGHTRAG_WORKSPACES.parent)),
        "indexed_documents": len(docs),
        "metadata": final_metadata,
    }


async def _query_lightrag_async(
    question: str,
    topic: str,
    mode: str,
    metadata: dict[str, Any],
    on_token: Any = None,
    conversation_history: list[dict[str, str]] | None = None,
    include_retrieval_trace: bool = False,
    workspace_dir: Path | None = None,
    workspace_name: str | None = None,
) -> dict[str, Any]:
    rag = _create_rag(
        topic,
        working_dir=workspace_dir,
        workspace=workspace_name,
        use_selected_qa_provider=True,
    )
    try:
        await rag.initialize_storages()
        from lightrag import QueryParam

        param = QueryParam(
            mode=mode,
            include_references=True,
            stream=on_token is not None,
            enable_rerank=False,
            conversation_history=conversation_history or [],
            response_type="中文结构化回答，包含简短结论和要点列表",
            user_prompt=(
                "你只能依据上方检索上下文回答。"
                "如果问题中的术语在上下文中有特定定义，必须采用上下文中的定义，"
                "不得替换为外部常识或同名概念。"
                "不要补充上下文没有支持的行业通用解释。"
                "如果证据不足，直接说明不足。"
                "回答末尾用一句话说明主要依据来自哪些文档。"
            ),
        )
        result = await rag.aquery_llm(
            question,
            param,
        )
        llm_response = result.get("llm_response") or {}
        if on_token is not None and llm_response.get("is_streaming"):
            chunks: list[str] = []
            async for token in llm_response.get("response_iterator"):
                text = str(token or "")
                if text:
                    chunks.append(text)
                    on_token(text)
            llm_response["content"] = "".join(chunks)
    finally:
        await rag.finalize_storages()

    llm_response = result.get("llm_response") or {}
    data = result.get("data") or {}
    citations = _citations_from_lightrag_data(data)
    status = "answered" if result.get("status") == "success" else "error"
    output = {
        "status": status,
        "topic": topic,
        "backend": "lightrag",
        "mode": mode,
        "answer": (llm_response.get("content") or "").strip(),
        "citations": citations,
        "confidence": "medium" if citations else "low",
        "gaps": [] if status == "answered" and citations else [result.get("message") or "LightRAG did not return citations."],
        "metadata": {
            "embedding_model": metadata.get("embedding_model"),
            "embedding_dim": metadata.get("embedding_dim"),
            "indexed_at": metadata.get("indexed_at"),
        },
    }
    if include_retrieval_trace:
        output["retrieval_trace"] = _retrieval_trace_from_lightrag_data(data)
    return output


def _create_rag(
    topic: str,
    *,
    working_dir: Path | None = None,
    workspace: str | None = None,
    use_selected_qa_provider: bool = False,
):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from openai import AsyncOpenAI
    import numpy as np

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST") or None
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY") or None
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_token_limit = int(os.getenv("EMBEDDING_TOKEN_LIMIT", "512"))

    async def embed(texts: list[str], context: str = "document", max_token_size: int | None = None):
        del context, max_token_size
        client_kwargs: dict[str, Any] = {
            "api_key": embedding_api_key,
        }
        if embedding_base_url:
            client_kwargs["base_url"] = embedding_base_url
        client = AsyncOpenAI(**client_kwargs)
        clipped = [text[:500] for text in texts]
        try:
            response = await client.embeddings.create(
                model=embedding_model,
                input=clipped,
            )
        finally:
            await client.close()
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=embed,
        max_token_size=embedding_token_limit,
        send_dimensions=False,
        model_name=embedding_model,
        supports_asymmetric=True,
    )
    llm_model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    llm_model_func = _main_agent_lightrag_model_func()
    if use_selected_qa_provider and _selected_qa_model_uses_preferred_provider():
        try:
            from ..backend.research.model_runtime import preferred_provider_model_name
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.research.model_runtime import preferred_provider_model_name

        llm_model_func = _preferred_provider_lightrag_model_func()
        llm_model_name = preferred_provider_model_name()
    return LightRAG(
        working_dir=str(working_dir or _workspace_dir(topic)),
        workspace=workspace or _workspace_name(topic),
        embedding_func=embedding_func,
        embedding_batch_num=min(int(os.getenv("EMBEDDING_BATCH_NUM", "16")), 2),
        embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "4")),
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "60")),
        # Use the same credential rotation, cooldown, timeout, disabled-thinking setting,
        # and MiMo provider fallback as the main Agent. The stock LightRAG openai_complete
        # helper retries one credential internally and bypasses all of those safeguards.
        llm_model_func=llm_model_func,
        llm_model_name=llm_model_name,
        llm_model_kwargs={},
        default_embedding_timeout=int(os.getenv("EMBEDDING_TIMEOUT", "180")),
    )


def _selected_qa_model_uses_preferred_provider() -> bool:
    """Return whether the user-facing model route is the preferred provider.

    ``main.build_model`` treats an enabled preferred/unified route as atomic and
    authoritative even when the legacy ``MODEL_NAME`` differs. LightRAG must make the
    same choice so its ``llm_model_name`` and cache identity describe the model that is
    actually called.
    """

    try:
        from ..backend.research.model_runtime import preferred_provider_enabled
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.research.model_runtime import preferred_provider_enabled

    return preferred_provider_enabled()


def _preferred_provider_lightrag_model_func():
    """Adapt the configured preferred research channel to LightRAG Q&A."""

    try:
        from ..backend.research.model_runtime import (
            _is_invalid_provider_response_error,
            invoke_preferred_chat,
        )
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.research.model_runtime import (
            _is_invalid_provider_response_error,
            invoke_preferred_chat,
        )

    circuit_lock = threading.Lock()
    circuit_error: str | None = None

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        nonlocal circuit_error
        with circuit_lock:
            open_reason = circuit_error
        if open_reason:
            raise RuntimeError(f"LightRAG model circuit is open after provider exhaustion: {open_reason}")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        for item in history_messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(item.get("content") or "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": str(prompt or "")})
        try:
            # ``invoke_preferred_chat`` owns the single bounded gateway retry budget.
            # Retrying again here multiplies physical requests (for example 3 x 3 for
            # two retries at each layer) without changing the prompt or route.
            response = await asyncio.to_thread(
                invoke_preferred_chat,
                messages,
                timeout_seconds=float(os.getenv("CHAT_MODEL_TIMEOUT_SECONDS", "60")),
                call_kind="knowledge_qa",
            )
        except Exception as exc:
            if _is_invalid_provider_response_error(exc) or _should_open_lightrag_model_circuit(exc):
                with circuit_lock:
                    circuit_error = str(exc) or type(exc).__name__
                emit_event(
                    "knowledge_qa.lightrag_model.circuit_opened",
                    "tools",
                    "Stopped repeated LightRAG model calls after the preferred provider pool was exhausted.",
                    level="error",
                    data={"error_type": type(exc).__name__, "error": str(exc)[:500]},
                )
            raise
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or item.get("content") or "")
                if isinstance(item, dict)
                else str(item)
                for item in content
            )
        return str(content or "").strip()

    return complete


def _main_agent_lightrag_model_func():
    """Adapt the main Agent's synchronous rotating model to LightRAG's async callback.

    ``RotatingChatOpenAI`` intentionally implements rotation on the synchronous ``invoke`` /
    ``stream`` paths used by the main Agent. Calling ``ainvoke`` would fall back to LangChain's
    inherited async implementation and silently bypass rotation. ``asyncio.to_thread`` keeps
    LightRAG's event loop non-blocking while exercising the exact synchronous path that owns
    key rotation and MiMo fallback.
    """
    try:
        from ..main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    model = build_model(streaming=False)
    circuit_lock = threading.Lock()
    circuit_error: str | None = None

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        nonlocal circuit_error
        with circuit_lock:
            open_reason = circuit_error
        if open_reason:
            raise RuntimeError(f"LightRAG model circuit is open after provider exhaustion: {open_reason}")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        for item in history_messages or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(item.get("content") or "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": str(prompt or "")})
        try:
            response = await asyncio.to_thread(model.invoke, messages)
        except Exception as exc:
            if _should_open_lightrag_model_circuit(exc):
                with circuit_lock:
                    circuit_error = str(exc) or type(exc).__name__
                emit_event(
                    "knowledge_qa.lightrag_model.circuit_opened",
                    "tools",
                    "Stopped repeated LightRAG model calls after the main Agent provider pool was exhausted.",
                    level="error",
                    data={"error_type": type(exc).__name__, "error": str(exc)[:500]},
                )
            raise
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or item.get("content") or "")
                if isinstance(item, dict)
                else str(item)
                for item in content
            )
        return str(content or "").strip()

    return complete


def _should_open_lightrag_model_circuit(exc: Exception) -> bool:
    message = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in message
        for marker in (
            "429",
            "rate limit",
            "rate_limit",
            "quota",
            "timeout",
            "timed out",
        )
    )


async def _drop_existing_storage(rag: Any) -> None:
    for storage in (
        rag.full_docs,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.chunk_entity_relation_graph,
        rag.llm_response_cache,
        rag.doc_status,
    ):
        if storage:
            await storage.drop()


def _citations_from_lightrag_data(data: dict[str, Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in data.get("chunks", []) or []:
        path = str(chunk.get("file_path") or "")
        snippet = re.sub(r"\s+", " ", str(chunk.get("content") or "")).strip()
        key = (path, snippet[:80])
        if not path or key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "title": Path(path).stem,
                "path": path,
                "snippet": snippet[:500],
            }
        )
    if citations:
        return citations[:6]
    for ref in data.get("references", []) or []:
        path = str(ref.get("file_path") or "")
        if path:
            citations.append({"title": Path(path).stem, "path": path, "snippet": ""})
    return citations[:6]


def _retrieval_trace_from_lightrag_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve the final ordered chunks that were supplied to answer generation.

    LightRAG does not expose a comparable relevance score in this response, so the
    trace records the real rank and explicitly leaves score unset.
    """
    trace: list[dict[str, Any]] = []
    for rank, raw in enumerate(data.get("chunks", []) or [], start=1):
        if not isinstance(raw, dict):
            continue
        path = str(raw.get("file_path") or "")
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        trace.append(
            {
                "rank": rank,
                "chunk_id": str(raw.get("chunk_id") or ""),
                "reference_id": str(raw.get("reference_id") or ""),
                "title": Path(path).stem if path else "",
                "path": path,
                "content": content,
                "score": None,
                "score_type": "not_provided",
            }
        )
    return trace


def _run_async(coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    loop = _ensure_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result() or {}


def _ensure_async_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP, _ASYNC_LOOP_THREAD
    with _ASYNC_LOOP_LOCK:
        if _ASYNC_LOOP and _ASYNC_LOOP.is_running():
            return _ASYNC_LOOP
        ready = threading.Event()
        loop = asyncio.new_event_loop()

        def runner() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=runner, name="lightrag-async-loop", daemon=True)
        thread.start()
        ready.wait(timeout=5)
        _ASYNC_LOOP = loop
        _ASYNC_LOOP_THREAD = thread
        return loop
