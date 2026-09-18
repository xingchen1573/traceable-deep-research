"""Index service functions."""

from __future__ import annotations

import shutil

try:
    from ...observability import current_job_id
    from ...tools.knowledge_manager import delete_knowledge_base_index
    from ...tools.knowledge_qa import index_knowledge_base_lightrag
    from ...tools.lightrag_qa import build_research_lightrag_index
    from ...tools._registry import LIGHTRAG_WORKSPACES
except ImportError:  # pragma: no cover
    from observability import current_job_id
    from tools.knowledge_manager import delete_knowledge_base_index
    from tools.knowledge_qa import index_knowledge_base_lightrag
    from tools.lightrag_qa import build_research_lightrag_index
    from tools._registry import LIGHTRAG_WORKSPACES

from .common import invoke_tool
from .job_service import job_service
from . import knowledge_tree_service


def create_index_job(topic: str, *, force: bool = False, mode: str = "lightrag") -> dict:
    def target(_: dict) -> dict:
        job_id = current_job_id()
        if not job_id:
            raise RuntimeError("knowledge index job context is missing")
        job_service.update_step(
            job_id,
            "validate",
            "running",
            summary="正在校验知识库和索引参数",
        )
        if mode != "lightrag":
            raise ValueError("mode must be lightrag")
        job_service.update_step(
            job_id,
            "validate",
            "succeeded",
            summary="知识库和 LightRAG 参数校验通过",
        )
        job_service.update_step(
            job_id,
            "build",
            "running",
            summary="正在抽取实体与关系并写入向量、图谱索引",
        )
        result = invoke_tool(index_knowledge_base_lightrag, {"topic": topic, "force": force})
        job_service.update_step(
            job_id,
            "build",
            "succeeded",
            summary=f"已处理 {int(result.get('indexed_documents') or 0)} 篇文档",
        )
        job_service.update_step(
            job_id,
            "verify",
            "running",
            summary="正在校验索引元数据和必需存储文件",
        )
        if str(result.get("status") or "") not in {"indexed", "ready"}:
            raise RuntimeError(f"LightRAG returned unexpected index status: {result.get('status')}")
        job_service.update_step(
            job_id,
            "verify",
            "succeeded",
            summary="索引元数据、向量库和图谱文件均已就绪",
        )
        return result

    return job_service.create(
        job_type="knowledge_index",
        input_data={"topic": topic, "force": force, "mode": mode},
        target=target,
        steps=[
            {"id": "validate", "title": "校验知识库与参数"},
            {"id": "build", "title": "构建向量与图谱索引"},
            {"id": "verify", "title": "验证索引完整性"},
        ],
    )


def delete_index(topic: str) -> dict:
    return invoke_tool(delete_knowledge_base_index, {"topic": topic})


def create_research_index_job(research_id: str, *, force: bool = False) -> dict:
    """Index exactly one ResearchItem using its stable ID as workspace identity."""

    research = knowledge_tree_service.get_research_node(research_id)

    def target(_: dict) -> dict:
        job_id = current_job_id()
        if not job_id:
            raise RuntimeError("research index job context is missing")
        knowledge_tree_service.update_research_index_status(research_id, "running", job_id=job_id)
        job_service.update_step(job_id, "validate", "running", summary="正在校验 ResearchItem 文档")
        current = knowledge_tree_service.get_research_node(research_id)
        job_service.update_step(job_id, "validate", "succeeded", summary="ResearchItem 文档与 manifest 校验通过")
        job_service.update_step(job_id, "build", "running", summary="正在构建独立 LightRAG 索引")
        try:
            result = build_research_lightrag_index(
                research_id,
                current["absolute_path"],
                force=force,
            )
        except Exception as exc:
            knowledge_tree_service.update_research_index_status(
                research_id,
                "failed",
                job_id=job_id,
                error=str(exc) or type(exc).__name__,
            )
            raise
        job_service.update_step(
            job_id,
            "build",
            "succeeded",
            summary=f"已索引 {int(result.get('indexed_documents') or 0)} 篇文档",
        )
        job_service.update_step(job_id, "verify", "running", summary="正在验证索引完整性")
        if str(result.get("status") or "") not in {"indexed", "ready"}:
            raise RuntimeError(f"LightRAG returned unexpected research index status: {result.get('status')}")
        knowledge_tree_service.update_research_index_status(
            research_id,
            "ready",
            job_id=job_id,
            workspace=result.get("workspace"),
            indexed_documents=result.get("indexed_documents"),
        )
        job_service.update_step(job_id, "verify", "succeeded", summary="ResearchItem 索引已就绪")
        return result

    return job_service.create(
        job_type="research_index",
        input_data={"research_id": research_id, "path": research["path"], "force": force, "mode": "lightrag"},
        target=target,
        steps=[
            {"id": "validate", "title": "校验 ResearchItem"},
            {"id": "build", "title": "构建独立索引"},
            {"id": "verify", "title": "验证索引完整性"},
        ],
    )


def delete_research_index(research_id: str) -> dict:
    """Delete one rebuildable ResearchItem workspace and reset its manifest state."""

    knowledge_tree_service.get_research_node(research_id)
    workspace = (LIGHTRAG_WORKSPACES / "research" / research_id).resolve()
    research_root = (LIGHTRAG_WORKSPACES / "research").resolve()
    if workspace.parent != research_root:
        raise ValueError("invalid research workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
    knowledge_tree_service.update_research_index_status(research_id, "pending")
    return {
        "status": "deleted",
        "research_id": research_id,
        "deleted_path": str(workspace.relative_to(LIGHTRAG_WORKSPACES.parent)),
    }
