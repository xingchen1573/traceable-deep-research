"""Restricted worker agents exposed to the main agent as tools."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# invoke_research_chat remains imported lazily inside each call site below. This preserves the
# established main.py/workers.py import boundary and keeps worker module import independent from
# research runtime initialization.


def _submit_with_audit_context(executor: Any, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    from backend.research.audit import submit_with_audit_context

    return submit_with_audit_context(executor, fn, *args, **kwargs)


def _future_result_after_soft_timeout(
    future: Any,
    *,
    timeout_seconds: float,
    on_soft_timeout: Callable[[], None],
) -> Any:
    """Wait past an orchestration threshold without orphaning an in-flight model call.

    Python cannot cancel a thread that is already inside an HTTP request. Treating this wrapper
    threshold as a hard failure used to discard a result while that exact request continued in
    the background. Individual model calls still have provider timeouts; this outer threshold is
    therefore diagnostic only, and a slow in-flight call is joined before the pipeline decides
    whether extraction actually succeeded.
    """

    try:
        return future.result(timeout=max(0.001, timeout_seconds))
    except TimeoutError:
        if not future.done():
            on_soft_timeout()
        # If the task itself raised TimeoutError this immediately re-raises it. Otherwise it
        # joins the still-running call so its eventual result cannot leak into the background.
        return future.result()


def _audit_model_scope(*, stage: str, question_id: str | None = None, section_id: str | None = None):
    from backend.research.audit import audit_scope

    return audit_scope(stage=stage, question_id=question_id, section_id=section_id)


def _record_model_parse(response: Any, *, parsed_result: Any = None, error: Any = None) -> None:
    from backend.research.audit import record_parse_result

    metadata = getattr(response, "response_metadata", None) or {}
    call_id = metadata.get("harness_call_id") if isinstance(metadata, dict) else None
    record_parse_result(
        call_id,
        parsed_result=parsed_result,
        status="parse_error" if error is not None else "success",
        error=error,
    )

try:
    from .middleware import ContextSnipMiddleware, LoopGuardMiddleware
    from .observability import emit_event
    from .prompts import (
        KNOWLEDGE_MANAGER_AGENT_PROMPT,
        REPO_WORKER_PROMPT,
        WEB_WORKER_EXTRACT_PROMPT,
        WEB_WORKER_PROMPT,
    )
    from .tools import (
        bind_channel_to_topic_tool,
        bind_channel_to_scope_tool,
        broad_discovery,
        check_knowledge_base_index,
        docs_search,
        evaluate_knowledge_base_qa,
        fetch_url,
        fetch_url_with_chain,
        get_channel_push_config_tool,
        get_channel_topic_binding_tool,
        index_knowledge_base_lightrag,
        inspect_knowledge_base,
        inspect_theme_tool,
        list_personal_knowledge_tree_tool,
        list_files,
        list_knowledge_bases,
        list_knowledge_base_status,
        query_knowledge_base,
        read_file,
        search_web,
        set_active_knowledge_base,
        set_channel_push_config_tool,
        preview_knowledge_tree_change_tool,
        preview_research_archive_tool,
        query_personal_knowledge_scope_tool,
        summarize_knowledge_base,
        unbind_channel_topic_tool,
    )
    from .tools._registry import current_thread_id
except ImportError:  # pragma: no cover - script mode fallback
    from middleware import ContextSnipMiddleware, LoopGuardMiddleware
    from observability import emit_event
    from prompts import (
        KNOWLEDGE_MANAGER_AGENT_PROMPT,
        REPO_WORKER_PROMPT,
        WEB_WORKER_EXTRACT_PROMPT,
        WEB_WORKER_PROMPT,
    )
    from tools import (
        bind_channel_to_topic_tool,
        bind_channel_to_scope_tool,
        broad_discovery,
        check_knowledge_base_index,
        docs_search,
        evaluate_knowledge_base_qa,
        fetch_url,
        fetch_url_with_chain,
        get_channel_push_config_tool,
        get_channel_topic_binding_tool,
        index_knowledge_base_lightrag,
        inspect_knowledge_base,
        inspect_theme_tool,
        list_personal_knowledge_tree_tool,
        list_files,
        list_knowledge_bases,
        list_knowledge_base_status,
        query_knowledge_base,
        read_file,
        search_web,
        set_active_knowledge_base,
        set_channel_push_config_tool,
        preview_knowledge_tree_change_tool,
        preview_research_archive_tool,
        query_personal_knowledge_scope_tool,
        summarize_knowledge_base,
        unbind_channel_topic_tool,
    )
    from tools._registry import current_thread_id


MAX_QUERIES = 5
MAX_URLS = 5
MAX_FETCH_URLS = 8
MAX_SOURCES = 20
MAX_OUTPUT_SOURCES = 10
MAX_OUTPUT_DOCUMENTS = 5
MAX_OUTPUT_EVIDENCES = 8
MAX_OUTPUT_KEY_FACTS = 6
MAX_OUTPUT_EVIDENCE_PACK = 10
WEB_SYNTHESIS_SNIP_CHARS = 18000
WEB_SYNTHESIS_INPUT_CHARS = 24000
SECTION_EVIDENCE_INPUT_CHARS = 9000
SECTION_EVIDENCE_MAX_SECTIONS = 6
SECTION_EVIDENCE_MAX_CHUNKS = 10
SECTION_EVIDENCE_MODEL_CHUNKS = 4
SECTION_EVIDENCE_MODEL_TIMEOUT_SECONDS = 1200
QUESTION_EVIDENCE_MODEL_CHUNKS = 3
QUESTION_EVIDENCE_TARGET = 4
QUESTION_EVIDENCE_TIMEOUT_SECONDS = 1200
# Batch-extraction item cap scales with how many chunks were actually sent to the model,
# instead of a flat number -- a flat cap silently discarded output once "exhaust every chunk"
# (the prompt instruction) started legitimately returning more than one evidence item per chunk.
EVIDENCE_ITEMS_PER_CHUNK = 6
HYBRID_RERANK_CANDIDATES = 18
HYBRID_RERANK_ZERO_LEXICAL_CANDIDATES = 8
HYBRID_RERANK_INPUT_CHARS = 500
RRF_CANDIDATES_PER_CHANNEL = 24
RRF_DEFAULT_K = 60
DOCUMENT_EXCERPT_CHARS = 2600
MATERIAL_EXCERPT_CHARS = 900
CHUNK_CHARS = 1300
CHUNK_OVERLAP_CHARS = 180
# Layer-1 coarse selection no longer hands every question a flat 3/5-chunk quota regardless of
# how many relevant candidates actually exist -- thin material naturally yields fewer picks
# (nothing pads it out), abundant material scales up to this ceiling instead of being clipped
# at the old fixed value.
CHUNK_CANDIDATES_PER_QUESTION_CAP = 8
DEFAULT_WEB_RESEARCH_ROUNDS = 2

_WEB_RESEARCH_ROUNDS: dict[tuple[str, str], int] = {}


def _evidence_semantic_arbitration_mode() -> str:
    value = os.getenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_MODE", "off").strip().lower()
    return value if value in {"off", "shadow", "live"} else "off"


def _model(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        streaming=streaming,
        timeout=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
    )


def _evidence_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("RESEARCH_EVIDENCE_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        api_key=os.getenv("RESEARCH_EVIDENCE_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("RESEARCH_EVIDENCE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        timeout=float(os.getenv("RESEARCH_EVIDENCE_TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "180"))),
    )


def _worker_model() -> ChatOpenAI:
    """Build the cheap model used for the one-shot Repo Reader Worker.

    repo_worker_tool has a narrow, bounded, read-only task scope -- see
    docs/modules/lite-model-routing-v0.1.md -- so a lighter model is an acceptable cost/latency
    trade here, unlike build_agent()'s main chat model which drives free-form multi-turn tool
    routing. Independent WORKER_* env vars (not reusing RECALL_*/DREAM_*) so this role can be
    retuned (timeout, model) without affecting the unrelated long-term-memory-recall and
    AutoDream-consolidation roles, matching the existing one-role-per-concern pattern
    (RECALL_*/DREAM_*/RESEARCH_EVIDENCE_*/RESEARCH_WRITER_*).
    """
    return ChatOpenAI(
        model=os.getenv("WORKER_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        api_key=os.getenv("WORKER_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("WORKER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
        timeout=float(os.getenv("WORKER_MODEL_TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "180"))),
    )


def _main_chat_model() -> ChatOpenAI:
    """Reuse the main user-facing model route for knowledge management.

    Import lazily because ``main.py`` imports this module while constructing the main Agent.
    ``main.build_model`` owns the atomic model/base URL/key selection for the system
    unified/preferred provider, so the main Agent, KB QA Agent and Knowledge Manager cannot drift
    onto different gateways again.
    """
    try:
        from .main import build_model
    except ImportError:  # pragma: no cover - script mode fallback
        from main import build_model

    return build_model(streaming=False)


_DEFAULT_EVIDENCE_MODEL_FACTORY = _evidence_model


def _invoke_evidence_chat(messages: list[dict[str, Any]], *, timeout_seconds: float | None = None) -> Any:
    from backend.research.model_runtime import invoke_research_chat

    # Preserve the established model-injection seam used by tests and local adapters.
    if _evidence_model is not _DEFAULT_EVIDENCE_MODEL_FACTORY:
        return _evidence_model().invoke(messages)
    return invoke_research_chat(
        messages,
        timeout_seconds=timeout_seconds or float(os.getenv("RESEARCH_EVIDENCE_TIMEOUT_SECONDS", os.getenv("MODEL_TIMEOUT_SECONDS", "180"))),
        evidence=True,
    )


def _configured_evidence_route() -> dict[str, Any]:
    """Return non-secret configured routing identity for extraction diagnostics."""

    from backend.research.model_runtime import (
        preferred_provider_base_url,
        preferred_provider_enabled,
        preferred_provider_model_name,
    )

    primary_model = os.getenv("RESEARCH_EVIDENCE_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4.1-mini")
    primary_base_url = os.getenv("RESEARCH_EVIDENCE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if preferred_provider_enabled():
        return {
            "configured_route": "preferred_then_primary",
            "model": preferred_provider_model_name(),
            "base_url": preferred_provider_base_url(),
            "primary_model": primary_model,
            "primary_base_url": primary_base_url,
        }
    return {
        "configured_route": "primary",
        "model": primary_model,
        "base_url": primary_base_url,
        "primary_model": primary_model,
        "primary_base_url": primary_base_url,
    }


def arbitrate_evidence_semantics(
    *,
    question_id: str,
    section_id: str,
    question: str,
    purpose: str,
    criteria: list[dict[str, Any]],
    target_subject: str,
    comparison_subjects: list[str],
    comparison_axis: str,
    evidence: list[dict[str, Any]],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Resolve one Question+Round batch of semantic conflicts with a bounded JSON call."""
    support_relations = {"direct_support", "inference_support", "context_only", "unrelated"}
    target_fits = {"exact_target", "related_target", "general"}
    confidence_values = {"high", "medium", "low"}
    batch: list[dict[str, Any]] = []
    expected_ids: list[str] = []
    for item in evidence:
        arbitration = item.get("semantic_arbitration") or {}
        evidence_id = str(arbitration.get("evidence_id") or "")
        if not evidence_id or not arbitration.get("conflict_reasons"):
            continue
        expected_ids.append(evidence_id)
        batch.append({
            "evidence_id": evidence_id,
            "claim": item.get("claim") or item.get("atomic_claim"),
            "quote": item.get("raw_quote") or item.get("evidence_text"),
            "quote_context": arbitration.get("quote_context"),
            "subject": item.get("subject"),
            "subject_relation": item.get("subject_relation"),
            "criterion_ids": item.get("criterion_ids") or [],
            "initial_support_relation": (arbitration.get("initial") or {}).get("support_relation"),
            "initial_target_fit": (arbitration.get("initial") or {}).get("target_fit"),
            "initial_reason": (arbitration.get("initial") or {}).get("reason"),
            "program_diagnostics": {
                **(arbitration.get("program_signals") or {}),
                "conflict_reasons": arbitration.get("conflict_reasons") or [],
            },
        })
    if not batch:
        return {"status": "not_needed", "verdicts": {}, "reason": ""}
    payload = {
        "task": "Arbitrate semantic evidence relations for one research question and round. Return JSON only.",
        "question": question,
        "purpose": purpose,
        "criteria": criteria,
        "target_subject": target_subject,
        "allowed_comparison_subjects": comparison_subjects,
        "comparison_axis": comparison_axis,
        "evidence": batch,
        "rules": [
            "Use only the supplied question, criteria, quote, and bounded quote_context. Do not use external knowledge.",
            "Do not invent or modify quotes, subjects, criteria, comparison subjects, or axes.",
            "direct_support means the quote itself directly supports the current criterion without cross-target or cross-domain inference.",
            "inference_support requires at least one reasoning step; context_only is background; unrelated does not answer the question or criterion.",
            "exact_target means the quote directly discusses the requested target or an explicitly allowed comparison subject; related_target is a related object/component/domain; general is general knowledge.",
            "Return exactly one verdict for every supplied evidence_id and no unknown or duplicate IDs.",
            "unrelated requires criterion_supported=false. direct_support requires requires_inference=false.",
            "reason must be concrete and at most 160 characters.",
            "Return shape: {\"verdicts\":[{\"evidence_id\":\"candidate_001\",\"support_relation\":\"direct_support|inference_support|context_only|unrelated\",\"target_fit\":\"exact_target|related_target|general\",\"criterion_supported\":true,\"requires_inference\":false,\"reason\":\"...\",\"confidence\":\"high|medium|low\"}]}",
        ],
    }
    with _audit_model_scope(
        stage="evidence_semantic_arbitration",
        question_id=question_id or None,
        section_id=section_id or None,
    ):
        try:
            response = _invoke_evidence_chat(
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                timeout_seconds=timeout_seconds or float(
                    os.getenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_TIMEOUT_SECONDS", "45")
                ),
            )
        except TimeoutError as exc:
            return {"status": "unavailable", "verdicts": {}, "reason": f"timeout:{exc}"[:300]}
        except Exception as exc:
            return {"status": "unavailable", "verdicts": {}, "reason": f"provider_error:{exc}"[:300]}
        call_id = str((getattr(response, "response_metadata", None) or {}).get("harness_call_id") or "") or None
        try:
            data = _json_from_text(_content_from_response(response))
            rows = data.get("verdicts") or data.get("evidence") or []
            if not isinstance(rows, list):
                raise ValueError("verdicts must be a list")
            verdicts: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("verdict row must be an object")
                evidence_id = str(row.get("evidence_id") or "")
                support_relation = str(row.get("support_relation") or "")
                target_fit = str(row.get("target_fit") or "")
                confidence = str(row.get("confidence") or "").lower()
                criterion_supported = row.get("criterion_supported")
                requires_inference = row.get("requires_inference")
                reason = str(row.get("reason") or "").strip()
                if evidence_id not in expected_ids or evidence_id in verdicts:
                    raise ValueError(f"unknown or duplicate evidence_id: {evidence_id}")
                if support_relation not in support_relations or target_fit not in target_fits:
                    raise ValueError(f"invalid semantic enum for {evidence_id}")
                if not isinstance(criterion_supported, bool) or not isinstance(requires_inference, bool):
                    raise ValueError(f"invalid boolean contract for {evidence_id}")
                if support_relation == "unrelated" and criterion_supported:
                    raise ValueError(f"unrelated cannot support criterion: {evidence_id}")
                if support_relation == "direct_support" and requires_inference:
                    raise ValueError(f"direct_support cannot require inference: {evidence_id}")
                if confidence not in confidence_values or not reason or len(reason) > 160:
                    raise ValueError(f"invalid confidence or reason for {evidence_id}")
                verdicts[evidence_id] = {
                    "support_relation": support_relation,
                    "target_fit": target_fit,
                    "criterion_supported": criterion_supported,
                    "requires_inference": requires_inference,
                    "reason": reason,
                    "confidence": confidence,
                }
            if set(verdicts) != set(expected_ids):
                missing = sorted(set(expected_ids) - set(verdicts))
                raise ValueError(f"missing verdict evidence_ids: {missing}")
        except Exception as exc:
            _record_model_parse(response, error=exc)
            return {"status": "invalid_response", "verdicts": {}, "reason": str(exc)[:300]}
        _record_model_parse(response, parsed_result={
            "verdicts": [
                {"evidence_id": evidence_id, **verdict}
                for evidence_id, verdict in verdicts.items()
            ],
        })
        return {"status": "resolved", "verdicts": verdicts, "reason": ""}


def _last_text(result: Any) -> str:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return str(result)


@tool
def repo_worker_tool(task: str) -> str:
    """将狭窄的本地仓库只读检查任务委派给 Repo Reader Worker。

    当主 Agent 需要检查当前仓库根目录下的本地文件时使用此工具，
    例如列出文件、读取特定文件、检查提示词/配置/函数，或在回答前收集文件证据。

    输入：
    - 简洁、具体的本地文件读取任务。
    - 尽可能包含目标路径、文件名、符号、关键词或需要验证的具体问题。

    输出：
    - JSON 字符串，包含 findings、evidence_paths、uncertainties 和 suggested_next_step。
    """
    denied_reason = _repo_task_denied_reason(task)
    if denied_reason:
        output = json.dumps(
            {
                "worker": "repo_worker",
                "status": "denied",
                "findings": [],
                "evidence_paths": [],
                "uncertainties": [denied_reason],
                "suggested_next_step": "Use web_worker_tool only for this source_mode=web task.",
            },
            ensure_ascii=False,
        )
        emit_event(
            "worker.repo.denied",
            "workers",
            "Repo worker denied by source-mode constraint.",
            data={"reason": denied_reason, "task_chars": len(task)},
        )
        return output

    emit_event(
        "worker.repo.started",
        "workers",
        "Repo worker started.",
        data={"task_chars": len(task)},
    )
    worker = create_agent(
        model=_worker_model(),
        tools=[list_files, read_file],
        system_prompt=REPO_WORKER_PROMPT,
        middleware=[ContextSnipMiddleware(max_chars=4000), LoopGuardMiddleware(max_repeats=1)],
    )
    result = worker.invoke({"messages": [{"role": "user", "content": task[:1000]}]})
    text = _last_text(result)
    output = _ensure_jsonish("repo_worker", text)
    emit_event(
        "worker.repo.finished",
        "workers",
        "Repo worker finished.",
        data={"output_chars": len(output)},
    )
    return output


@tool
def web_worker_tool(task_or_url: str) -> str:
    """将联网检索任务委派给 Web Worker，内部支持轻量 DeepSearch 管线。

    当主 Agent 需要外部信息、最新事实、官方文档、技术参考资料、论文、政策、新闻、
    产品/版本信息或来自特定 URL 的证据时使用此工具。

    内部行为：
    1. 提取 retrieval plan：queries、urls、capabilities、must_fetch。
    2. 执行 broad_discovery / docs_search / url_fetch。
    3. 对来源去重、抓取正文、抽取 evidence 并整理 facts。

    输入：
    - 聚焦的搜索查询或明确的 URL，以及需要提取的具体信息。
    - 复杂任务会被自动拆成多个子查询并并发执行。

    输出：
    - JSON 字符串，包含 sources、documents、evidences、key_facts、gaps、provider_attempts 和 suggested_next_step。
    """
    round_key = _web_round_key(task_or_url)
    max_rounds = _max_web_rounds(task_or_url)
    if round_key:
        thread_key = (current_thread_id(), round_key)
        used_rounds = _WEB_RESEARCH_ROUNDS.get(thread_key, 0)
        if used_rounds >= max_rounds:
            output = json.dumps(
                {
                    "worker": "web_worker",
                    "status": "round_limit_reached",
                    "sources": [],
                    "documents": [],
                    "evidences": [],
                    "key_facts": [],
                    "evidence_pack": [],
                    "gaps": [
                        f"Web research round limit reached for this document-set task ({used_rounds}/{max_rounds})."
                    ],
                    "suggested_next_step": (
                        "Call research_coverage_check with the evidence already collected, then write with "
                        "save_knowledge_set or mark remaining gaps."
                    ),
                },
                ensure_ascii=False,
            )
            emit_event(
                "worker.web.round_limit",
                "workers",
                "Web worker denied by research round limit.",
                data={"thread_id": current_thread_id(), "round_key": round_key, "used_rounds": used_rounds},
            )
            return output
        _WEB_RESEARCH_ROUNDS[thread_key] = used_rounds + 1

    emit_event(
        "worker.web.started",
        "workers",
        "Web worker started (concurrent).",
        data={"task_chars": len(task_or_url)},
    )

    # Stage 1: Extract retrieval plan
    research_payload = _extract_research_payload(task_or_url)
    plan = _retrieval_plan_from_payload(research_payload) or _extract_retrieval_plan(task_or_url)
    query_limit = _task_int_setting(task_or_url, "query_limit", MAX_QUERIES)
    fetch_limit = _task_int_setting(task_or_url, "fetch_url_limit", MAX_FETCH_URLS)
    retrieval_mode = _task_text_setting(task_or_url, "retrieval_mode") or "full"
    queries = plan.get("queries", [])[:query_limit]
    urls = plan.get("urls", [])[:MAX_URLS]
    capabilities = plan.get("required_capabilities", [])
    must_fetch = bool(plan.get("must_fetch", True))
    if retrieval_mode == "explicit_fetch":
        urls = _dedupe_texts(re.findall(r"https?://[^\s)>\"]+", task_or_url))[:fetch_limit]
        capabilities = ["url_fetch"]
        must_fetch = True

    emit_event(
        "retrieval.plan.extracted",
        "workers",
        "Extracted retrieval plan.",
        data={
            "query_count": len(queries),
            "url_count": len(urls),
            "capabilities": capabilities,
            "must_fetch": must_fetch,
        },
    )

    # Stage 2: Discovery and docs search
    sources: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    provider_attempts: list[dict[str, Any]] = []

    if retrieval_mode == "material_extract":
        sources = [item for item in research_payload.get("source_candidates") or [] if isinstance(item, dict)]
        documents = [item for item in research_payload.get("documents") or [] if isinstance(item, dict)]
        provider_attempts = [item for item in research_payload.get("provider_attempts") or [] if isinstance(item, dict)]
    elif queries and "docs_search" in capabilities:
        docs_results = _batch_docs_search_sync(queries)
        for result in docs_results:
            sources.extend(result.get("sources", []))
            documents.extend(result.get("documents", []))
            provider_attempts.extend(result.get("provider_attempts", []))

    if retrieval_mode != "material_extract" and queries and "broad_discovery" in capabilities:
        discovery_results = _batch_discovery_sync(queries)
        for result in discovery_results:
            sources.extend(result.get("sources", []))
            provider_attempts.extend(result.get("provider_attempts", []))

    sources = _dedupe_sources(sources)[:MAX_SOURCES]
    emit_event(
        "retrieval.sources.discovered",
        "workers",
        "Discovered and deduplicated retrieval sources.",
        data={"source_count": len(sources)},
    )

    if retrieval_mode == "discovery_only":
        output = json.dumps({
            "worker": "web_worker",
            "status": "discovered",
            "sources": sources,
            "documents": [],
            "materials": [],
            "selected_chunks": [],
            "evidence_map": [],
            "provider_attempts": provider_attempts,
            "attempted_urls": [],
            "_diagnostics": {
                "retrieval_mode": retrieval_mode,
                "candidate_source_count": len(sources),
                "model_call_count": 0,
            },
        }, ensure_ascii=False)
        emit_event(
            "worker.web.finished",
            "workers",
            "Web worker finished discovery-only mode.",
            data={"source_count": len(sources), "output_chars": len(output)},
        )
        return output

    # Stage 3: Fetch explicit and discovered URLs
    candidate_urls = (
        [str(item.get("url") or "") for item in documents if str(item.get("url") or "").startswith("http")]
        if retrieval_mode == "material_extract"
        else _select_fetch_urls(
            urls,
            sources,
            max_urls=fetch_limit,
            preferred_source_types=plan.get("preferred_source_types") or [],
        )
    )
    if retrieval_mode != "material_extract" and candidate_urls and (must_fetch or urls):
        fetch_results = _batch_fetch_documents_sync(candidate_urls)
        for result in fetch_results:
            document = result.get("document")
            if document:
                documents.append(document)
            provider_attempts.extend(result.get("provider_attempts", []))

    documents = _dedupe_documents(documents)
    emit_event(
        "retrieval.documents.fetched",
        "workers",
        "Fetched retrieval documents.",
        data={
            "url_count": len(candidate_urls),
            "document_count": len(documents),
            "document_content_chars": sum(len(str(doc.get("content") or "")) for doc in documents),
            "document_qualities": _count_by(documents, "content_quality"),
        },
    )

    if not sources and not documents:
        output = _ensure_jsonish(
            "web_worker",
            "未能获取任何候选来源或网页正文。任务可能过于狭窄、来源不足，或网络/API 暂时不可用。",
        )
        emit_event(
            "worker.web.finished",
            "workers",
            "Web worker finished with no results.",
            data={"output_chars": len(output)},
        )
        return output

    # Stage 4: Build auditable material pool and select chunks
    research_sections = _extract_research_sections(task_or_url)
    materials = _build_materials(documents)
    selected_chunks = _select_research_chunks(materials, research_sections, task_or_url)
    emit_event(
        "retrieval.material_pool.built",
        "workers",
        "Built material pool and selected research chunks.",
        data={
            "material_count": len(materials),
            "selected_chunk_count": len(selected_chunks),
            "selected_chunk_chars": sum(len(str(item.get("text") or "")) for item in selected_chunks),
        },
    )
    context_text = _build_retrieval_context(
        task=task_or_url,
        plan=plan,
        sources=sources,
        documents=documents,
        provider_attempts=provider_attempts,
        selected_chunks=selected_chunks,
    )
    input_text = context_text[:WEB_SYNTHESIS_INPUT_CHARS]
    _dump_web_synthesis_debug(
        task=task_or_url,
        context_text=context_text,
        input_text=input_text,
        selected_chunks=selected_chunks,
        documents=documents,
        sources=sources,
    )
    pre_diagnostics = _retrieval_pre_diagnostics(
        sources=sources,
        documents=documents,
        provider_attempts=provider_attempts,
        context_text=context_text,
        input_text=input_text,
    )
    pre_diagnostics["material_pool"] = {
        "material_count": len(materials),
        "selected_chunk_count": len(selected_chunks),
        "selected_chunk_chars": sum(len(str(item.get("text") or "")) for item in selected_chunks),
        "chunk_ranker": _selected_chunk_ranker_diagnostics(selected_chunks),
    }
    synthesis_started = time.perf_counter()
    text = _global_synthesis_or_empty(input_text)
    synthesis_elapsed_ms = int((time.perf_counter() - synthesis_started) * 1000)
    pre_diagnostics["global_synthesis"] = {
        "mode": "direct_model" if os.getenv("RESEARCH_MODEL_GLOBAL_SYNTHESIS", "0") == "1" else "deterministic_skip",
        "elapsed_ms": synthesis_elapsed_ms,
        "model_input_chars": len(input_text),
        "raw_model_output_chars": len(text),
    }
    emit_event(
        "retrieval.global_synthesis.finished",
        "workers",
        "Finished direct web synthesis.",
        data={
            "elapsed_ms": synthesis_elapsed_ms,
            "model_input_chars": len(input_text),
            "raw_model_output_chars": len(text),
        },
    )
    section_started = time.perf_counter()
    section_evidence, section_evidence_stats = _extract_section_evidence_maps(
        sections=research_sections,
        selected_chunks=selected_chunks,
        task=task_or_url,
    )
    pre_diagnostics["section_evidence_extraction"] = {
        "elapsed_ms": int((time.perf_counter() - section_started) * 1000),
        "section_count": len(research_sections),
        "evidence_count": len(section_evidence),
        **section_evidence_stats,
    }
    _write_research_trace(
        task=task_or_url,
        materials=materials,
        selected_chunks=selected_chunks,
        evidence=section_evidence,
        diagnostics=pre_diagnostics,
    )
    output = _finalize_retrieval_output(
        text,
        sources,
        documents,
        provider_attempts,
        materials=materials,
        selected_chunks=selected_chunks,
        research_sections=research_sections,
        section_evidence=section_evidence,
        section_evidence_stats=section_evidence_stats,
        attempted_urls=candidate_urls,
    )
    output = _attach_retrieval_diagnostics(output, pre_diagnostics, text)
    emit_event(
        "retrieval.evidence.extracted",
        "workers",
        "Synthesized retrieval evidence.",
        data={
            "output_chars": len(output),
            "raw_model_output_chars": len(text),
            "context_chars": len(context_text),
            "model_input_chars": len(input_text),
        },
    )
    emit_event(
        "worker.web.finished",
        "workers",
        "Web worker finished.",
        data={"output_chars": len(output)},
    )
    return output


def _write_research_trace(
    *,
    task: str,
    materials: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    if os.getenv("RESEARCH_TRACE_JSONL", "1") == "0":
        return
    key = _safe_trace_key(_web_round_key(task) or "web_research")
    trace_dir = Path(os.getenv("RESEARCH_TRACE_DIR", "logs/research_jobs")) / key
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(trace_dir / "materials.jsonl", [_strip_internal_fields(item) for item in materials])
        _append_jsonl(trace_dir / "selected_chunks.jsonl", selected_chunks)
        _append_jsonl(trace_dir / "evidence_map.jsonl", evidence)
        calls = []
        global_synthesis = diagnostics.get("global_synthesis")
        if isinstance(global_synthesis, dict):
            calls.append({"stage": "global_synthesis", **global_synthesis})
        section_stats = (diagnostics.get("section_evidence_extraction") or {}).get("sections")
        if isinstance(section_stats, list):
            calls.extend({"stage": "section_evidence", **item} for item in section_stats if isinstance(item, dict))
        _append_jsonl(trace_dir / "model_calls.jsonl", calls)
    except Exception as exc:  # pragma: no cover - trace best effort
        emit_event(
            "retrieval.trace.write_failed",
            "workers",
            "Failed to write research trace JSONL.",
            level="warning",
            data={"error": str(exc)[:300]},
        )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"timestamp": stamp, **row}, ensure_ascii=False) + "\n")


def _strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not str(key).startswith("_")}


def _safe_trace_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", str(value))[:100] or "web_research"


def _extract_retrieval_plan(task: str) -> dict[str, Any]:
    """Use a cheap model call to extract independent queries, URLs, and capabilities."""
    from backend.research.model_runtime import invoke_research_chat

    model = _model()
    prompt = (
        WEB_WORKER_EXTRACT_PROMPT
        + "\n\n任务描述：\n"
        + task[:800]
        + "\n\n只返回 JSON，不要解释。"
    )
    response = invoke_research_chat(
        [{"role": "user", "content": prompt}],
        timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
    )
    text = _content_from_response(response)
    data = _json_from_text(text)
    return _normalize_retrieval_plan(task, data)


def _retrieval_plan_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query_plan = payload.get("query_plan") if isinstance(payload, dict) else {}
    if not isinstance(query_plan, dict):
        return {}
    raw_queries = query_plan.get("queries")
    if not isinstance(raw_queries, list):
        return {}
    queries = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if query:
            queries.append(query)
    if not queries:
        return {}
    raw_sections = ((payload.get("research_plan") or {}).get("sections") or []) if isinstance(payload.get("research_plan"), dict) else []
    preferred_source_types: list[str] = []
    for section in raw_sections:
        if not isinstance(section, dict):
            continue
        preferred_source_types.extend(_coerce_str_list(section.get("preferred_source_types")))
        for requirement in section.get("source_requirements") or []:
            if isinstance(requirement, dict) and requirement.get("source_type"):
                preferred_source_types.append(str(requirement["source_type"]))
    return {
        "task_type": "research",
        "queries": _dedupe_texts(queries)[: max(MAX_QUERIES, int(query_plan.get("query_limit") or 0))],
        "urls": [],
        "required_capabilities": ["docs_search", "broad_discovery"],
        "must_fetch": True,
        "from_research_payload": True,
        "preferred_source_types": _dedupe_texts(preferred_source_types),
    }


def _dedupe_texts(items: list[str]) -> list[str]:
    output = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _direct_web_synthesis(input_text: str) -> str:
    from backend.research.model_runtime import invoke_research_chat

    prompt = (
        WEB_WORKER_PROMPT
        + "\n\n约束补充：只允许基于用户消息中的 Source、Document、Selected Chunks 输出 JSON；"
        + "不要新增来源 URL，不要推断 provider/status，不要把未出现的事实写成证据。"
        + "\n\n"
        + input_text[:WEB_SYNTHESIS_INPUT_CHARS]
    )
    response = invoke_research_chat(
        [{"role": "user", "content": prompt}],
        timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "180")),
    )
    return _content_from_response(response)


def _global_synthesis_or_empty(input_text: str) -> str:
    if os.getenv("RESEARCH_MODEL_GLOBAL_SYNTHESIS", "0") != "1":
        return json.dumps(
            {
                "worker": "web_worker",
                "status": "completed",
                "evidences": [],
                "key_facts": [],
                "evidence_pack": [],
                "gaps": [],
                "suggested_next_step": None,
            },
            ensure_ascii=False,
        )
    return _direct_web_synthesis(input_text)


def _normalize_retrieval_plan(task: str, data: dict[str, Any]) -> dict[str, Any]:
    queries = _coerce_str_list(data.get("queries"))[:MAX_QUERIES]
    urls = _coerce_str_list(data.get("urls"))[:MAX_URLS]
    capabilities = _coerce_str_list(data.get("required_capabilities"))
    task_type = str(data.get("task_type") or "research").strip() or "research"
    must_fetch = data.get("must_fetch")

    if not queries and not urls:
        found_urls = re.findall(r"https?://[^\s)>\"]+", task)
        if found_urls:
            urls = found_urls[:MAX_URLS]
            query_text = re.sub(r"https?://[^\s)>\"]+", "", task).strip()
            if query_text:
                queries = [query_text]
        else:
            queries = [task.strip()]

    if not capabilities:
        capabilities = []
        if _looks_technical(task):
            capabilities.append("docs_search")
        if queries:
            capabilities.append("broad_discovery")
        if urls:
            capabilities.append("url_fetch")

    allowed = {"broad_discovery", "docs_search", "url_fetch"}
    capabilities = [item for item in capabilities if item in allowed]
    if queries and not any(item in capabilities for item in ["broad_discovery", "docs_search"]):
        capabilities.append("broad_discovery")
    if queries and "docs_search" in capabilities and "broad_discovery" not in capabilities:
        capabilities.append("broad_discovery")
    if urls and "url_fetch" not in capabilities:
        capabilities.append("url_fetch")

    if must_fetch is None:
        must_fetch = bool(urls or "研究" in task or "调研" in task or "整理" in task or "学习" in task)

    return {
        "task_type": task_type,
        "queries": queries,
        "urls": urls,
        "required_capabilities": capabilities,
        "must_fetch": bool(must_fetch),
    }


def _task_int_setting(task: str, key: str, default: int) -> int:
    match = re.search(rf"{re.escape(key)}\s*=\s*(\d+)", task, re.I)
    if not match:
        return default
    try:
        return max(1, min(int(match.group(1)), 20))
    except ValueError:
        return default


def _task_text_setting(task: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}\s*=\s*([A-Za-z0-9_-]+)", task, re.I)
    return str(match.group(1)).strip().lower() if match else ""


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _looks_technical(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "api",
        "sdk",
        "framework",
        "docs",
        "documentation",
        "langgraph",
        "langchain",
        "fastapi",
        "openai",
        "python",
        "官方文档",
        "框架",
        "接口",
        "库",
    ]
    return any(marker in lowered for marker in markers)


def _web_round_key(task: str) -> str:
    lowered = task.lower()
    markers = [
        "文档集",
        "调研报告",
        "报告文件夹",
        "save_knowledge_set",
        "source_mode=web",
        "source mode=web",
        "topic 使用",
        "topic=",
    ]
    if not any(marker in lowered for marker in markers):
        return ""
    topic_match = re.search(
        r"topic[^\r\n]{0,12}?(?:使用|为|是|=|：|:)\s*([A-Za-z0-9_.\-\u4e00-\u9fff]+)",
        task,
        re.I,
    )
    if topic_match:
        return topic_match.group(1).strip().rstrip("。；;,，")
    title_match = re.search(r"(?:主题|調研|调研)[：:\s]+([^\n。；;]+)", task)
    if title_match:
        return title_match.group(1).strip()[:80]
    return re.sub(r"\s+", " ", task.strip())[:80]


def _max_web_rounds(task: str) -> int:
    lowered = task.lower()
    if any(marker in lowered for marker in ["max_rounds=3", "深度", "全面", "comprehensive", "deep research"]):
        return 3
    return DEFAULT_WEB_RESEARCH_ROUNDS


def _batch_discovery_sync(queries: list[str]) -> list[dict[str, Any]]:
    """Execute multiple broad_discovery calls concurrently."""
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=min(len(queries), 5))
    try:
        futures = {
            _submit_with_audit_context(executor, broad_discovery, q): q
            for q in queries
        }
        for future in as_completed(futures, timeout=25):
            q = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "sources": [],
                    "provider_attempts": [{
                        "capability": "broad_discovery",
                        "provider": "serper",
                        "status": "error",
                        "reason": f"{q}: {exc}",
                        "fallback_used": False,
                        "degraded": True,
                        "selected": False,
                        "content_length": 0,
                        "content_quality": "low",
                    }],
                })
    except TimeoutError:
        done = {future for future in futures if future.done()}
        pending_queries = [q for future, q in futures.items() if future not in done]
        for q in pending_queries:
            results.append({
                "sources": [],
                "provider_attempts": [{
                    "capability": "broad_discovery",
                    "provider": "serper",
                    "status": "timeout",
                    "reason": f"{q}: batch timeout",
                    "fallback_used": False,
                    "degraded": True,
                    "selected": False,
                    "content_length": 0,
                    "content_quality": "low",
                }],
            })
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _batch_docs_search_sync(queries: list[str]) -> list[dict[str, Any]]:
    """Execute multiple docs_search calls concurrently."""
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=min(len(queries), 5))
    try:
        futures = {
            _submit_with_audit_context(executor, docs_search, q): q
            for q in queries
        }
        for future in as_completed(futures, timeout=28):
            q = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "sources": [],
                    "documents": [],
                    "provider_attempts": [{
                        "capability": "docs_search",
                        "provider": "context7",
                        "status": "error",
                        "reason": f"{q}: {exc}",
                        "fallback_used": False,
                        "degraded": True,
                        "selected": False,
                        "content_length": 0,
                        "content_quality": "low",
                    }],
                })
    except TimeoutError:
        done = {future for future in futures if future.done()}
        pending_queries = [q for future, q in futures.items() if future not in done]
        for q in pending_queries:
            results.append({
                "sources": [],
                "documents": [],
                "provider_attempts": [{
                    "capability": "docs_search",
                    "provider": "context7",
                    "status": "timeout",
                    "reason": f"{q}: batch timeout",
                    "fallback_used": False,
                    "degraded": True,
                    "selected": False,
                    "content_length": 0,
                    "content_quality": "low",
                }],
            })
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _batch_fetch_documents_sync(urls: list[str]) -> list[dict[str, Any]]:
    """Execute multiple fetch_url_with_chain calls concurrently."""
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=min(len(urls), 5))
    try:
        futures = {
            _submit_with_audit_context(executor, fetch_url_with_chain, u): u
            for u in urls
        }
        for future in as_completed(futures, timeout=45):
            u = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "document": None,
                    "provider_attempts": [{
                        "capability": "url_fetch",
                        "provider": "provider_chain",
                        "status": "error",
                        "reason": f"{u}: {exc}",
                        "fallback_used": False,
                        "degraded": True,
                        "selected": False,
                        "content_length": 0,
                        "content_quality": "low",
                    }],
                })
    except TimeoutError:
        done = {future for future in futures if future.done()}
        pending_urls = [u for future, u in futures.items() if future not in done]
        for u in pending_urls:
            results.append({
                "document": None,
                "provider_attempts": [{
                    "capability": "url_fetch",
                    "provider": "provider_chain",
                    "status": "timeout",
                    "reason": f"{u}: batch timeout",
                    "fallback_used": False,
                    "degraded": True,
                    "selected": False,
                    "content_length": 0,
                    "content_quality": "low",
                }],
            })
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for source in sources:
        url = str(source.get("normalized_url") or source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(source)
    return sorted(deduped, key=_source_fetch_priority)


def _source_fetch_priority(source: dict[str, Any]) -> tuple[int, int, int]:
    url = str(source.get("url") or "").strip()
    provider = str(source.get("provider") or "").strip().lower()
    source_type = str(source.get("source_type") or "").strip().lower()
    host = urlparse(url).netloc.lower()
    if source_type in {"docs", "official_docs", "paper", "repository", "github"}:
        authority = 0
    elif any(token in host for token in (".gov", ".edu", "github.com", "arxiv.org", "docs.", "developer.")):
        authority = 0
    elif source_type in {"forum", "community", "video"} or any(token in host for token in ("youtube.com", "youtu.be", "reddit.com", "zhihu.com", "csdn.net", "linkedin.com")):
        authority = 2
    else:
        authority = 1
    if url.startswith("http"):
        return (0, authority, 0 if provider == "serper" else 1)
    return (1, authority, 0)


def _dedupe_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by (url, content), not url alone.

    Context7 (`fetch_context7_context`) can legitimately return several distinct snippets
    from the same source file -- they all carry that file's URL as their `codeId`/`url` since
    they came from the same page, but each snippet's content is a different excerpt. Deduping
    on url alone silently drops every snippet after the first, even when a later one is the
    only material answering a still-open success criterion. See
    project_context7_multi_snippet_dedup_drop_20260818: a "Capability Seam" snippet (the only
    material addressing "接口抽象") was fetched but discarded this way, while the coverage
    checker flagged that same criterion as missing.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for document in documents:
        url = str(document.get("url") or "").strip()
        content = str(document.get("content") or "")
        key = (url, content)
        if not url or key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _select_fetch_urls(
    explicit_urls: list[str],
    sources: list[dict[str, Any]],
    max_urls: int,
    preferred_source_types: list[str] | None = None,
) -> list[str]:
    candidates = [*explicit_urls]
    for source in sources:
        url = str(source.get("url") or "").strip()
        if url.startswith("http"):
            candidates.append(url)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        normalized = re.sub(r"#.*$", "", url.strip())
        if not normalized.startswith("http") or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    selected: list[str] = []
    preferred = [_canonical_source_type(item) for item in (preferred_source_types or [])]
    source_by_url = {str(item.get("url") or "").split("#", 1)[0]: item for item in sources}
    for source_type in preferred:
        for url in deduped:
            source = source_by_url.get(url) or {}
            if url in selected or _canonical_source_type(source.get("source_type")) != source_type:
                continue
            selected.append(url)
            break
        if len(selected) >= max_urls:
            return selected
    seen_domains: set[str] = {urlparse(url).netloc.lower() for url in selected}
    for url in deduped:
        domain = urlparse(url).netloc.lower()
        if domain in seen_domains:
            continue
        selected.append(url)
        seen_domains.add(domain)
        if len(selected) >= max_urls:
            return selected
    for url in deduped:
        if url in selected:
            continue
        selected.append(url)
        if len(selected) >= max_urls:
            return selected
    return selected


def _canonical_source_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "docs": "official_docs",
        "documentation": "official_docs",
        "repository": "github",
        "repo": "github",
        "other": "web_article",
    }.get(normalized, normalized or "web_article")


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _retrieval_pre_diagnostics(
    *,
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
    context_text: str,
    input_text: str,
) -> dict[str, Any]:
    return {
        "sources": {
            "count": len(sources),
            "snippet_chars": sum(len(str(item.get("snippet") or "")) for item in sources),
        },
        "fetched_documents": {
            "count": len(documents),
            "total_content_chars": sum(len(str(item.get("content") or "")) for item in documents),
            "included_content_chars": sum(min(len(str(item.get("content") or "")), DOCUMENT_EXCERPT_CHARS) for item in documents[:MAX_FETCH_URLS]),
            "max_per_document_excerpt_chars": DOCUMENT_EXCERPT_CHARS,
            "qualities": _count_by(documents, "content_quality"),
            "degraded_count": sum(1 for item in documents if item.get("degraded")),
            "items": [
                {
                    "title": str(item.get("title") or "")[:160],
                    "url": str(item.get("url") or "")[:500],
                    "provider": item.get("provider"),
                    "content_length": item.get("content_length") or len(str(item.get("content") or "")),
                    "content_quality": item.get("content_quality"),
                    "degraded": item.get("degraded"),
                    "included_excerpt_chars": min(len(str(item.get("content") or "")), DOCUMENT_EXCERPT_CHARS),
                }
                for item in documents[:MAX_FETCH_URLS]
            ],
        },
        "provider_attempts": {
            "count": len(provider_attempts),
            "selected_count": sum(1 for item in provider_attempts if item.get("selected")),
            "degraded_count": sum(1 for item in provider_attempts if item.get("degraded")),
            "by_provider_status": _provider_attempt_counts(provider_attempts),
        },
        "evidence_model_input": {
            "context_chars": len(context_text),
            "sent_chars": len(input_text),
            "truncated": len(context_text) > len(input_text),
            "input_limit_chars": WEB_SYNTHESIS_INPUT_CHARS,
        },
    }


def _dump_web_synthesis_debug(
    *,
    task: str,
    context_text: str,
    input_text: str,
    selected_chunks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> None:
    if os.getenv("RESEARCH_DEBUG_DUMP") != "1":
        return
    debug_dir = Path(os.getenv("RESEARCH_DEBUG_DIR", "logs/research_debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_topic = re.sub(r"[^A-Za-z0-9_.-]+", "_", _web_round_key(task) or "web_synthesis")[:80]
    base = debug_dir / f"{stamp}-{safe_topic}"
    (base.with_suffix(".input.txt")).write_text(input_text, encoding="utf-8")
    metadata = {
        "task_chars": len(task),
        "context_chars": len(context_text),
        "input_chars": len(input_text),
        "truncated": len(context_text) > len(input_text),
        "selected_chunks_count": len(selected_chunks),
        "selected_chunks_chars": sum(len(str(item.get("text") or "")) for item in selected_chunks),
        "documents_count": len(documents),
        "documents_chars": sum(len(str(item.get("content") or "")) for item in documents),
        "sources_count": len(sources),
        "input_path": str(base.with_suffix(".input.txt")),
    }
    (base.with_suffix(".meta.json")).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _provider_attempt_counts(provider_attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in provider_attempts:
        key = "|".join(
            [
                str(item.get("capability") or "unknown"),
                str(item.get("provider") or "unknown"),
                str(item.get("status") or "unknown"),
                str(item.get("content_quality") or "unknown"),
            ]
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _attach_retrieval_diagnostics(output: str, pre_diagnostics: dict[str, Any], raw_model_output: str) -> str:
    data = _json_from_text(output)
    if not data:
        return output
    data["_diagnostics"] = {
        **pre_diagnostics,
        "section_evidence": data.get("evidence_diagnostics", {}),
        "evidence_model_output": {
            "raw_chars": len(raw_model_output),
            "final_output_chars": len(output),
            "sources_count": len(data.get("sources") or []),
            "documents_count": len(data.get("documents") or []),
            "materials_count": len(data.get("materials") or []),
            "selected_chunks_count": len(data.get("selected_chunks") or []),
            "evidences_count": len(data.get("evidences") or []),
            "key_facts_count": len(data.get("key_facts") or []),
            "evidence_pack_count": len(data.get("evidence_pack") or []),
            "evidence_map_count": len(data.get("evidence_map") or []),
            "compact_chars": {
                "sources": len(json.dumps(data.get("sources") or [], ensure_ascii=False)),
                "documents": len(json.dumps(data.get("documents") or [], ensure_ascii=False)),
                "materials": len(json.dumps(data.get("materials") or [], ensure_ascii=False)),
                "selected_chunks": len(json.dumps(data.get("selected_chunks") or [], ensure_ascii=False)),
                "evidences": len(json.dumps(data.get("evidences") or [], ensure_ascii=False)),
                "key_facts": len(json.dumps(data.get("key_facts") or [], ensure_ascii=False)),
                "evidence_pack": len(json.dumps(data.get("evidence_pack") or [], ensure_ascii=False)),
                "evidence_map": len(json.dumps(data.get("evidence_map") or [], ensure_ascii=False)),
            },
        },
    }
    return json.dumps(data, ensure_ascii=False)


def _build_retrieval_context(
    task: str,
    plan: dict[str, Any],
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Build compact context fed to the retrieval synthesis worker."""
    compact_sources = [
        {
            "title": source.get("title"),
            "url": source.get("url"),
            "snippet": source.get("snippet", "")[:500],
            "source_type": source.get("source_type"),
            "provider": source.get("provider"),
            "query": source.get("query"),
        }
        for source in sources[:MAX_SOURCES]
    ]
    compact_documents = [
        {
            "title": doc.get("title"),
            "url": doc.get("url"),
            "provider": doc.get("provider"),
            "source_type": doc.get("source_type"),
            "content_quality": doc.get("content_quality"),
            "degraded": doc.get("degraded"),
            "content_excerpt": str(doc.get("content", ""))[:DOCUMENT_EXCERPT_CHARS],
        }
        for doc in documents[:MAX_FETCH_URLS]
    ]
    compact_chunks = [
        {
            "id": chunk.get("id"),
            "material_id": chunk.get("material_id"),
            "section_id": chunk.get("section_id"),
            "question_id": chunk.get("question_id"),
            "source_url": chunk.get("source_url"),
            "source_title": chunk.get("source_title"),
            "score": chunk.get("score"),
            "text": str(chunk.get("text") or "")[:1200],
        }
        for chunk in (selected_chunks or [])[:40]
    ]
    parts = [
        f"原始任务：{task}",
        "",
        "检索计划：",
        json.dumps(plan, ensure_ascii=False, indent=2),
        "",
        "候选来源（Source）：",
        json.dumps(compact_sources, ensure_ascii=False, indent=2),
        "",
        "抓取文档（Document）：",
        json.dumps(compact_documents, ensure_ascii=False, indent=2),
        "",
        "按研究问题筛选出的正文片段（Selected Chunks，优先从这里抽取证据）：",
        json.dumps(compact_chunks, ensure_ascii=False, indent=2),
        "",
        "Provider attempts：",
        json.dumps(provider_attempts, ensure_ascii=False, indent=2)[:3000],
    ]
    return "\n".join(parts)


def _extract_section_evidence_maps(
    *,
    sections: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "mode": "controlled_model",
        "sections": [],
        "model_evidence_count": 0,
        "timeout_count": 0,
        "soft_timeout_count": 0,
        "error_count": 0,
        "retry_exhausted_count": 0,
        "rejected_count": 0,
    }
    if not sections or not selected_chunks:
        return [], stats
    if os.getenv("RESEARCH_MODEL_SECTION_EVIDENCE", "1") == "0":
        raise RuntimeError(
            "RESEARCH_MODEL_SECTION_EVIDENCE=0 is no longer supported: "
            "research evidence must be extracted by the model"
        )
    output: list[dict[str, Any]] = []
    section_jobs: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for section in sections[:SECTION_EVIDENCE_MAX_SECTIONS]:
        section_id = str(section.get("id") or "")
        chunks = [
            chunk for chunk in selected_chunks
            if str(chunk.get("section_id") or "") == section_id
        ]
        if not chunks:
            continue
        chunks.sort(key=lambda chunk: float(chunk.get("score") or 0), reverse=True)
        # Widened from SECTION_EVIDENCE_MODEL_CHUNKS (4) to SECTION_EVIDENCE_MAX_CHUNKS (10) --
        # layer 1 (_select_research_chunks) already scores and ranks up to
        # CHUNK_CANDIDATES_PER_QUESTION_CAP=8 candidates per question; capping model_chunks at 4
        # here discarded the bottom half of that already-fetched, already-scored pool before a
        # single model call ever saw it. _extract_single_section_evidence_model batches
        # model_chunks into concurrent SECTION_EVIDENCE_MODEL_CHUNKS-sized calls, so widening this
        # selection increases coverage without growing any single call's input size.
        model_chunks = _balanced_section_chunks_for_evidence(chunks, SECTION_EVIDENCE_MAX_CHUNKS)
        section_jobs.append((section, model_chunks, chunks))
    max_workers = max(1, min(len(section_jobs), int(os.getenv("RESEARCH_SECTION_EVIDENCE_PARALLELISM", "4"))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_index = {
            _submit_with_audit_context(
                executor,
                _extract_single_section_evidence_with_budget,
                task=task,
                section=section,
                model_chunks=model_chunks,
                all_chunks=all_chunks,
            ): index
            for index, (section, model_chunks, all_chunks) in enumerate(section_jobs)
        }
        results: list[dict[str, Any] | None] = [None] * len(section_jobs)
        for future in as_completed(future_by_index):
            results[future_by_index[future]] = future.result()
    for result in [item for item in results if item]:
        output.extend(result["evidence"])
        stats["sections"].append(result["stats"])
        stats["model_evidence_count"] += result["stats"].get("model_evidence_count", 0)
        stats["timeout_count"] += int(result["stats"].get("timeout_count") or 0)
        stats["soft_timeout_count"] += int(result["stats"].get("soft_timeout_count") or 0)
        stats["error_count"] += int(result["stats"].get("error_count") or 0)
        stats["retry_exhausted_count"] += 1 if result["stats"].get("retry_exhausted") else 0
        stats["rejected_count"] += result["stats"].get("rejected_count", 0)
        emit_event(
            "retrieval.section_evidence.extracted",
            "workers",
            "Extracted section evidence.",
            data={
                "section_id": result["stats"].get("section_id"),
                "evidence_count": len(output),
                "status": result["stats"].get("status"),
                "retry_exhausted": result["stats"].get("retry_exhausted"),
            },
        )
    return output, stats


def _extract_single_section_evidence_with_budget(
    *,
    task: str,
    section: dict[str, Any],
    model_chunks: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    section_id = str(section.get("id") or "")
    chunks = all_chunks or model_chunks
    started = time.perf_counter()
    timeout_seconds = float(os.getenv("RESEARCH_SECTION_EVIDENCE_TIMEOUT_SECONDS", str(SECTION_EVIDENCE_MODEL_TIMEOUT_SECONDS)))
    route = _configured_evidence_route()
    stats: dict[str, Any] = {
        "section_id": section_id,
        "title": section.get("title"),
        "status": "pending",
        "chunk_count": len(chunks),
        "model_chunk_count": len(model_chunks),
        **route,
        "model_evidence_count": 0,
        "rejected_count": 0,
        "timeout_count": 0,
        "soft_timeout_count": 0,
        "error_count": 0,
        "retry_exhausted": False,
        "failed_question_ids": [],
        "empty_question_ids": [],
        "elapsed_ms": 0,
    }
    if not chunks:
        stats["status"] = "empty"
        return {"evidence": [], "stats": stats}

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        def record_section_soft_timeout() -> None:
            stats["soft_timeout_count"] = 1
            emit_event(
                "retrieval.section_evidence.soft_timeout",
                "workers",
                "Section evidence extraction exceeded its orchestration threshold; waiting for the bounded model calls to finish.",
                level="warning",
                data={"section_id": section_id, "timeout_seconds": timeout_seconds},
            )

        future = _submit_with_audit_context(
            executor, _extract_single_section_evidence_model, task, section, model_chunks
        )
        model_evidence, rejected_count = _future_result_after_soft_timeout(
            future,
            timeout_seconds=timeout_seconds,
            on_soft_timeout=record_section_soft_timeout,
        )
        stats["status"] = "complete" if model_evidence else "empty"
        stats["model_evidence_count"] = len(model_evidence)
        stats["rejected_count"] = rejected_count
        evidence = model_evidence
    except TimeoutError:
        stats["status"] = "timeout"
        stats["timeout_count"] = 1
        evidence = []
        emit_event(
            "retrieval.section_evidence.timeout",
            "workers",
            "Section evidence model extraction timed out.",
            level="warning",
            data={"section_id": section_id, "timeout_seconds": timeout_seconds},
        )
    except Exception as exc:  # pragma: no cover - model/network dependent
        stats["status"] = "error"
        stats["error_count"] = 1
        stats["error"] = str(exc)[:300]
        evidence = []
        emit_event(
            "retrieval.section_evidence.failed",
            "workers",
            "Section evidence extraction failed.",
            level="warning",
            data={"section_id": section_id, "error": str(exc)[:300]},
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if not evidence:
        # Section-level pass produced nothing at all -- every question in it is starved, so
        # retrying the whole section is equivalent to (and simpler than) scoping to starved
        # questions only.
        retry = _extract_question_level_evidence_with_budget(
            task=task,
            section=section,
            chunks=chunks,
        )
        stats["question_retry_count"] = retry["stats"].get("question_retry_count", 0)
        stats["question_retry_evidence_count"] = retry["stats"].get("question_retry_evidence_count", 0)
        stats["question_retry_timeout_count"] = retry["stats"].get("question_retry_timeout_count", 0)
        stats["question_retry_soft_timeout_count"] = retry["stats"].get("question_retry_soft_timeout_count", 0)
        stats["question_retry_error_count"] = retry["stats"].get("question_retry_error_count", 0)
        stats["failed_question_ids"] = retry["stats"].get("failed_question_ids", [])
        stats["empty_question_ids"] = retry["stats"].get("empty_question_ids", [])
        evidence = retry["evidence"]
    else:
        # Retry granularity is per-question, not per-section: a section that got evidence for
        # some questions but not others used to skip retry entirely (the section "had
        # evidence"), leaving the zero-evidence questions to fall straight to the more heavily
        # gated fallback tiers. Only the actually-empty questions get retried here, which is
        # also cheaper than the old section-wide retry -- questions that already have evidence
        # don't get a redundant model call.
        covered_question_ids = {str(item.get("question_id") or "") for item in evidence}
        starved_chunks = [
            chunk for chunk in chunks
            if str(chunk.get("question_id") or "") not in covered_question_ids
        ]
        if starved_chunks:
            retry = _extract_question_level_evidence_with_budget(
                task=task,
                section=section,
                chunks=starved_chunks,
            )
            stats["question_retry_count"] = retry["stats"].get("question_retry_count", 0)
            stats["question_retry_evidence_count"] = retry["stats"].get("question_retry_evidence_count", 0)
            stats["question_retry_timeout_count"] = retry["stats"].get("question_retry_timeout_count", 0)
            stats["question_retry_soft_timeout_count"] = retry["stats"].get("question_retry_soft_timeout_count", 0)
            stats["question_retry_error_count"] = retry["stats"].get("question_retry_error_count", 0)
            stats["failed_question_ids"] = retry["stats"].get("failed_question_ids", [])
            stats["empty_question_ids"] = retry["stats"].get("empty_question_ids", [])
            evidence = _dedupe_evidence_pack(evidence + retry["evidence"])

    stats["timeout_count"] += int(stats.get("question_retry_timeout_count") or 0)
    stats["soft_timeout_count"] += int(stats.get("question_retry_soft_timeout_count") or 0)
    stats["error_count"] += int(stats.get("question_retry_error_count") or 0)
    # A successful empty extraction means the supplied chunks do not support the
    # question.  That is a valid research outcome, not an exhausted model call.
    # Keep it visible in empty_question_ids and let the question-loop coverage and
    # budget logic decide whether to search again.  Only actual model failures
    # (timeout/error/invalid response after retry) abort the job.
    failed_question_ids = list(dict.fromkeys(stats.get("failed_question_ids", [])))
    if failed_question_ids:
        stats["retry_exhausted"] = True
        stats["status"] = "failed"
    elif evidence:
        stats["status"] = "complete"
    else:
        stats["status"] = "empty"

    concrete_criterion_count = sum(
        1
        for question in section.get("questions") or []
        if isinstance(question, dict)
        for criterion in question.get("criteria") or []
        if isinstance(criterion, dict)
        and str(criterion.get("criterion_id") or "")
        and not str(criterion.get("description") or "").startswith("实质回答研究问题")
    )
    unknown_filtered_count = sum(
        int(item.pop("_criterion_unknown_filtered_count", 0) or 0)
        for item in evidence
        if isinstance(item, dict)
    )
    attributed_count = sum(
        bool(item.get("criterion_ids"))
        for item in evidence
        if isinstance(item, dict)
    )
    stats.update({
        "criteria_seen_count": concrete_criterion_count,
        "criterion_attributed_evidence_count": attributed_count,
        "criterion_empty_evidence_count": max(0, len(evidence) - attributed_count),
        "criterion_unknown_filtered_count": unknown_filtered_count,
        "criterion_attribution_rate": round(attributed_count / len(evidence), 3) if evidence else 0.0,
    })
    stats["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return {"evidence": evidence, "stats": stats}


def _extract_question_level_evidence_with_budget(
    *,
    task: str,
    section: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    target = max(1, int(os.getenv("RESEARCH_TARGET_EVIDENCE_PER_QUESTION", str(QUESTION_EVIDENCE_TARGET))))
    timeout_seconds = float(os.getenv("RESEARCH_QUESTION_EVIDENCE_TIMEOUT_SECONDS", str(QUESTION_EVIDENCE_TIMEOUT_SECONDS)))
    by_question: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        question_id = str(chunk.get("question_id") or "_unknown")
        by_question.setdefault(question_id, []).append(chunk)
    question_jobs = []
    for question_id, question_chunks in by_question.items():
        question_chunks.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        question_jobs.append((question_id, _balanced_question_chunks_for_evidence(question_chunks, QUESTION_EVIDENCE_MODEL_CHUNKS)))
    if not question_jobs:
        return {"evidence": [], "stats": {}}

    max_workers = max(1, min(len(question_jobs), int(os.getenv("RESEARCH_QUESTION_EVIDENCE_PARALLELISM", "4"))))
    evidence: list[dict[str, Any]] = []
    stats = {
        "question_retry_count": len(question_jobs),
        "question_retry_evidence_count": 0,
        "question_retry_timeout_count": 0,
        "question_retry_soft_timeout_count": 0,
        "question_retry_error_count": 0,
        "failed_question_ids": [],
        "empty_question_ids": [],
    }
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            _submit_with_audit_context(
                executor,
                _extract_single_question_evidence_model,
                task,
                section,
                question_id,
                question_chunks,
                target,
            ): (question_id, question_chunks)
            for question_id, question_chunks in question_jobs
        }
        for future, (question_id, question_chunks) in list(futures.items()):
            try:
                def record_question_soft_timeout(question_id: str = question_id) -> None:
                    stats["question_retry_soft_timeout_count"] = (
                        int(stats.get("question_retry_soft_timeout_count") or 0) + 1
                    )
                    emit_event(
                        "retrieval.question_evidence.soft_timeout",
                        "workers",
                        "Question evidence extraction exceeded its orchestration threshold; waiting for the bounded model calls to finish.",
                        level="warning",
                        data={
                            "section_id": section.get("id"),
                            "question_id": question_id,
                            "timeout_seconds": timeout_seconds,
                        },
                    )

                items, _rejected = _future_result_after_soft_timeout(
                    future,
                    timeout_seconds=timeout_seconds,
                    on_soft_timeout=record_question_soft_timeout,
                )
                if items:
                    evidence.extend(items[:target])
                    continue
                stats["empty_question_ids"].append(question_id)
            except TimeoutError:
                stats["question_retry_timeout_count"] += 1
                stats["failed_question_ids"].append(question_id)
                emit_event(
                    "retrieval.question_evidence.timeout",
                    "workers",
                    "Question evidence extraction timed out.",
                    level="warning",
                    data={"section_id": section.get("id"), "question_id": question_id, "timeout_seconds": timeout_seconds},
                )
            except Exception as exc:  # pragma: no cover - model/network dependent
                stats["question_retry_error_count"] += 1
                stats["failed_question_ids"].append(question_id)
                emit_event(
                    "retrieval.question_evidence.failed",
                    "workers",
                    "Question evidence extraction failed.",
                    level="warning",
                    data={"section_id": section.get("id"), "question_id": question_id, "error": str(exc)[:300]},
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    deduped = _dedupe_evidence_pack(evidence)
    stats["question_retry_evidence_count"] = len(deduped)
    return {"evidence": deduped, "stats": stats}


def _balanced_question_chunks_for_evidence(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: float(item.get("score") or 0), reverse=True):
        source = _chunk_source_key(chunk)
        if source in seen_sources:
            continue
        picked.append(chunk)
        if source:
            seen_sources.add(source)
        if len(picked) >= limit:
            return picked
    for chunk in sorted(chunks, key=lambda item: float(item.get("score") or 0), reverse=True):
        if chunk in picked:
            continue
        picked.append(chunk)
        if len(picked) >= limit:
            break
    return picked


def _extract_single_question_evidence_model(
    task: str,
    section: dict[str, Any],
    question_id: str,
    chunks: list[dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], int]:
    question = str(chunks[0].get("question") or question_id) if chunks else question_id
    original_question = next(
        (
            item
            for item in section.get("questions") or []
            if isinstance(item, dict)
            and str(item.get("id") or item.get("question_id") or "") == question_id
        ),
        None,
    )
    scoped_question = dict(original_question) if isinstance(original_question, dict) else {}
    scoped_question["id"] = question_id
    scoped_question.setdefault("text", question)
    scoped_section = {**section, "questions": [scoped_question]}
    model_chunks = chunks[:QUESTION_EVIDENCE_MODEL_CHUNKS]
    output: list[dict[str, Any]] = []
    rejected = 0
    for batch in _evidence_chunk_batches_for_input_budget(task, scoped_section, model_chunks):
        batch_output, batch_rejected = _extract_evidence_batch(
            task,
            scoped_section,
            batch,
            timeout_seconds=float(os.getenv("RESEARCH_QUESTION_EVIDENCE_MODEL_TIMEOUT_SECONDS", "300")),
            derived_from="question_model",
            question_id=question_id,
        )
        output.extend(batch_output)
        rejected += batch_rejected
    return _dedupe_evidence_pack(output)[:target], rejected


def _balanced_section_chunks_for_evidence(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick section chunks without letting one question or source consume the model budget."""
    if limit <= 0:
        return []
    by_question: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        question_id = str(chunk.get("question_id") or "_unknown")
        by_question.setdefault(question_id, []).append(chunk)
    for bucket in by_question.values():
        bucket.sort(key=lambda item: float(item.get("score") or 0), reverse=True)

    picked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for question_id in sorted(by_question):
        chunk = _best_chunk_preferring_new_source(by_question[question_id], seen_sources) or by_question[question_id][0]
        chunk_id = str(chunk.get("id") or "")
        if chunk_id in seen_ids:
            continue
        picked.append(chunk)
        seen_ids.add(chunk_id)
        source_key = _chunk_source_key(chunk)
        if source_key:
            seen_sources.add(source_key)
        if len(picked) >= limit:
            return picked

    for chunk in sorted(chunks, key=lambda item: float(item.get("score") or 0), reverse=True):
        chunk_id = str(chunk.get("id") or "")
        if chunk_id in seen_ids:
            continue
        picked.append(chunk)
        seen_ids.add(chunk_id)
        if len(picked) >= limit:
            break
    return picked


def _best_chunk_preferring_new_source(
    chunks: list[dict[str, Any]],
    seen_sources: set[str],
) -> dict[str, Any] | None:
    for chunk in chunks:
        source_key = _chunk_source_key(chunk)
        if source_key and source_key not in seen_sources:
            return chunk
    return chunks[0] if chunks else None


def _chunk_source_key(chunk: dict[str, Any]) -> str:
    source_url = str(chunk.get("source_url") or "")
    if not source_url:
        return ""
    try:
        host = urlparse(source_url).netloc.lower()
    except Exception:
        host = ""
    return host[4:] if host.startswith("www.") else host or source_url


def _extract_single_section_evidence_model(
    task: str,
    section: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    # Batch instead of hard-truncating to SECTION_EVIDENCE_MODEL_CHUNKS: layer 1
    # (`_select_research_chunks`) already picks up to CHUNK_CANDIDATES_PER_QUESTION_CAP=8
    # candidates per question, but a single model call only ever saw the first
    # SECTION_EVIDENCE_MODEL_CHUNKS=4 -- the other half of an already-scored, already-fetched
    # candidate pool was silently discarded before extraction even ran (confirmed on a real
    # --via-api job: 8 candidates for sec_03_q1, ranks 5-8 including a Martin Fowler / Anthropic
    # engineering-blog paragraph never entered a model call at all). See docs/evaluations/
    # deep-research-revision-loop-reuse-mismatch-20260811-handoff.md follow-up.
    #
    # Widening the single-call chunk count instead (SECTION_EVIDENCE_MODEL_CHUNKS 4->8) was
    # rejected: real model_calls.jsonl traces show `deepseek-v4-flash` (the evidence-extraction
    # model, routed through the SenseNova gateway) already times out on 18.5% and falls back
    # completely on 11% of calls at the CURRENT 4-chunk/9000-char size -- doubling the input
    # would push an already-strained call further past its failure point, not just take longer.
    # Splitting into concurrent SECTION_EVIDENCE_MODEL_CHUNKS-sized batches keeps each
    # individual call's shape identical to what's already proven (barely) stable, and the
    # parallel batches run within the same wall-clock budget as one call.
    batches: list[list[dict[str, Any]]] = []
    for offset in range(0, len(chunks), SECTION_EVIDENCE_MODEL_CHUNKS):
        batches.extend(
            _evidence_chunk_batches_for_input_budget(
                task,
                section,
                chunks[offset:offset + SECTION_EVIDENCE_MODEL_CHUNKS],
            )
        )
    if len(batches) <= 1:
        return _extract_section_evidence_batch(task, section, batches[0] if batches else [])
    output: list[dict[str, Any]] = []
    rejected = 0
    max_workers = max(1, min(len(batches), int(os.getenv("RESEARCH_SECTION_EVIDENCE_BATCH_PARALLELISM", "4"))))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            _submit_with_audit_context(executor, _extract_section_evidence_batch, task, section, batch)
            for batch in batches
        ]
        for future in as_completed(futures):
            # Isolate one batch's timeout/error from the others. The outer extraction wrapper
            # records a soft orchestration threshold and joins these bounded model calls, so a
            # single slow/failed batch must not discard evidence the others already extracted.
            try:
                batch_output, batch_rejected = future.result()
            except Exception:
                continue
            output.extend(batch_output)
            rejected += batch_rejected
    return _dedupe_evidence_pack(output), rejected


def _extract_section_evidence_batch(
    task: str,
    section: dict[str, Any],
    model_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    return _extract_evidence_batch(
        task,
        section,
        model_chunks,
        timeout_seconds=float(os.getenv("RESEARCH_SECTION_EVIDENCE_MODEL_TIMEOUT_SECONDS", "300")),
        derived_from="section_model",
    )


def _extract_evidence_batch(
    task: str,
    section: dict[str, Any],
    model_chunks: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    derived_from: str,
    question_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    variant = str(os.getenv("RESEARCH_EVIDENCE_PROMPT_VARIANT", "claim_quote_system_v4")).strip().lower()
    prompt = _section_evidence_prompt(task, section, model_chunks, variant=variant)
    system_prompt = _section_evidence_system_prompt(variant)
    input_chars = len(prompt) + len(system_prompt)
    if input_chars > SECTION_EVIDENCE_INPUT_CHARS:
        raise ValueError(
            "Evidence extraction prompt exceeds the input budget after packing: "
            f"{input_chars} > {SECTION_EVIDENCE_INPUT_CHARS}"
        )
    with _audit_model_scope(
        stage="content_extraction",
        question_id=question_id,
        section_id=str(section.get("id") or "") or None,
    ):
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = _invoke_evidence_chat(messages, timeout_seconds=timeout_seconds)
        try:
            data = _json_from_text(_content_from_response(response))
        except Exception as exc:
            _record_model_parse(response, error=exc)
            raise
        _record_model_parse(response, parsed_result=data)
    items = _evidence_items_from_model_payload(data)
    if not isinstance(items, list):
        return [], 0
    chunk_by_id = {str(chunk.get("id") or ""): chunk for chunk in model_chunks}
    output: list[dict[str, Any]] = []
    rejected = 0
    # Cap scales with how many chunks the model actually saw (rather than a flat 12) so the
    # "exhaust each chunk" prompt instruction above isn't undermined by a fixed ceiling that
    # assumed one-evidence-per-chunk.
    max_items = max(1, len(model_chunks)) * EVIDENCE_ITEMS_PER_CHUNK
    for item in items[:max_items]:
        normalized = _normalize_section_evidence_item(item, section, chunk_by_id)
        if normalized and _evidence_quote_is_grounded(normalized, chunk_by_id):
            output.append({**normalized, "derived_from": derived_from})
        else:
            rejected += 1
    return _dedupe_evidence_pack(output), rejected


def _evidence_items_from_model_payload(data: Any) -> list[Any]:
    """Accept the canonical evidence list plus a common gateway/model nested variant."""

    if not isinstance(data, dict):
        return []
    for key in ("evidence", "evidence_pack", "evidence_map", "evidences"):
        items = data.get(key)
        if isinstance(items, list) and items:
            return items
    documents = data.get("documents")
    if not isinstance(documents, list):
        return []
    output: list[Any] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        for key in ("evidence", "evidences", "evidence_pack", "evidence_map"):
            items = document.get(key)
            if isinstance(items, list) and items:
                output.extend(items)
                break
    return output


def _evidence_chunk_batches_for_input_budget(
    task: str,
    section: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Pack every selected chunk into valid JSON prompts within the fixed input budget."""
    if not chunks:
        return []
    variant = str(os.getenv("RESEARCH_EVIDENCE_PROMPT_VARIANT", "claim_quote_system_v4")).strip().lower()
    system_chars = len(_section_evidence_system_prompt(variant))
    prompt = _section_evidence_prompt(task, section, chunks)
    if len(prompt) + system_chars <= SECTION_EVIDENCE_INPUT_CHARS:
        return [chunks]
    if len(chunks) > 1:
        midpoint = max(1, len(chunks) // 2)
        return [
            *_evidence_chunk_batches_for_input_budget(task, section, chunks[:midpoint]),
            *_evidence_chunk_batches_for_input_budget(task, section, chunks[midpoint:]),
        ]

    chunk = chunks[0]
    original_text = str(chunk.get("text") or "")[:CHUNK_CHARS]
    empty_chunk = {**chunk, "text": ""}
    empty_prompt = _section_evidence_prompt(task, section, [empty_chunk])
    available_text_chars = SECTION_EVIDENCE_INPUT_CHARS - len(empty_prompt) - system_chars
    if available_text_chars <= 0:
        raise ValueError(
            "Evidence extraction contract exceeds the input budget before chunk text is added: "
            f"{len(empty_prompt)} > {SECTION_EVIDENCE_INPUT_CHARS}"
        )

    fitted_text = _clip_evidence_chunk_text(original_text, available_text_chars)
    fitted_chunk = {**chunk, "text": fitted_text}
    fitted_prompt = _section_evidence_prompt(task, section, [fitted_chunk])
    while fitted_text and len(fitted_prompt) + system_chars > SECTION_EVIDENCE_INPUT_CHARS:
        fitted_text = fitted_text[:-1]
        fitted_chunk["text"] = fitted_text
        fitted_prompt = _section_evidence_prompt(task, section, [fitted_chunk])
    if not fitted_text:
        raise ValueError("Evidence extraction input budget leaves no room for chunk text")

    emit_event(
        "retrieval.section_evidence.input_trimmed",
        "workers",
        "Trimmed one oversized chunk to keep the evidence prompt valid JSON.",
        level="warning",
        data={
            "section_id": section.get("id"),
            "chunk_id": chunk.get("id"),
            "original_chars": len(original_text),
            "sent_chars": len(fitted_text),
            "input_budget_chars": SECTION_EVIDENCE_INPUT_CHARS,
        },
    )
    return [[fitted_chunk]]


def _clip_evidence_chunk_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    boundary = max(candidate.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?", "\n"))
    if boundary >= max(40, int(limit * 0.6)):
        return candidate[:boundary + 1].rstrip()
    return candidate.rstrip()



def _evidence_quote_is_grounded(evidence: dict[str, Any], chunk_by_id: dict[str, dict[str, Any]]) -> bool:
    chunk_id = str(evidence.get("chunk_id") or "")
    chunk = chunk_by_id.get(chunk_id)
    if not chunk:
        return False
    quote = _normalize_quote_text(str(evidence.get("evidence_text") or ""))
    chunk_text = _normalize_quote_text(str(chunk.get("text") or ""))
    if len(quote) < 10 or not chunk_text:
        return False
    return quote in chunk_text


def _normalize_quote_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def _section_evidence_prompt(
    task: str,
    section: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    variant: str | None = None,
) -> str:
    prompt_variant = str(
        variant or os.getenv("RESEARCH_EVIDENCE_PROMPT_VARIANT", "claim_quote_system_v4")
    ).strip().lower()
    compact_chunks = [
        {
            "chunk_id": chunk.get("id"),
            "question_id": chunk.get("question_id"),
            "question": chunk.get("question"),
            "subject": chunk.get("subject"),
            "comparison_axis": chunk.get("comparison_axis"),
            "matrix_cell_id": chunk.get("matrix_cell_id"),
            "source_url": chunk.get("source_url"),
            "source_title": chunk.get("source_title"),
            "text": str(chunk.get("text") or "")[:CHUNK_CHARS],
        }
        for chunk in chunks
    ]
    payload = {
        "task": "从当前章节的网页片段中抽取可审计证据。只返回 JSON。",
        "rules": [
            "只能使用 provided chunks 中的文字，不要补充外部知识。",
            "每条 evidence 必须绑定 chunk_id、question_id、section_id、source_url。",
            "claim 是对证据的最小事实陈述，不要写泛泛总结。",
            "每条 evidence 必须返回 subject 和 subject_relation。subject 是原文事实实际讨论的主体；subject_relation 只能是 target、target_component、comparison、general 或 unknown，并且相对于该 question 的 topic_anchor 判断。不要因为材料与目标主题相关就把比较对象改写成目标主体。",
            "每条 evidence 必须返回 comparison_axis 和 comparison_metric。comparison_axis 是顶层比较桶：若 quote 直接讨论 question.comparison_axis 或其自然子维度，必须原样返回该结构化 comparison_axis；comparison_metric 记录 quote 实际衡量的更具体指标，如时间、长度、导出方式或验证方式。若 quote 属于另一个顶层维度，comparison_axis 忠实返回那个维度；无法确定时返回空字符串。不得仅因片段被检索到当前章节就复制目标轴。",
            "每个 question 的 success_criteria/criteria 是内容维度清单；优先抽取能直接回答这些维度的事实，并保留对应 question_id。",
            "每条 evidence 必须返回 criterion_ids 字段。先判断它回答哪个 question，再对照该 question 的 concrete criteria；只有证据主体与 criterion 目标兼容、且 comparison_axis 与该 question 的结构化维度一致时，才能填写 1-2 个真实 criterion_id。主体不同但可用于对照时保留证据并标 subject_relation=comparison；主体或维度不兼容时 criterion_ids 必须为空。只有确实不直接支持任何具体标准时才允许返回空数组；描述以“实质回答研究问题”开头的合成总括条件不要填写。",
            "active_queries 表示本轮正在补查的具体缺口；片段若只与主题词相似、却不回答问题或任何内容维度，不要为了凑数抽取。",
            "对每个 chunk，必须穷尽其中能独立成立的事实——不要只抽一条就交差；但不设条数上限，材料本来就少的 chunk 不要为了凑数编造或把一个事实拆成多条。",
            "evidence_text 必须是 chunk 中一个连续、逐字一致的原文句子，禁止拼接不相邻句子。",
            "scope 只能是 domain_specific、cross_domain 或 general；只有原文直接讨论问题领域时才能标 domain_specific。",
            "claim_type 只能是 direct_fact、design_principle 或 limitation；跨领域材料必须标 design_principle。",
            "当原文明确描述失败、限制、风险、信息损失、性能下降、缺点或代价时，claim_type 必须标 limitation。",
            "directness_score 和 applicability_score 是 0 到 1 的数字，不确定时取较低值。",
            "不要改写 source_url/source_title/provider；这些字段仅用于绑定，不用于生成新来源。",
            "如果证据不足，少返回或返回空数组，不要编造。",
            "返回形状：{\"evidence\":[{\"chunk_id\":\"...\",\"question_id\":\"...\",\"criterion_ids\":[\"criterion_02\"],\"subject\":\"...\",\"subject_relation\":\"target|target_component|comparison|general|unknown\",\"comparison_axis\":\"顶层比较轴或空字符串\",\"comparison_metric\":\"quote 实际指标或空字符串\",\"claim\":\"...\",\"evidence_text\":\"...\",\"source_url\":\"...\",\"source_title\":\"...\",\"scope\":\"domain_specific|cross_domain|general\",\"claim_type\":\"direct_fact|design_principle|limitation\",\"directness_score\":0.0,\"applicability_score\":0.0,\"confidence\":\"high|medium|low\"}]}",
        ],
        "user_task": task[:1000],
        "section": {
            "id": section.get("id"),
            "title": section.get("title"),
            "questions": section.get("questions") or [],
        },
        "chunks": compact_chunks,
    }
    if _evidence_semantic_arbitration_mode() != "off":
        payload["rules"].insert(
            -1,
            "每条 evidence 必须返回 support_relation：direct_support 表示 quote 直接支持当前 criterion；inference_support 表示至少需要一步推断；context_only 仅作背景；unrelated 不回答当前问题。",
        )
        payload["rules"].insert(
            -1,
            "每条 evidence 必须返回 target_fit：exact_target 表示原文直接讨论目标或合法比较对象；related_target 表示相关对象、组件或上位领域；general 表示一般知识。另返回 semantic_reason，用不超过160字符说明判断依据。",
        )
        payload["rules"][-1] = (
            "返回形状：{\"evidence\":[{\"chunk_id\":\"...\",\"question_id\":\"...\","
            "\"criterion_ids\":[\"criterion_02\"],\"subject\":\"...\","
            "\"subject_relation\":\"target|target_component|comparison|general|unknown\","
            "\"comparison_axis\":\"顶层比较轴或空字符串\",\"comparison_metric\":\"quote 实际指标或空字符串\","
            "\"claim\":\"...\",\"evidence_text\":\"...\",\"source_url\":\"...\","
            "\"source_title\":\"...\",\"scope\":\"domain_specific|cross_domain|general\","
            "\"claim_type\":\"direct_fact|design_principle|limitation\","
            "\"support_relation\":\"direct_support|inference_support|context_only|unrelated\","
            "\"target_fit\":\"exact_target|related_target|general\",\"semantic_reason\":\"...\","
            "\"directness_score\":0.0,\"applicability_score\":0.0,\"confidence\":\"high|medium|low\"}]}"
        )
    if prompt_variant in {"claim_quote_v2", "claim_quote_balanced_v3", "claim_quote_system_v4"}:
        payload["rules"][2:3] = [
            "claim 是对本条 evidence_text 的最小事实陈述。只看这一条 evidence_text，读者也必须能核验 claim 的主体、动作或关系、对象、条件、因果、比较、限制和代价；不得从同一 chunk 的其他句子补全本条 claim。",
            "claim 可以忠实压缩或改写原文，但语义范围不得强于 evidence_text。原文只说明能力或功能存在时，不得增加效果、成本、门槛、风险、适用人群或因果结论。",
            "若一个 claim 需要相邻两句共同支持，evidence_text 必须覆盖这两个连续句子；若两句是可独立成立的事实，应拆成两条 evidence。代词主体只有在连续 quote 内有明确先行词，或 provided chunk 的 subject 字段明确给出时，才能改写为具体主体。",
        ]
        payload["claim_quote_examples"] = [
            {
                "chunk": "工具 A 支持批量处理。但输入规模过大时，结果可能偏离预期。",
                "bad": {
                    "claim": "工具 A 支持批量处理，但输入规模过大时结果可能偏离预期。",
                    "evidence_text": "工具 A 支持批量处理。",
                    "why_bad": "claim 加入了本条 quote 没有包含的限制。",
                },
                "good": [
                    {"claim": "工具 A 支持批量处理。", "evidence_text": "工具 A 支持批量处理。"},
                    {"claim": "输入规模过大时，结果可能偏离预期。", "evidence_text": "但输入规模过大时，结果可能偏离预期。"},
                ],
            },
            {
                "chunk": "平台提供扩展接口。",
                "bad": {
                    "claim": "扩展接口提高了非技术用户的使用门槛。",
                    "evidence_text": "平台提供扩展接口。",
                    "why_bad": "功能存在不能单独推出门槛或成本。",
                },
                "good": [
                    {"claim": "平台提供扩展接口。", "evidence_text": "平台提供扩展接口。"},
                ],
            },
        ]
        payload["comparison_axis_examples"] = [
            {
                "question_comparison_axis": "报告生成",
                "quote_fact": "生成一份报告通常需要 10 分钟。",
                "good": {"comparison_axis": "报告生成", "comparison_metric": "生成时间"},
                "why": "时间是报告生成轴内的具体指标。",
            },
            {
                "question_comparison_axis": "证据引用",
                "quote_fact": "总体有用性评分为 4.2。",
                "good": {"comparison_axis": "总体有用性", "comparison_metric": "评分"},
                "why": "总体有用性不是证据引用的子维度，不能复制目标轴。",
            },
        ]
        if prompt_variant in {"claim_quote_balanced_v3", "claim_quote_system_v4"}:
            for index, rule in enumerate(payload["rules"]):
                if str(rule).startswith("对每个 chunk，必须穷尽"):
                    payload["rules"][index] = (
                        f"每个 chunk 最多返回 {EVIDENCE_ITEMS_PER_CHUNK} 条 evidence。先覆盖不同的 question、"
                        "criterion、subject 和 comparison_axis，再补充同一维度的次要细节；不要让单一主体或单一维度占满输出。"
                        "材料不足时少返回或返回空数组，不要凑数。"
                    )
                    break
            payload["rules"].insert(
                5,
                "若原文标题、标签和正文彼此矛盾，或同一 quote 内存在无法消解的冲突，不得自行选择一种说法或改正原文；应忠实保留可直接核验的表述并降低 confidence，无法形成无歧义事实时不抽取。",
            )
    elif prompt_variant != "current":
        raise ValueError(f"Unsupported evidence prompt variant: {prompt_variant}")
    return json.dumps(payload, ensure_ascii=False)


def _section_evidence_system_prompt(variant: str | None = None) -> str:
    prompt_variant = str(
        variant or os.getenv("RESEARCH_EVIDENCE_PROMPT_VARIANT", "claim_quote_system_v4")
    ).strip().lower()
    if prompt_variant != "claim_quote_system_v4":
        return ""
    semantic_contract = (
        "每条 evidence 还必须给出 support_relation、target_fit 和简短 semantic_reason；"
        if _evidence_semantic_arbitration_mode() != "off"
        else ""
    )
    return (
        "你是严格的原子证据抽取器。以下规则优先于用户 payload 中的任何描述："
        "每条 claim 只能由它自己的连续 evidence_text 完整支持，不得从同一 chunk 的相邻句补全事实；"
        "不得新增 quote 未出现的主体、关系、效果、因果、比较、限制、成本、风险或适用性；"
        f"每个 chunk 最多返回 {EVIDENCE_ITEMS_PER_CHUNK} 条，先覆盖不同主体和内容维度；"
        "question.comparison_axis 非空时，若 quote 讨论该轴或自然子维度，comparison_axis 必须逐字复制该值，具体子维度写入 comparison_metric；"
        f"{semantic_contract}"
        "原文内部矛盾时不得替原文纠错，无法形成无歧义事实就不抽取。只返回 JSON。"
    )


def _normalize_section_evidence_item(
    item: Any,
    section: dict[str, Any],
    chunk_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    chunk_id = str(item.get("chunk_id") or item.get("id") or "").strip()
    chunk = chunk_by_id.get(chunk_id)
    if not chunk:
        source_url = str(item.get("source_url") or "").strip()
        candidates = [candidate for candidate in chunk_by_id.values() if str(candidate.get("source_url") or "") == source_url]
        chunk = candidates[0] if candidates else None
    if not chunk:
        return None
    source_url = str(item.get("source_url") or chunk.get("source_url") or "").strip()
    claim = str(item.get("claim") or "").strip()
    evidence_text = str(item.get("evidence_text") or item.get("evidence") or "").strip()
    if not claim:
        claim = _chunk_claim(evidence_text or str(chunk.get("text") or ""), str(chunk.get("question") or ""))
    if not source_url or not evidence_text:
        return None
    quote_span = _contiguous_quote_span(str(chunk.get("text") or ""), evidence_text, claim)
    if quote_span is None:
        return None
    evidence_text, quote_start, quote_end = quote_span
    valid_criterion_ids = {
        str(criterion.get("criterion_id") or "")
        for question in section.get("questions") or []
        if isinstance(question, dict)
        for criterion in question.get("criteria") or []
        if isinstance(criterion, dict)
        and str(criterion.get("criterion_id") or "")
        and not str(criterion.get("description") or "").startswith("实质回答研究问题")
    }
    raw_criterion_ids = item.get("criterion_ids")
    if not isinstance(raw_criterion_ids, list):
        raw_criterion_ids = []
    raw_known_criterion_ids = [
        str(value)
        for value in raw_criterion_ids
        if str(value)
    ]
    criterion_ids = list(dict.fromkeys(
        str(value)
        for value in raw_known_criterion_ids
        if str(value) in valid_criterion_ids
    ))[:2]
    normalized = {
            "section_id": str(section.get("id") or chunk.get("section_id") or ""),
            "question_id": str(item.get("question_id") or chunk.get("question_id") or ""),
            "criterion_ids": criterion_ids,
            "_criterion_unknown_filtered_count": sum(
                value not in valid_criterion_ids for value in raw_known_criterion_ids
            ),
            "subject": str(item.get("subject") or chunk.get("subject") or ""),
            "subject_relation": str(item.get("subject_relation") or "unknown").strip().lower(),
            "comparison_axis": str(item.get("comparison_axis") or item.get("axis") or "").strip(),
            "comparison_metric": str(item.get("comparison_metric") or item.get("metric") or "").strip(),
            "comparison_axis_declared": bool(
                str(item.get("comparison_axis") or item.get("axis") or "").strip()
            ),
            "expected_comparison_axis": str(chunk.get("comparison_axis") or "").strip(),
            "matrix_cell_id": str(item.get("matrix_cell_id") or chunk.get("matrix_cell_id") or ""),
            "claim": claim[:240],
            "evidence_text": evidence_text[:900],
            "quote_start": quote_start,
            "quote_end": quote_end,
            "source_url": source_url,
        "source_title": str(item.get("source_title") or chunk.get("source_title") or "")[:180],
        "confidence": item.get("confidence") or "medium",
        "scope": item.get("scope"),
        "claim_type": item.get("claim_type"),
        "directness_score": item.get("directness_score"),
        "applicability_score": item.get("applicability_score"),
        "publisher_relation": item.get("publisher_relation"),
        "chunk_id": str(chunk.get("id") or ""),
        "derived_from": "section_model",
    }
    if _evidence_semantic_arbitration_mode() != "off":
        normalized.update({
            "support_relation": str(item.get("support_relation") or "").strip().lower(),
            "target_fit": str(item.get("target_fit") or "").strip().lower(),
            "semantic_reason": str(item.get("semantic_reason") or item.get("reason") or "").strip()[:160],
        })
    return normalized


def _contiguous_quote_span(chunk_text: str, proposed_quote: str, claim: str) -> tuple[str, int, int] | None:
    """Resolve model output to one exact contiguous span owned by the chunk."""
    if not chunk_text.strip():
        return None
    compact_chunk, mapping = _normalized_with_mapping(chunk_text)
    compact_quote = _normalize_quote_text(proposed_quote)
    if len(compact_quote) >= 10:
        offset = compact_chunk.find(compact_quote)
        if offset >= 0:
            start = mapping[offset]
            end = mapping[offset + len(compact_quote) - 1] + 1
            return chunk_text[start:end].strip(), start, end

    target_terms = _span_terms(f"{claim} {proposed_quote}")
    claim_terms = _span_terms(claim)
    if not target_terms:
        return None
    candidates: list[tuple[float, int, int, str]] = []
    for match in re.finditer(r"[^。！？.!?\n]{12,480}[。！？.!?]?", chunk_text):
        sentence = match.group(0).strip()
        if len(sentence) < 18:
            continue
        sentence_terms = _span_terms(sentence)
        target_overlap = len(sentence_terms & target_terms) / max(1, len(target_terms))
        claim_overlap = len(sentence_terms & claim_terms) / max(1, len(claim_terms)) if claim_terms else 0.0
        score = target_overlap * 0.6 + claim_overlap * 0.4
        candidates.append((score, match.start(), match.end(), sentence))
    if not candidates:
        return None
    score, start, end, sentence = max(candidates, key=lambda item: item[0])
    sentence_terms = _span_terms(sentence)
    claim_overlap = len(sentence_terms & claim_terms) / max(1, len(claim_terms)) if claim_terms else 0.0
    if score < 0.42 or claim_overlap < 0.35:
        return None
    return sentence, start, end


def _normalized_with_mapping(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(str(text or "")):
        if char.isspace():
            continue
        chars.append(char.lower())
        mapping.append(index)
    return "".join(chars), mapping


def _span_terms(text: str) -> set[str]:
    from backend.research.text_utils import extract_terms

    return extract_terms(text)


def _dedupe_evidence_pack(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("chunk_id") or ""),
            str(item.get("source_url") or item.get("url") or ""),
            str(item.get("claim") or item.get("evidence_text") or "")[:120],
        )
        if key in seen:
            existing = seen[key]
            existing["criterion_ids"] = list(dict.fromkeys([
                *[str(value) for value in existing.get("criterion_ids") or [] if value],
                *[str(value) for value in item.get("criterion_ids") or [] if value],
            ]))[:2]
            continue
        copied = dict(item)
        seen[key] = copied
        deduped.append(copied)
    return deduped


def _finalize_retrieval_output(
    text: str,
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    provider_attempts: list[dict[str, Any]],
    *,
    materials: list[dict[str, Any]] | None = None,
    selected_chunks: list[dict[str, Any]] | None = None,
    research_sections: list[dict[str, Any]] | None = None,
    section_evidence: list[dict[str, Any]] | None = None,
    section_evidence_stats: dict[str, Any] | None = None,
    attempted_urls: list[str] | None = None,
) -> str:
    parsed = _json_from_text(text)
    if not parsed:
        parsed = {
            "worker": "web_worker",
            "status": "completed",
            "summary": text.strip()[:1200],
            "gaps": ["Worker output was not valid JSON; wrapped for coordinator use."],
        }
    parsed["sources"] = _compact_sources_for_output([
        {
            "title": source.get("title"),
            "url": source.get("url"),
            "source_type": source.get("source_type"),
            "snippet": source.get("snippet"),
            "provider": source.get("provider"),
            "query": source.get("query"),
        }
        for source in sources[:MAX_SOURCES]
    ])
    parsed["documents"] = _compact_documents_for_output([
        {
            "title": doc.get("title"),
            "url": doc.get("url"),
            "provider": doc.get("provider"),
            "source_type": doc.get("source_type"),
            "content_quality": doc.get("content_quality"),
            "degraded": doc.get("degraded"),
            "summary": str(doc.get("content", ""))[:500],
            "content_length": doc.get("content_length") or len(str(doc.get("content") or "")),
        }
        for doc in documents[:MAX_FETCH_URLS]
    ])
    parsed["provider_attempts"] = _compact_provider_attempts(provider_attempts)
    parsed["attempted_urls"] = [str(url)[:500] for url in (attempted_urls or []) if str(url).startswith("http")]
    parsed["evidences"] = _compact_evidences(parsed.get("evidences") or [])
    parsed["key_facts"] = _compact_key_facts(parsed.get("key_facts") or [])
    normalized_pack = _normalize_evidence_pack(parsed, sources, documents)
    parsed["evidence_pack"] = _dedupe_evidence_pack([*(section_evidence or []), *normalized_pack])
    parsed["materials"] = _compact_materials_for_output(materials or [])
    parsed["selected_chunks"] = _compact_chunks_for_output(selected_chunks or [])
    parsed["evidence_map"] = _build_evidence_map(parsed, selected_chunks or [], research_sections or [])
    parsed["evidence_diagnostics"] = _evidence_diagnostics(
        selected_chunks=selected_chunks or [],
        evidence_map=parsed["evidence_map"],
        sections=research_sections or [],
    )
    if section_evidence_stats:
        parsed["evidence_diagnostics"] = {
            **parsed["evidence_diagnostics"],
            "section_count": len(section_evidence_stats.get("sections") or []),
            "model_evidence_count": section_evidence_stats.get("model_evidence_count", 0),
            "timeout_count": section_evidence_stats.get("timeout_count", 0),
            "error_count": section_evidence_stats.get("error_count", 0),
            "retry_exhausted_count": section_evidence_stats.get("retry_exhausted_count", 0),
            "rejected_count": section_evidence_stats.get("rejected_count", 0),
            "extraction_mode": section_evidence_stats.get("mode"),
            "extraction_sections": section_evidence_stats.get("sections") or [],
        }
    parsed["gaps"] = [str(gap)[:300] for gap in (parsed.get("gaps") or [])[:10]]
    parsed.setdefault("suggested_next_step", None)
    return json.dumps(parsed, ensure_ascii=False)


def _extract_research_sections(task: str) -> list[dict[str, Any]]:
    payload = _extract_research_payload(task)
    sections = _sections_from_payload(payload)
    if sections:
        return sections
    section_pattern = re.compile(r"(?P<id>\d{2}_[^\s\n,，、；;:：()（）]+)")
    titles: list[str] = []
    for match in section_pattern.finditer(task):
        title = match.group("id").strip().rstrip("。；;,，")
        if title and title not in titles:
            titles.append(title)
    if not titles:
        titles = ["00_总览"]
    sections: list[dict[str, Any]] = []
    for index, title in enumerate(titles):
        section_id = _section_id(title, index)
        label = _strip_section_prefix(title)
        questions = _section_questions(label)
        sections.append({
            "id": section_id,
            "title": title,
            "questions": questions,
        })
    return sections


def _extract_research_payload(task: str) -> dict[str, Any]:
    match = re.search(r"<<<RESEARCH_PAYLOAD_JSON\s*(\{.*?\})\s*RESEARCH_PAYLOAD_JSON>>>", task, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _sections_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    research_plan = payload.get("research_plan") if isinstance(payload, dict) else {}
    query_plan = payload.get("query_plan") if isinstance(payload, dict) else {}
    raw_sections = research_plan.get("sections") if isinstance(research_plan, dict) else []
    raw_queries = query_plan.get("queries") if isinstance(query_plan, dict) else []
    if not isinstance(raw_sections, list) or not raw_sections:
        return []
    queries_by_section: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_queries, list):
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            section_id = str(item.get("section_id") or "")
            if section_id:
                queries_by_section.setdefault(section_id, []).append(item)
    output: list[dict[str, Any]] = []
    for index, section in enumerate(raw_sections):
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or _section_id(str(section.get("title") or ""), index))
        query_items = queries_by_section.get(section_id) or []
        original_questions = [
            item for item in section.get("questions") or [] if isinstance(item, dict)
        ]
        original_by_id = {
            str(item.get("id") or item.get("question_id") or ""): item
            for item in original_questions
            if str(item.get("id") or item.get("question_id") or "")
        }
        original_by_text = {
            _payload_question_text(item): item
            for item in original_questions
            if _payload_question_text(item)
        }
        questions = []
        for q_index, item in enumerate(query_items):
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            question_id = str(item.get("question_id") or f"{section_id}_q{q_index + 1}")
            original = original_by_id.get(question_id) or original_by_text.get(question) or {}
            question_row = {
                **original,
                "id": question_id,
                "text": question,
                "query": str(item.get("query") or ""),
            }
            for source_key, target_key in (
                ("subject", "subject"),
                ("comparison_axis", "comparison_axis"),
                ("matrix_cell_id", "matrix_cell_id"),
                ("topic_anchor", "topic_anchor"),
                ("question_kind", "question_kind"),
            ):
                value = str(item.get(source_key) or "").strip()
                if value:
                    question_row[target_key] = value
            for key in (
                "purpose",
                "success_criteria",
                "criteria",
                "active_queries",
                "preferred_source_types",
                "source_requirements",
                "dependencies",
                "answer_contract_level",
            ):
                value = item.get(key)
                if value not in (None, "", []):
                    question_row[key] = value
            questions.append(question_row)
        if not questions:
            for q_index, question in enumerate(section.get("questions") or []):
                if isinstance(question, dict):
                    text = _payload_question_text(question)
                    question_row = {
                        **question,
                        "id": str(question.get("id") or question.get("question_id") or f"{section_id}_q{q_index + 1}"),
                        "text": text,
                        "query": str(question.get("query") or ""),
                    }
                else:
                    text = str(question).strip()
                    question_row = {"id": f"{section_id}_q{q_index + 1}", "text": text, "query": ""}
                if text:
                    questions.append(question_row)
        output.append({
            "id": section_id,
            "title": str(section.get("title") or section_id),
            "axis": section.get("axis"),
            "questions": questions or [{"id": f"{section_id}_q1", "text": str(section.get("title") or section_id), "query": ""}],
        })
    return output


def _payload_question_text(question: dict[str, Any]) -> str:
    return str(
        question.get("text") or question.get("question") or question.get("title") or ""
    ).strip()


def _section_id(title: str, index: int) -> str:
    prefix = re.match(r"^(\d{2})[_\-]", title.strip())
    if prefix:
        return f"sec_{prefix.group(1)}"
    return f"sec_{index:02d}"


def _strip_section_prefix(title: str) -> str:
    text = str(title).strip()
    text = re.sub(r"^\d{2}[_\-]", "", text)
    return text or title


def _section_questions(label: str) -> list[str]:
    normalized = label.lower()
    if "纵向" in label or "演化" in label or "历史" in label:
        return ["起源", "发展历程", "关键节点", "阶段变化", "爆发原因"]
    if "横向" in label or "对比" in label or "竞品" in label:
        return ["同类方案", "竞品", "差异", "生态位置", "优劣势"]
    if "机制" in label or "概念" in label or "核心" in label:
        return ["定义", "核心概念", "工作机制", "组成部分", "原理"]
    if "实操" in label or "指南" in label or "路径" in label or "使用" in label:
        return ["实践步骤", "使用方法", "案例", "工作流", "注意事项"]
    if "证据" in label or "参考" in label:
        return ["来源", "证据", "不确定性", "参考资料"]
    if "overview" in normalized or "总览" in label:
        return ["定义", "背景", "核心结论", "价值", "边界"]
    return [label]


def _build_materials(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materials: list[dict[str, Any]] = []
    for index, doc in enumerate(documents):
        content = str(doc.get("content") or "")
        chunks = _split_text_chunks(content, provider=str(doc.get("provider") or ""), url=str(doc.get("url") or ""))
        materials.append({
            "id": f"mat_{index + 1:03d}",
            "title": str(doc.get("title") or doc.get("url") or f"material {index + 1}")[:180],
            "url": str(doc.get("url") or "")[:500],
            "provider": doc.get("provider"),
            "source_type": doc.get("source_type"),
            "content_length": doc.get("content_length") or len(content),
            "content_quality": doc.get("content_quality"),
            "degraded": bool(doc.get("degraded")),
            "excerpt": content[:MATERIAL_EXCERPT_CHARS],
            "chunk_count": len(chunks),
            "_chunks": chunks,
        })
    return materials


# Providers whose fetched content is markdown (tavily/firecrawl always; basic once trafilatura
# is wired in) -- these get structure-aware splitting that protects code/table/list blocks and
# treats #/## headings as soft break points. Everything else (pdf_text, non-.md github_raw
# files) has no reliable markdown syntax to protect, so it falls back to sentence-boundary-only
# or, for source files, the original fixed-length slicer.
_MARKDOWN_CHUNK_PROVIDERS = {"tavily", "firecrawl", "basic"}
_MARKDOWN_FILE_EXTENSIONS = {".md", ".mdx"}

_MD_FENCE_RE = re.compile(r"^```")
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_MD_LIST_ITEM_RE = re.compile(r"^\s*([-*]|\d+[.)])\s+")
_MD_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?.])")


def _split_text_chunks(text: str, *, provider: str = "", url: str = "") -> list[dict[str, Any]]:
    """Dispatch chunk splitting by provider, per the discussed per-provider content shapes.

    tavily/firecrawl/basic emit markdown -> structure-aware split (_split_markdown_chunks).
    github_raw content shape depends on the linked file's extension: .md/.mdx behaves like
    markdown, everything else (source code) has no reliable prose or markdown syntax to key
    off of, so it keeps the original fixed-length slicer verbatim. pdf_text has already lost
    all structure during extraction, so it only gets sentence-boundary-aware splitting.
    """
    provider_key = str(provider or "").strip().lower()
    if provider_key in _MARKDOWN_CHUNK_PROVIDERS:
        return _split_markdown_chunks(text)
    if provider_key == "github_raw":
        path = urlparse(str(url or "")).path.lower()
        ext = path.rsplit(".", 1)
        extension = f".{ext[1]}" if len(ext) == 2 else ""
        if extension in _MARKDOWN_FILE_EXTENSIONS:
            return _split_markdown_chunks(text)
        return _split_fixed_length_chunks(text)
    return _split_sentence_chunks(text)


def _sentence_segments(text: str) -> list[str]:
    """Split text at sentence-ending punctuation without dropping or reordering content."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(compact) if part.strip()]


def _split_sentence_chunks(text: str) -> list[dict[str, Any]]:
    """Sentence-boundary-aware chunking with CHUNK_CHARS as a soft cap: sentences accumulate
    until the next one would exceed the cap, then the chunk is cut at that sentence boundary
    (never mid-sentence). No overlap -- the original overlap existed only to hedge against a
    sentence getting cut in half, which sentence-boundary splitting already prevents."""
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    sentences = _sentence_segments(compact)
    if not sentences:
        return []
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_len = 0
    for sentence in sentences:
        added_len = len(sentence) + (1 if buf else 0)
        if buf and buf_len + added_len > CHUNK_CHARS:
            chunks.append({"text": " ".join(buf), "header_path": ""})
            buf = []
            added_len = len(sentence)
        buf.append(sentence)
        buf_len = (buf_len + added_len) if len(buf) > 1 else added_len
    if buf:
        chunks.append({"text": " ".join(buf), "header_path": ""})
    return chunks


def _split_fixed_length_chunks(text: str) -> list[dict[str, Any]]:
    """The original (pre-sentence-aware) fixed-length slicer with overlap, kept verbatim for
    github_raw source-code files: sentence-boundary regexes are unreliable on code (periods
    show up in calls/decimals), and code has no markdown prose structure to key off of either."""
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not compact:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + CHUNK_CHARS)
        chunk = compact[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "header_path": ""})
        if end >= len(compact):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    """Parse markdown into ordered blocks. Fenced code, tables, and list runs become atomic
    blocks (never split internally); only h1/h2 headings are structural (### and deeper fall
    through as plain text, per the decision to avoid fragmenting on dense small headings)."""
    lines = text.split("\n")
    blocks: list[dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if _MD_FENCE_RE.match(stripped):
            j = i + 1
            while j < n and not _MD_FENCE_RE.match(lines[j].strip()):
                j += 1
            j = min(j + 1, n)
            blocks.append({"kind": "atomic", "text": "\n".join(lines[i:j]).strip()})
            i = j
            continue
        if _MD_TABLE_ROW_RE.match(line) or _MD_TABLE_SEP_RE.match(line):
            j = i
            while j < n and (_MD_TABLE_ROW_RE.match(lines[j]) or _MD_TABLE_SEP_RE.match(lines[j])):
                j += 1
            blocks.append({"kind": "atomic", "text": "\n".join(lines[i:j]).strip()})
            i = j
            continue
        if _MD_LIST_ITEM_RE.match(line):
            j = i + 1
            while j < n and lines[j].strip() and (_MD_LIST_ITEM_RE.match(lines[j]) or lines[j] != lines[j].lstrip()):
                j += 1
            blocks.append({"kind": "atomic", "text": "\n".join(lines[i:j]).strip()})
            i = j
            continue
        heading_match = _MD_HEADING_RE.match(stripped)
        if heading_match:
            blocks.append({"kind": "heading", "level": len(heading_match.group(1)), "text": heading_match.group(2).strip()})
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        j = i
        para_lines: list[str] = []
        while j < n:
            candidate = lines[j]
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                break
            if (
                _MD_FENCE_RE.match(candidate_stripped)
                or _MD_TABLE_ROW_RE.match(candidate)
                or _MD_TABLE_SEP_RE.match(candidate)
                or _MD_LIST_ITEM_RE.match(candidate)
                or _MD_HEADING_RE.match(candidate_stripped)
            ):
                break
            para_lines.append(candidate)
            j += 1
        blocks.append({"kind": "text", "text": "\n".join(para_lines).strip()})
        i = j
    return blocks


def _split_markdown_chunks(text: str) -> list[dict[str, Any]]:
    """Structure-aware chunking for markdown content. Atomic blocks (code/table/list) are
    never split even if they alone exceed CHUNK_CHARS (favor_recall: overlength beats a
    truncated code block or table row). #/## headings are soft break preferences: accumulation
    keeps going past a heading if there's still room under CHUNK_CHARS (so short sections merge
    with neighbors instead of fragmenting), and only takes the most recent heading boundary as
    the cut point once the soft cap would otherwise be exceeded. The heading text itself is
    never appended to chunk bodies -- it's carried only in the `header_path` field, since
    injecting heading vocabulary into every chunk under it would both inflate lexical chunk
    scores and let a model frame an evidence quote across text that was never contiguous in the
    source (see _strip_code_and_table_noise for the analogous code/table risk)."""
    compact = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not compact:
        return []
    blocks = _markdown_blocks(compact)

    units: list[tuple[str, str]] = []
    h1 = h2 = ""
    for block in blocks:
        if block["kind"] == "heading":
            if block["level"] == 1:
                h1, h2 = block["text"], ""
            else:
                h2 = block["text"]
            continue
        header_path = " > ".join(part for part in (h1, h2) if part)
        if block["kind"] == "atomic":
            unit_text = block["text"].strip()
            if unit_text:
                units.append((unit_text, header_path))
        else:
            for sentence in _sentence_segments(block["text"]):
                if sentence:
                    units.append((sentence, header_path))
    if not units:
        return []

    def text_len(buf: list[tuple[str, str]]) -> int:
        return sum(len(item_text) for item_text, _ in buf) + max(0, len(buf) - 1)

    def flush_chunk(chunks_out: list[dict[str, Any]], items: list[tuple[str, str]]) -> None:
        if not items:
            return
        chunks_out.append({
            "text": " ".join(item_text for item_text, _ in items),
            "header_path": items[0][1],
        })

    chunks: list[dict[str, Any]] = []
    buf: list[tuple[str, str]] = []
    break_index: int | None = None

    for unit_text, header_path in units:
        if buf and header_path != buf[-1][1] and break_index is None:
            break_index = len(buf)
        candidate_len = text_len(buf + [(unit_text, header_path)])
        if buf and candidate_len > CHUNK_CHARS:
            if break_index is not None and 0 < break_index < len(buf):
                flush_chunk(chunks, buf[:break_index])
                buf = buf[break_index:]
            else:
                flush_chunk(chunks, buf)
                buf = []
            break_index = None
            if buf and header_path != buf[-1][1] and break_index is None:
                break_index = len(buf)
        buf.append((unit_text, header_path))

    flush_chunk(chunks, buf)
    return chunks


def _rank_chunks_for_query(
    *,
    materials: list[dict[str, Any]],
    terms: list[str],
    query: str,
    topic_anchor: str,
    question_kind: str = "",
    cache_get: Callable[[list[str]], dict[str, list[float]]] | None = None,
    cache_set: Callable[[dict[str, list[float]]], None] | None = None,
) -> tuple[
    list[tuple[float, dict[str, Any], int, str]],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, Any],
]:
    legacy_scored: list[tuple[float, dict[str, Any], int, str]] = []
    zero_lexical_candidates: list[tuple[float, dict[str, Any], int, str]] = []
    for material in materials:
        for c_index, chunk_item in enumerate(material.get("_chunks") or []):
            chunk = chunk_item.get("text", "") if isinstance(chunk_item, dict) else str(chunk_item)
            score = _chunk_score(chunk, terms)
            if score <= 0:
                if _hybrid_rerank_enabled() and len(zero_lexical_candidates) < _hybrid_zero_lexical_candidate_limit():
                    zero_lexical_candidates.append((0.0, material, c_index, chunk))
                continue
            legacy_scored.append((score, material, c_index, chunk))
    legacy_scored.sort(key=lambda item: item[0], reverse=True)
    legacy_ranked, legacy_metrics = _hybrid_rerank_scored_chunks(
        query,
        legacy_scored,
        zero_lexical_candidates=zero_lexical_candidates,
    )
    mode = _chunk_ranker_mode()
    if mode == "legacy":
        return legacy_ranked, legacy_metrics, {
            "ranker_mode": "legacy",
            "lexical_candidate_count": len(legacy_scored),
            "semantic_candidate_count": 0,
            "rrf_union_count": 0,
            "entity_rejected_count": 0,
            "strong_match_count": 0,
            "protected_reinsert_count": 0,
            "ranker_degraded": False,
            "ranker_degraded_reason": "",
        }

    rrf_ranked, rrf_metrics, diagnostics = _rrf_rank_scored_chunks(
        query,
        materials,
        terms,
        topic_anchor=topic_anchor,
        question_kind=question_kind,
        cache_get=cache_get,
        cache_set=cache_set,
    )
    if mode == "rrf_live":
        diagnostics["ranker_mode"] = "rrf_live"
        return rrf_ranked, rrf_metrics, diagnostics

    rrf_rank_by_key = {
        _scored_chunk_key(material, c_index): rank
        for rank, (_score, material, c_index, _chunk) in enumerate(rrf_ranked, start=1)
    }
    for key, rrf_row in rrf_metrics.items():
        legacy_row = legacy_metrics.setdefault(key, {})
        for metric_name in (
            "lexical_score",
            "keyword_coverage_score",
            "lexical_rank",
            "semantic_score",
            "semantic_rank",
            "rrf_score",
            "entity_match_score",
            "canonical_project",
            "entity_mismatch_reason",
            "strong_text_match",
            "strong_match_reason",
        ):
            legacy_row[f"shadow_{metric_name}"] = rrf_row.get(metric_name)
        legacy_row["shadow_rrf_rank"] = rrf_rank_by_key.get(key)
    diagnostics["ranker_mode"] = "rrf_shadow"
    return legacy_ranked, legacy_metrics, diagnostics


def _select_research_chunks(
    materials: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    task: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_terms = _score_terms(task)
    embedding_cache: dict[str, list[float]] = {}

    def cache_get(keys: list[str]) -> dict[str, list[float]]:
        return {key: embedding_cache[key] for key in keys if key in embedding_cache}

    def cache_set(values: dict[str, list[float]]) -> None:
        embedding_cache.update(values)

    for section in sections:
        question_items = _normalize_section_questions(section)
        section_terms = _score_terms(" ".join([str(section.get("title") or ""), *[item["text"] for item in question_items]]))
        for q_index, question in enumerate(question_items):
            question_text = question["text"]
            question_id = question["id"] or f"{section.get('id')}_q{q_index + 1}"
            terms = list(dict.fromkeys([*_score_terms(question_text), *_score_terms(question.get("query") or ""), *section_terms, *task_terms[:8]]))
            rerank_query = " ".join(
                part
                for part in [
                    question_text,
                    str(question.get("query") or ""),
                    str(section.get("title") or ""),
                    task[:300],
                ]
                if part
            )
            topic_anchor = str(question.get("topic_anchor") or question.get("subject") or task[:300])
            scored, rerank_metrics, ranker_diagnostics = _rank_chunks_for_query(
                materials=materials,
                terms=terms,
                query=rerank_query,
                topic_anchor=topic_anchor,
                question_kind=str(question.get("question_kind") or ""),
                cache_get=cache_get,
                cache_set=cache_set,
            )
            per_question_limit = min(CHUNK_CANDIDATES_PER_QUESTION_CAP, len(scored))
            picked = _diverse_scored_chunks(scored, limit=per_question_limit)
            protected_reinsert_count = 0
            if _chunk_ranker_mode() == "rrf_live":
                picked, protected_reinsert_count = _merge_protected_chunks(
                    picked,
                    scored,
                    rerank_metrics,
                    limit=per_question_limit,
                )
            ranker_diagnostics["protected_reinsert_count"] = protected_reinsert_count
            for rank, (score, material, c_index, chunk) in enumerate(picked):
                metrics = rerank_metrics.get(_scored_chunk_key(material, c_index), {})
                chunk_items = material.get("_chunks") or []
                header_path = ""
                if 0 <= c_index < len(chunk_items) and isinstance(chunk_items[c_index], dict):
                    header_path = str(chunk_items[c_index].get("header_path") or "")
                selected.append({
                    "id": f"chk_{len(selected) + 1:04d}",
                    "material_id": material.get("id"),
                    "section_id": section.get("id"),
                    "question_id": question_id,
                    "question": question_text,
                    "subject": question.get("subject") or "",
                    "comparison_axis": question.get("comparison_axis") or "",
                    "matrix_cell_id": question.get("matrix_cell_id") or "",
                    "text": chunk[:CHUNK_CHARS],
                    "header_path": header_path,
                    "score": round(score, 3),
                    "lexical_score": round(float(metrics.get("lexical_score", score)), 3),
                    "semantic_score": _round_optional(metrics.get("semantic_score"), 4),
                    "hybrid_score": _round_optional(metrics.get("hybrid_score"), 4),
                    "rerank_mode": metrics.get("rerank_mode") or "lexical",
                    "lexical_rank": metrics.get("lexical_rank"),
                    "semantic_rank": metrics.get("semantic_rank"),
                    "rrf_score": _round_optional(metrics.get("rrf_score"), 6),
                    "keyword_coverage_score": _round_optional(metrics.get("keyword_coverage_score"), 4),
                    "entity_match_score": _round_optional(metrics.get("entity_match_score"), 3),
                    "canonical_project": metrics.get("canonical_project") or "",
                    "entity_mismatch_reason": metrics.get("entity_mismatch_reason") or "",
                    "strong_text_match": bool(metrics.get("strong_text_match")),
                    "strong_match_reason": metrics.get("strong_match_reason") or "",
                    "protected_after_rerank": bool(metrics.get("protected_after_rerank")),
                    "shadow_rrf_rank": metrics.get("shadow_rrf_rank"),
                    "shadow_rrf_score": _round_optional(metrics.get("shadow_rrf_score"), 6),
                    "shadow_keyword_coverage_score": _round_optional(metrics.get("shadow_keyword_coverage_score"), 4),
                    "shadow_semantic_rank": metrics.get("shadow_semantic_rank"),
                    "shadow_strong_text_match": bool(metrics.get("shadow_strong_text_match")),
                    "chunk_ranker_diagnostics": dict(ranker_diagnostics),
                    "source_url": material.get("url"),
                    "source_title": material.get("title"),
                    "source_type": material.get("source_type"),
                    "chunk_index": c_index,
                    "rank": rank + 1,
                })
    return _dedupe_chunks(selected)


def _normalize_section_questions(section: dict[str, Any]) -> list[dict[str, str]]:
    raw_questions = section.get("questions") or [section.get("title")]
    output: list[dict[str, str]] = []
    for index, item in enumerate(raw_questions):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("question") or item.get("title") or "").strip()
            question_id = str(item.get("id") or item.get("question_id") or f"{section.get('id')}_q{index + 1}")
            query = str(item.get("query") or "")
            subject = str(item.get("subject") or "")
            axis = str(item.get("comparison_axis") or item.get("axis") or "")
            cell_id = str(item.get("matrix_cell_id") or "")
            topic_anchor = str(item.get("topic_anchor") or "")
            question_kind = str(item.get("question_kind") or "")
        else:
            text = str(item or "").strip()
            question_id = f"{section.get('id')}_q{index + 1}"
            query = ""
            subject = ""
            axis = ""
            cell_id = ""
            topic_anchor = ""
            question_kind = ""
        if text:
            row = {
                "id": question_id,
                "text": text,
                "query": query,
            }
            if subject:
                row["subject"] = subject
            if axis:
                row["comparison_axis"] = axis
            if cell_id:
                row["matrix_cell_id"] = cell_id
            if topic_anchor:
                row["topic_anchor"] = topic_anchor
            if question_kind:
                row["question_kind"] = question_kind
            output.append(row)
    return output or [{"id": f"{section.get('id')}_q1", "text": str(section.get("title") or ""), "query": ""}]


def _selected_chunk_ranker_diagnostics(selected_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    per_question: dict[str, dict[str, Any]] = {}
    for chunk in selected_chunks:
        question_id = str(chunk.get("question_id") or "")
        diagnostics = chunk.get("chunk_ranker_diagnostics")
        if question_id and isinstance(diagnostics, dict) and question_id not in per_question:
            per_question[question_id] = dict(diagnostics)
    totals = {
        "lexical_candidate_count": 0,
        "semantic_candidate_count": 0,
        "rrf_union_count": 0,
        "entity_rejected_count": 0,
        "strong_match_count": 0,
        "protected_reinsert_count": 0,
    }
    degraded_reasons: list[str] = []
    for diagnostics in per_question.values():
        for key in totals:
            totals[key] += int(diagnostics.get(key) or 0)
        reason = str(diagnostics.get("ranker_degraded_reason") or "")
        if reason and reason not in degraded_reasons:
            degraded_reasons.append(reason)
    return {
        "mode": _chunk_ranker_mode(),
        "question_count": len(per_question),
        **totals,
        "ranker_degraded": any(bool(item.get("ranker_degraded")) for item in per_question.values()),
        "ranker_degraded_reasons": degraded_reasons,
        "questions": per_question,
    }


def _diverse_scored_chunks(
    scored: list[tuple[float, dict[str, Any], int, str]],
    *,
    limit: int,
) -> list[tuple[float, dict[str, Any], int, str]]:
    picked: list[tuple[float, dict[str, Any], int, str]] = []
    seen_materials: set[str] = set()
    for item in scored:
        material_id = str(item[1].get("id") or item[1].get("url") or "")
        if material_id in seen_materials:
            continue
        picked.append(item)
        seen_materials.add(material_id)
        if len(picked) >= limit:
            return picked
    for item in scored:
        key = (str(item[1].get("id") or item[1].get("url") or ""), item[2])
        if any((str(existing[1].get("id") or existing[1].get("url") or ""), existing[2]) == key for existing in picked):
            continue
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def _chunk_ranker_mode() -> str:
    mode = str(os.getenv("RESEARCH_CHUNK_RANKER_MODE", "legacy")).strip().lower()
    return mode if mode in {"legacy", "rrf_shadow", "rrf_live"} else "legacy"


def _rrf_candidate_limit() -> int:
    return max(4, int(os.getenv("RESEARCH_RRF_CANDIDATES_PER_CHANNEL", str(RRF_CANDIDATES_PER_CHANNEL))))


def _rrf_k() -> int:
    return max(1, int(os.getenv("RESEARCH_RRF_K", str(RRF_DEFAULT_K))))


def _fielded_keyword_metrics(
    material: dict[str, Any],
    c_index: int,
    chunk: str,
    terms: list[str],
) -> dict[str, Any]:
    """Normalized distinct-term coverage across title, heading path and body.

    The old scorer rewards repeated occurrences in the body and never looks at title/header.
    This scorer gives every distinct query term one vote and records the strongest field where
    it appears: title=3, header=2, body=1. Repetition cannot inflate the result.
    """
    from backend.research.text_utils import to_simplified

    unique_terms = list(dict.fromkeys(str(item).strip().lower() for item in terms if str(item).strip()))[:40]
    chunk_items = material.get("_chunks") or []
    header_path = ""
    if 0 <= c_index < len(chunk_items) and isinstance(chunk_items[c_index], dict):
        header_path = str(chunk_items[c_index].get("header_path") or "")
    fields = {
        "title": to_simplified(str(material.get("title") or "").lower()),
        "header": to_simplified(header_path.lower()),
        "body": to_simplified(str(chunk or "").lower()),
    }
    matched_terms: list[str] = []
    field_hits = {"title": 0, "header": 0, "body": 0}
    weighted_hits = 0.0
    for term in unique_terms:
        normalized = to_simplified(term)
        if normalized and normalized in fields["title"]:
            weighted_hits += 3.0
            field_hits["title"] += 1
            matched_terms.append(term)
        elif normalized and normalized in fields["header"]:
            weighted_hits += 2.0
            field_hits["header"] += 1
            matched_terms.append(term)
        elif normalized and normalized in fields["body"]:
            weighted_hits += 1.0
            field_hits["body"] += 1
            matched_terms.append(term)
    coverage = weighted_hits / (3.0 * len(unique_terms)) if unique_terms else 0.0
    return {
        "keyword_coverage_score": max(0.0, min(coverage, 1.0)),
        "matched_keyword_count": len(matched_terms),
        "query_keyword_count": len(unique_terms),
        "matched_keywords": matched_terms[:20],
        "keyword_field_hits": field_hits,
        "header_path": header_path,
    }


def _chunk_embedding_cache_key(material: dict[str, Any], c_index: int, chunk: str) -> str:
    source = str(material.get("url") or material.get("id") or "")
    digest = hashlib.sha1(_embedding_chunk_text(material, chunk).encode("utf-8")).hexdigest()  # noqa: S324
    return f"{source}#{c_index}:{digest}"


def _semantic_rank_all_chunks(
    query: str,
    candidates: list[tuple[float, dict[str, Any], int, str]],
    *,
    cache_get: Callable[[list[str]], dict[str, list[float]]] | None = None,
    cache_set: Callable[[dict[str, list[float]]], None] | None = None,
) -> tuple[list[tuple[float, dict[str, Any], int, str]], dict[tuple[str, int], float]]:
    if not candidates:
        return [], {}
    cache_keys = [_chunk_embedding_cache_key(material, c_index, chunk) for _score, material, c_index, chunk in candidates]
    cached = cache_get(cache_keys) if cache_get else {}
    missing_indexes = [index for index, key in enumerate(cache_keys) if key not in cached]
    texts = [query[:HYBRID_RERANK_INPUT_CHARS]]
    texts.extend(
        _embedding_chunk_text(candidates[index][1], candidates[index][3])
        for index in missing_indexes
    )
    vectors = _embed_texts(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedding response length mismatch")
    query_vector = vectors[0]
    new_vectors = {
        cache_keys[index]: vector
        for index, vector in zip(missing_indexes, vectors[1:])
    }
    if new_vectors and cache_set:
        cache_set(new_vectors)
    cached.update(new_vectors)
    semantic_scores: dict[tuple[str, int], float] = {}
    ranked: list[tuple[float, dict[str, Any], int, str]] = []
    for item, cache_key in zip(candidates, cache_keys):
        _lexical, material, c_index, chunk = item
        vector = cached.get(cache_key)
        if not vector:
            continue
        semantic = _cosine_similarity(query_vector, vector)
        semantic_scores[_scored_chunk_key(material, c_index)] = semantic
        ranked.append((semantic, material, c_index, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked, semantic_scores


def _strong_text_match(
    *,
    material: dict[str, Any],
    c_index: int,
    chunk: str,
    topic_anchor: str,
    query: str,
    keyword_coverage_score: float,
    entity_match_score: float,
) -> tuple[bool, str]:
    if entity_match_score < 0.75:
        return False, "entity_not_confirmed"
    from backend.research.text_utils import normalize_for_substring, to_simplified

    chunk_items = material.get("_chunks") or []
    header_path = ""
    if 0 <= c_index < len(chunk_items) and isinstance(chunk_items[c_index], dict):
        header_path = str(chunk_items[c_index].get("header_path") or "")
    title_header = normalize_for_substring(f"{material.get('title') or ''} {header_path}")
    anchor_normalized = normalize_for_substring(topic_anchor)
    if 3 <= len(anchor_normalized) <= 80 and anchor_normalized in title_header:
        return True, "topic_anchor_phrase_in_title_or_header"

    anchor_terms = _score_terms(topic_anchor)[:8]
    normalized_title_header = to_simplified(f"{material.get('title') or ''} {header_path}".lower())
    if len(anchor_terms) >= 2 and all(to_simplified(term.lower()) in normalized_title_header for term in anchor_terms):
        return True, "all_topic_terms_in_title_or_header"

    query_normalized = normalize_for_substring(query)
    if (
        3 <= len(query_normalized) <= 60
        and query_normalized in title_header
        and keyword_coverage_score >= 0.5
    ):
        return True, "query_phrase_in_title_or_header"
    return False, ""


def _rrf_rank_scored_chunks(
    query: str,
    materials: list[dict[str, Any]],
    terms: list[str],
    *,
    topic_anchor: str,
    question_kind: str = "",
    cache_get: Callable[[list[str]], dict[str, list[float]]] | None = None,
    cache_set: Callable[[dict[str, list[float]]], None] | None = None,
) -> tuple[
    list[tuple[float, dict[str, Any], int, str]],
    dict[tuple[str, int], dict[str, Any]],
    dict[str, Any],
]:
    from backend.research.source_policy import entity_match_metadata

    all_candidates: list[tuple[float, dict[str, Any], int, str]] = []
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    for material in materials:
        for c_index, chunk_item in enumerate(material.get("_chunks") or []):
            chunk = chunk_item.get("text", "") if isinstance(chunk_item, dict) else str(chunk_item)
            keyword = _fielded_keyword_metrics(material, c_index, chunk, terms)
            score = float(keyword["keyword_coverage_score"])
            item = (score, material, c_index, chunk)
            all_candidates.append(item)
            metrics[_scored_chunk_key(material, c_index)] = {
                **keyword,
                "lexical_score": score,
                "semantic_score": None,
                "lexical_rank": None,
                "semantic_rank": None,
                "rrf_score": 0.0,
                "rerank_mode": "rrf",
            }

    lexical_ranked = sorted(
        (item for item in all_candidates if item[0] > 0),
        key=lambda item: (item[0], _chunk_score(item[3], terms)),
        reverse=True,
    )
    semantic_degraded_reason = ""
    try:
        semantic_ranked, semantic_scores = _semantic_rank_all_chunks(
            query,
            all_candidates,
            cache_get=cache_get,
            cache_set=cache_set,
        )
    except Exception as exc:  # pragma: no cover - provider failures vary
        semantic_ranked, semantic_scores = [], {}
        semantic_degraded_reason = str(exc)[:180]
        emit_event(
            "retrieval.rrf_semantic.fallback",
            "workers",
            "Independent semantic recall failed; RRF is using lexical results only.",
            level="warning",
            data={"candidate_count": len(all_candidates), "reason": semantic_degraded_reason},
        )

    limit = _rrf_candidate_limit()
    lexical_top = lexical_ranked[:limit]
    semantic_top = semantic_ranked[:limit]
    lexical_ranks = {
        _scored_chunk_key(material, c_index): rank
        for rank, (_score, material, c_index, _chunk) in enumerate(lexical_top, start=1)
    }
    semantic_ranks = {
        _scored_chunk_key(material, c_index): rank
        for rank, (_score, material, c_index, _chunk) in enumerate(semantic_top, start=1)
    }
    union: dict[tuple[str, int], tuple[float, dict[str, Any], int, str]] = {}
    for item in [*lexical_top, *semantic_top]:
        union[_scored_chunk_key(item[1], item[2])] = item

    ranked: list[tuple[float, dict[str, Any], int, str]] = []
    entity_rejected_count = 0
    strong_match_count = 0
    k = _rrf_k()
    for key, (_score, material, c_index, chunk) in union.items():
        lexical_rank = lexical_ranks.get(key)
        semantic_rank = semantic_ranks.get(key)
        rrf_score = (1.0 / (k + lexical_rank) if lexical_rank else 0.0) + (
            1.0 / (k + semantic_rank) if semantic_rank else 0.0
        )
        chunk_items = material.get("_chunks") or []
        header_path = ""
        if 0 <= c_index < len(chunk_items) and isinstance(chunk_items[c_index], dict):
            header_path = str(chunk_items[c_index].get("header_path") or "")
        entity = entity_match_metadata(
            url=material.get("url"),
            title=material.get("title"),
            snippet=f"{header_path} {chunk[:1000]}",
            topic=topic_anchor or query,
        )
        entity_score = float(entity.get("entity_match_score") or 0.0)
        metrics[key].update({
            "lexical_rank": lexical_rank,
            "semantic_rank": semantic_rank,
            "semantic_score": semantic_scores.get(key),
            "rrf_score": rrf_score,
            "hybrid_score": rrf_score,
            "entity_match_score": entity_score,
            "canonical_project": str(entity.get("canonical_project") or ""),
            "entity_mismatch_reason": str(entity.get("entity_mismatch_reason") or ""),
        })
        if entity_score < 0.5 and str(question_kind or "").lower() != "comparison":
            metrics[key]["entity_rejected"] = True
            entity_rejected_count += 1
            continue
        strong, reason = _strong_text_match(
            material=material,
            c_index=c_index,
            chunk=chunk,
            topic_anchor=topic_anchor,
            query=query,
            keyword_coverage_score=float(metrics[key]["keyword_coverage_score"]),
            entity_match_score=entity_score,
        )
        metrics[key]["strong_text_match"] = strong
        metrics[key]["strong_match_reason"] = reason
        if strong:
            strong_match_count += 1
        quality_tiebreak = _chunk_quality_signal(chunk, material) * 1e-6
        ranked.append((rrf_score + quality_tiebreak, material, c_index, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked, metrics, {
        "ranker_mode": "rrf",
        "lexical_candidate_count": len(lexical_ranked),
        "semantic_candidate_count": len(semantic_ranked),
        "rrf_union_count": len(union),
        "entity_rejected_count": entity_rejected_count,
        "strong_match_count": strong_match_count,
        "ranker_degraded": bool(semantic_degraded_reason),
        "ranker_degraded_reason": semantic_degraded_reason,
    }


def _merge_protected_chunks(
    picked: list[tuple[float, dict[str, Any], int, str]],
    ranked: list[tuple[float, dict[str, Any], int, str]],
    metrics: dict[tuple[str, int], dict[str, Any]],
    *,
    limit: int,
) -> tuple[list[tuple[float, dict[str, Any], int, str]], int]:
    protected: list[tuple[float, dict[str, Any], int, str]] = []
    seen_materials: set[str] = set()
    for item in ranked:
        key = _scored_chunk_key(item[1], item[2])
        if not metrics.get(key, {}).get("strong_text_match"):
            continue
        material_id = str(item[1].get("id") or item[1].get("url") or "")
        if material_id in seen_materials:
            continue
        protected.append(item)
        seen_materials.add(material_id)
        if len(protected) >= 2:
            break

    reinserted = 0
    for item in protected:
        key = _scored_chunk_key(item[1], item[2])
        if any(_scored_chunk_key(existing[1], existing[2]) == key for existing in picked):
            continue
        material_id = str(item[1].get("id") or item[1].get("url") or "")
        same_material_index = next(
            (
                index for index, existing in enumerate(picked)
                if str(existing[1].get("id") or existing[1].get("url") or "") == material_id
            ),
            None,
        )
        if same_material_index is not None:
            picked[same_material_index] = item
        elif len(picked) < limit:
            picked.append(item)
        else:
            replace_index = next(
                (
                    index for index in range(len(picked) - 1, -1, -1)
                    if not metrics.get(_scored_chunk_key(picked[index][1], picked[index][2]), {}).get("strong_text_match")
                ),
                None,
            )
            if replace_index is None:
                continue
            picked[replace_index] = item
        metrics[key]["protected_after_rerank"] = True
        reinserted += 1
    return picked[:limit], reinserted


def _hybrid_rerank_scored_chunks(
    query: str,
    scored: list[tuple[float, dict[str, Any], int, str]],
    *,
    zero_lexical_candidates: list[tuple[float, dict[str, Any], int, str]] | None = None,
) -> tuple[list[tuple[float, dict[str, Any], int, str]], dict[tuple[str, int], dict[str, Any]]]:
    # lexical_score is an unbounded keyword-count score (no fixed upper bound), so every
    # hybrid_score we hand out — whether or not the chunk actually went through embedding
    # rerank — must be normalized against the SAME ceiling. Otherwise chunks that fall
    # outside the rerank candidate window keep a raw, unbounded score that isn't comparable
    # to the normalized [0,1] scores produced below, which corrupts any downstream logic
    # that sorts/aggregates hybrid_score globally (e.g. section dossier ordering, final
    # evidence-priority ranking).
    lexical_ceiling = max((score for score, *_rest in scored), default=1.0) or 1.0
    lexical_only_weight = _hybrid_weights()["lexical"]
    metrics = {
        _scored_chunk_key(material, c_index): {
            "lexical_score": score,
            "semantic_score": None,
            "hybrid_score": lexical_only_weight * max(0.0, min(float(score) / lexical_ceiling, 1.0)),
            "rerank_mode": "lexical",
        }
        for score, material, c_index, _chunk in scored
    }
    if not scored:
        return scored, metrics
    if not _hybrid_rerank_enabled():
        return scored, metrics

    candidate_limit = _hybrid_candidate_limit()
    candidates = scored[:candidate_limit]
    if zero_lexical_candidates:
        candidates = candidates + zero_lexical_candidates[: _hybrid_zero_lexical_candidate_limit()]
    candidates = _unique_scored_items(candidates)
    if len(candidates) < 2:
        return scored, metrics

    try:
        texts = [query[:HYBRID_RERANK_INPUT_CHARS]]
        texts.extend(_embedding_chunk_text(material, chunk) for _score, material, _c_index, chunk in candidates)
        vectors = _embed_texts(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedding response length mismatch")
        query_vector = vectors[0]
        chunk_vectors = vectors[1:]
    except Exception as exc:  # pragma: no cover - exact provider failures vary
        emit_event(
            "retrieval.hybrid_rerank.fallback",
            "workers",
            "Hybrid rerank embedding failed; using lexical ranking.",
            level="warning",
            data={"candidate_count": len(candidates), "reason": str(exc)[:180]},
        )
        return scored, metrics

    # Reuse the same ceiling as the default lexical-only metrics above (not a
    # window-local max) so candidates inside and outside the rerank window share
    # one normalization basis and remain comparable.
    hybrid_items: list[tuple[float, dict[str, Any], int, str]] = []
    seen_candidate_keys: set[tuple[str, int]] = set()
    for (score, material, c_index, chunk), vector in zip(candidates, chunk_vectors):
        key = _scored_chunk_key(material, c_index)
        seen_candidate_keys.add(key)
        lexical_norm = max(0.0, min(float(score) / lexical_ceiling, 1.0))
        semantic = _cosine_similarity(query_vector, vector)
        semantic_norm = (semantic + 1.0) / 2.0
        quality = _chunk_quality_signal(chunk, material)
        weights = _hybrid_weights()
        hybrid_score = (
            weights["lexical"] * lexical_norm
            + weights["semantic"] * semantic_norm
            + weights["quality"] * quality
        )
        metrics[key] = {
            "lexical_score": score,
            "semantic_score": semantic,
            "semantic_norm": semantic_norm,
            "quality_score": quality,
            "weights": weights,
            "hybrid_score": hybrid_score,
            "rerank_mode": "hybrid_embedding",
        }
        hybrid_items.append((hybrid_score, material, c_index, chunk))

    hybrid_items.sort(key=lambda item: item[0], reverse=True)
    tail = [item for item in scored[candidate_limit:] if _scored_chunk_key(item[1], item[2]) not in seen_candidate_keys]
    return hybrid_items + tail, metrics


def _hybrid_rerank_enabled() -> bool:
    raw = os.getenv("RESEARCH_HYBRID_RERANK", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _hybrid_candidate_limit() -> int:
    return max(3, int(os.getenv("RESEARCH_HYBRID_RERANK_CANDIDATES", str(HYBRID_RERANK_CANDIDATES))))


def _hybrid_zero_lexical_candidate_limit() -> int:
    return max(
        0,
        int(os.getenv("RESEARCH_HYBRID_RERANK_ZERO_LEXICAL_CANDIDATES", str(HYBRID_RERANK_ZERO_LEXICAL_CANDIDATES))),
    )


def _hybrid_weights() -> dict[str, float]:
    lexical = _float_env("RESEARCH_HYBRID_LEXICAL_WEIGHT", 0.42)
    semantic = _float_env("RESEARCH_HYBRID_SEMANTIC_WEIGHT", 0.48)
    quality = _float_env("RESEARCH_HYBRID_QUALITY_WEIGHT", 0.10)
    total = lexical + semantic + quality
    if total <= 0:
        return {"lexical": 0.42, "semantic": 0.48, "quality": 0.10}
    return {
        "lexical": lexical / total,
        "semantic": semantic / total,
        "quality": quality / total,
    }


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _unique_scored_items(
    items: list[tuple[float, dict[str, Any], int, str]]
) -> list[tuple[float, dict[str, Any], int, str]]:
    output: list[tuple[float, dict[str, Any], int, str]] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        key = _scored_chunk_key(item[1], item[2])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _scored_chunk_key(material: dict[str, Any], c_index: int) -> tuple[str, int]:
    return (str(material.get("id") or material.get("url") or ""), int(c_index))


def _embedding_chunk_text(material: dict[str, Any], chunk: str) -> str:
    title = str(material.get("title") or "")
    return f"{title}\n{chunk}"[:HYBRID_RERANK_INPUT_CHARS]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    model = (
        os.getenv("RESEARCH_RERANK_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or "BAAI/bge-large-zh-v1.5"
    )
    base_url = (
        os.getenv("RESEARCH_RERANK_EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or None
    )
    api_key = (
        os.getenv("RESEARCH_RERANK_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ("local" if base_url else None)
    )
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("RESEARCH_RERANK_EMBEDDING_TIMEOUT_SECONDS", "20")),
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    batch_size = _embedding_batch_size()
    clipped = [text[: _embedding_input_chars()] for text in texts]
    embeddings: list[list[float]] = []
    try:
        for start in range(0, len(clipped), batch_size):
            response = client.embeddings.create(
                model=model,
                input=clipped[start : start + batch_size],
            )
            embeddings.extend(list(item.embedding) for item in response.data)
    finally:
        client.close()
    return embeddings


def _embedding_batch_size() -> int:
    raw = (
        os.getenv("RESEARCH_RERANK_EMBEDDING_BATCH_SIZE")
        or os.getenv("EMBEDDING_BATCH_NUM")
        or "32"
    )
    return max(1, min(int(raw), 64))


def _embedding_input_chars() -> int:
    raw = os.getenv("RESEARCH_RERANK_EMBEDDING_INPUT_CHARS") or "500"
    return max(120, min(int(raw), HYBRID_RERANK_INPUT_CHARS))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(float(left[index]) * float(right[index]) for index in range(size))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(size)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(size)))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(-1.0, min(dot / (left_norm * right_norm), 1.0))


def _chunk_quality_signal(chunk: str, material: dict[str, Any]) -> float:
    text = str(chunk or "")
    lowered = text.lower()
    signal = 0.0
    if len(text) >= 300:
        signal += 0.25
    if re.search(r"\d{4}|%|倍|million|billion|study|report|according", lowered):
        signal += 0.25
    if any(marker in lowered for marker in ["定义", "机制", "原因", "证据", "案例", "数据", "because", "example"]):
        signal += 0.25
    if material.get("degraded"):
        signal -= 0.2
    quality = str(material.get("content_quality") or "").lower()
    if quality in {"high", "good"}:
        signal += 0.15
    if quality in {"low", "poor"}:
        signal -= 0.15
    return max(0.0, min(signal, 1.0))


def _round_optional(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _score_terms(text: str) -> list[str]:
    from backend.research.text_utils import extract_key_terms

    stop = {"source", "mode", "web", "topic", "使用", "文档", "调研", "研究", "知识库"}
    return [term for term in extract_key_terms(text, top_k=80) if term.lower() not in stop]


def _chunk_score(chunk: str, terms: list[str]) -> float:
    from backend.research.text_utils import to_simplified

    lowered = to_simplified(chunk.lower())
    score = 0.0
    for term in terms:
        if not term:
            continue
        count = lowered.count(term.lower())
        if count:
            score += min(count, 5) * (1.0 + min(len(term), 8) / 10)
    evidence_markers = ["定义", "原因", "步骤", "对比", "优势", "劣势", "案例", "because", "compare", "example"]
    score += sum(0.4 for marker in evidence_markers if marker in lowered)
    return score


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        key = (
            str(chunk.get("material_id") or ""),
            int(chunk.get("chunk_index") or 0),
            str(chunk.get("section_id") or ""),
            str(chunk.get("question_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped[:80]


def _compact_materials_for_output(materials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for item in materials:
        output.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "provider": item.get("provider"),
            "source_type": item.get("source_type"),
            "content_length": item.get("content_length"),
            "content_quality": item.get("content_quality"),
            "url": str(item.get("url") or "")[:500],
            "degraded": item.get("degraded"),
            "excerpt": str(item.get("excerpt") or "")[:MATERIAL_EXCERPT_CHARS],
            "chunk_count": item.get("chunk_count"),
        })
    return output


def _compact_chunks_for_output(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "material_id": item.get("material_id"),
            "section_id": item.get("section_id"),
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "text": str(item.get("text") or "")[:CHUNK_CHARS],
            "header_path": item.get("header_path") or "",
            "score": item.get("score"),
            "lexical_score": item.get("lexical_score"),
            "semantic_score": item.get("semantic_score"),
            "hybrid_score": item.get("hybrid_score"),
            "rerank_mode": item.get("rerank_mode"),
            "lexical_rank": item.get("lexical_rank"),
            "semantic_rank": item.get("semantic_rank"),
            "rrf_score": item.get("rrf_score"),
            "keyword_coverage_score": item.get("keyword_coverage_score"),
            "entity_match_score": item.get("entity_match_score"),
            "canonical_project": item.get("canonical_project") or "",
            "entity_mismatch_reason": item.get("entity_mismatch_reason") or "",
            "strong_text_match": bool(item.get("strong_text_match")),
            "strong_match_reason": item.get("strong_match_reason") or "",
            "protected_after_rerank": bool(item.get("protected_after_rerank")),
            "shadow_rrf_rank": item.get("shadow_rrf_rank"),
            "shadow_rrf_score": item.get("shadow_rrf_score"),
            "shadow_keyword_coverage_score": item.get("shadow_keyword_coverage_score"),
            "shadow_semantic_rank": item.get("shadow_semantic_rank"),
            "shadow_strong_text_match": bool(item.get("shadow_strong_text_match")),
            "chunk_ranker_diagnostics": item.get("chunk_ranker_diagnostics") or {},
            "source_url": item.get("source_url"),
            "source_title": item.get("source_title"),
            "chunk_index": item.get("chunk_index"),
            "rank": item.get("rank"),
        }
        for item in chunks
    ]


def _build_evidence_map(
    parsed: dict[str, Any],
    selected_chunks: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    section_by_title = {str(section.get("title") or ""): section for section in sections}
    first_section = sections[0] if sections else {"id": "sec_00"}
    output: list[dict[str, Any]] = []
    evidence_items = parsed.get("evidence_pack") or parsed.get("evidences") or []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        explicit_section_id = str(item.get("section_id") or "").strip()
        explicit_question_id = str(item.get("question_id") or "").strip()
        suggested = item.get("suggested_titles") or item.get("suggested_sections") or []
        if isinstance(suggested, str):
            suggested = [suggested]
        section = first_section
        for title in suggested if isinstance(suggested, list) else []:
            if str(title) in section_by_title:
                section = section_by_title[str(title)]
                break
        source_url = str(item.get("source_url") or item.get("url") or "").strip()
        section_id = explicit_section_id or str(section.get("id") or "sec_00")
        chunk = _best_chunk_for_evidence(item, selected_chunks, source_url, section_id)
        chunk_id = chunk.get("id") if chunk else None
        evidence_text = str(item.get("evidence_text") or item.get("evidence") or "").strip()
        output.append({
            "id": f"ev_{len(output) + 1:03d}",
            "section_id": section_id,
            "question_id": explicit_question_id or (chunk.get("question_id") if chunk else None),
            "criterion_ids": list(dict.fromkeys(
                str(value)
                for value in item.get("criterion_ids") or []
                if str(value)
            ))[:2],
            "subject": str(item.get("subject") or (chunk.get("subject") if chunk else "") or ""),
            "subject_relation": str(item.get("subject_relation") or "unknown").strip().lower(),
            "comparison_axis": str(item.get("comparison_axis") or item.get("axis") or (chunk.get("comparison_axis") if chunk else "") or ""),
            "matrix_cell_id": str(item.get("matrix_cell_id") or (chunk.get("matrix_cell_id") if chunk else "") or ""),
            "claim": str(item.get("claim") or item.get("evidence_text") or "")[:240],
            "evidence_text": evidence_text[:900],
            "source_url": source_url or (str(chunk.get("source_url") or "") if chunk else ""),
            "source_title": str(item.get("source_title") or (chunk.get("source_title") if chunk else "") or _title_for_url(source_url, [], []) or "")[:180],
            "confidence": item.get("confidence") or "medium",
            "scope": item.get("scope"),
            "claim_type": item.get("claim_type"),
            "directness_score": item.get("directness_score"),
            "applicability_score": item.get("applicability_score"),
            "publisher_relation": item.get("publisher_relation"),
            "derived_from": item.get("derived_from"),
            "quote_start": item.get("quote_start"),
            "quote_end": item.get("quote_end"),
            "used_by_documents": [],
            "chunk_id": chunk_id,
        })
    return _renumber_evidence_ids(_dedupe_evidence_pack(output))


def _chunk_claim(text: str, question: str) -> str:
    # `question` is kept in the signature for call-site compatibility but no longer
    # concatenated: long research questions (200+ chars is common) used to eat the
    # 240-char budget before the actual sentence, leaving a claim that read like a
    # truncated question instead of a factual statement about the evidence.
    return _first_sentence(text)[:240]


def _renumber_evidence_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, item in enumerate(items, start=1):
        output.append({**item, "id": f"ev_{index:03d}"})
    return output


def _first_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    match = re.search(r"^(.{20,240}?[。！？.!?])", compact)
    if match:
        return match.group(1).strip()
    return compact[:180]


def _evidence_diagnostics(
    *,
    selected_chunks: list[dict[str, Any]],
    evidence_map: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_chunk_chars = sum(len(str(chunk.get("text") or "")) for chunk in selected_chunks)
    evidence_text_chars = sum(len(str(item.get("evidence_text") or "")) for item in evidence_map)
    evidence_chunk_ids = {
        str(item.get("chunk_id") or "")
        for item in evidence_map
        if str(item.get("chunk_id") or "").strip()
    }
    section_rows = []
    for section in sections:
        section_id = str(section.get("id") or "")
        section_chunks = [chunk for chunk in selected_chunks if str(chunk.get("section_id") or "") == section_id]
        section_evidence = [item for item in evidence_map if str(item.get("section_id") or "") == section_id]
        source_urls = {
            str(item.get("source_url") or "")
            for item in section_evidence
            if str(item.get("source_url") or "").strip()
        }
        source_domains = {_url_domain(url) for url in source_urls if _url_domain(url)}
        question_ids = {
            str(chunk.get("question_id") or "")
            for chunk in section_chunks
            if str(chunk.get("question_id") or "").strip()
        }
        covered_question_ids = {
            str(item.get("question_id") or "")
            for item in section_evidence
            if str(item.get("question_id") or "").strip()
        }
        question_rows = _question_evidence_rows(
            section=section,
            section_chunks=section_chunks,
            section_evidence=section_evidence,
        )
        weak_questions = [
            row for row in question_rows
            if row.get("evidence_count", 0) < int(os.getenv("RESEARCH_TARGET_EVIDENCE_PER_QUESTION", str(QUESTION_EVIDENCE_TARGET)))
        ]
        risk_single_source_count = sum(
            1 for row in question_rows
            if row.get("risk_flag_count", 0) > 0 and row.get("source_count", 0) < 2
        )
        section_rows.append({
            "section_id": section_id,
            "title": section.get("title"),
            "selected_chunk_count": len(section_chunks),
            "evidence_count": len(section_evidence),
            "source_count": len(source_urls),
            "domain_count": len(source_domains),
            "question_count": len(question_ids),
            "covered_question_count": len(question_ids & covered_question_ids),
            "question_rows": question_rows,
            "matrix_rows": _matrix_evidence_rows(section_chunks=section_chunks, section_evidence=section_evidence),
            "weak_question_count": len(weak_questions),
            "risk_single_source_count": risk_single_source_count,
            "selected_chunk_chars": sum(len(str(chunk.get("text") or "")) for chunk in section_chunks),
            "evidence_text_chars": sum(len(str(item.get("evidence_text") or "")) for item in section_evidence),
        })
    return {
        "selected_chunk_count": len(selected_chunks),
        "selected_chunk_chars": selected_chunk_chars,
        "evidence_count": len(evidence_map),
        "evidence_text_chars": evidence_text_chars,
        "chunk_to_evidence_ratio": round(len(evidence_chunk_ids) / len(selected_chunks), 3) if selected_chunks else 0,
        "sections": section_rows,
    }


def _question_evidence_rows(
    *,
    section: dict[str, Any],
    section_chunks: list[dict[str, Any]],
    section_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    questions = _normalize_section_questions(section)
    rows = []
    for question in questions:
        question_id = str(question.get("id") or "")
        chunks = [chunk for chunk in section_chunks if str(chunk.get("question_id") or "") == question_id]
        evidence = [item for item in section_evidence if str(item.get("question_id") or "") == question_id]
        source_urls = {
            str(item.get("source_url") or "")
            for item in evidence
            if str(item.get("source_url") or "").strip()
        }
        domains = {_url_domain(url) for url in source_urls if _url_domain(url)}
        risk_flags = [
            flag
            for item in evidence
            for flag in _evidence_risk_flags_for_item(item)
        ]
        rows.append({
            "question_id": question_id,
            "question": question.get("text"),
            "selected_chunk_count": len(chunks),
            "evidence_count": len(evidence),
            "source_count": len(source_urls),
            "domain_count": len(domains),
            "risk_flag_count": len(risk_flags),
            "risk_flags": sorted(set(risk_flags)),
        })
    return rows


def _matrix_evidence_rows(
    *,
    section_chunks: list[dict[str, Any]],
    section_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cell_keys: dict[str, dict[str, str]] = {}
    for item in [*section_chunks, *section_evidence]:
        cell_id = str(item.get("matrix_cell_id") or "").strip()
        subject = str(item.get("subject") or "").strip()
        axis = str(item.get("comparison_axis") or item.get("axis") or "").strip()
        if not cell_id and not (subject and axis):
            continue
        key = cell_id or f"{subject}::{axis}"
        cell_keys.setdefault(key, {"matrix_cell_id": cell_id, "subject": subject, "comparison_axis": axis})
    rows = []
    for key, meta in sorted(cell_keys.items()):
        chunks = [
            chunk for chunk in section_chunks
            if _matrix_key_for_item(chunk) == key
        ]
        evidence = [
            item for item in section_evidence
            if _matrix_key_for_item(item) == key
        ]
        source_urls = {
            str(item.get("source_url") or "")
            for item in evidence
            if str(item.get("source_url") or "").strip()
        }
        rows.append({
            **meta,
            "selected_chunk_count": len(chunks),
            "evidence_count": len(evidence),
            "source_count": len(source_urls),
            "domain_count": len({_url_domain(url) for url in source_urls if _url_domain(url)}),
        })
    return rows


def _matrix_key_for_item(item: dict[str, Any]) -> str:
    cell_id = str(item.get("matrix_cell_id") or "").strip()
    if cell_id:
        return cell_id
    subject = str(item.get("subject") or "").strip()
    axis = str(item.get("comparison_axis") or item.get("axis") or "").strip()
    return f"{subject}::{axis}" if subject and axis else ""


def _evidence_risk_flags_for_item(item: dict[str, Any]) -> list[str]:
    text = f"{item.get('claim') or ''}\n{item.get('evidence_text') or item.get('evidence') or ''}"
    flags: list[str] = []
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|倍|个|项|次|万|亿|ms|s|秒|分钟|小时|天|GB|MB|KB)", text):
        flags.append("number_claim")
    if re.search(r"\b\d{4}\b", text):
        flags.append("date_claim")
    if re.search(r"(?:API|SDK|版本|模型|产品|发布|推出|支持|内置|默认)", text, re.I):
        flags.append("product_claim")
    if str(item.get("confidence") or "").lower() == "low":
        flags.append("low_confidence")
    if not _url_domain(str(item.get("source_url") or "")):
        flags.append("missing_domain")
    return flags


def _url_domain(url: str) -> str:
    try:
        parsed = urlparse(str(url))
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _best_chunk_for_evidence(
    evidence: dict[str, Any],
    chunks: list[dict[str, Any]],
    source_url: str,
    section_id: str,
) -> dict[str, Any] | None:
    terms = _score_terms(" ".join([str(evidence.get("claim") or ""), str(evidence.get("evidence_text") or "")]))
    candidates = [
        chunk for chunk in chunks
        if (not source_url or chunk.get("source_url") == source_url)
        and (not section_id or chunk.get("section_id") == section_id)
    ] or chunks
    if not candidates:
        return None
    return max(candidates, key=lambda chunk: _chunk_score(str(chunk.get("text") or ""), terms))


def _compact_sources_for_output(items: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        compact.append({
            "title": str(item.get("title") or "")[:160],
            "url": str(item.get("url") or item.get("source_url") or "")[:500],
            "source_type": item.get("source_type"),
            "snippet": str(item.get("snippet") or item.get("why_useful") or "")[:180],
            "provider": item.get("provider"),
            "source_type": item.get("source_type"),
            "query": item.get("query"),
        })
        if len(compact) >= MAX_OUTPUT_SOURCES:
            break
    return compact


def _compact_documents_for_output(items: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        compact.append({
            "title": str(item.get("title") or "")[:160],
            "url": str(item.get("url") or "")[:500],
            "provider": item.get("provider"),
            "source_type": item.get("source_type"),
            "content_quality": item.get("content_quality"),
            "degraded": item.get("degraded"),
            "summary": str(item.get("summary") or item.get("content_excerpt") or item.get("content") or "")[:240],
            "content_length": item.get("content_length"),
        })
        if len(compact) >= MAX_OUTPUT_DOCUMENTS:
            break
    return compact


def _compact_provider_attempts(items: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        compact.append({
            "capability": item.get("capability"),
            "provider": item.get("provider"),
            "status": item.get("status"),
            "reason": str(item.get("reason") or "")[:180],
            "selected": item.get("selected"),
            "degraded": item.get("degraded"),
            "content_quality": item.get("content_quality"),
            "url": str(item.get("url") or "")[:500],
        })
        if len(compact) >= 10:
            break
    return compact


def _compact_evidences(items: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        compact.append({
            "claim": str(item.get("claim") or "")[:150],
            "evidence_text": str(item.get("evidence_text") or item.get("evidence") or "")[:240],
            "source_url": str(item.get("source_url") or item.get("url") or "")[:500],
            "confidence": item.get("confidence") or "medium",
        })
        if len(compact) >= MAX_OUTPUT_EVIDENCES:
            break
    return compact


def _compact_key_facts(items: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        source_urls = item.get("source_urls")
        if not isinstance(source_urls, list):
            source_urls = [item.get("source_url")] if item.get("source_url") else []
        compact.append({
            "claim": str(item.get("claim") or "")[:160],
            "evidence": str(item.get("evidence") or item.get("evidence_text") or "")[:260],
            "source_urls": [str(url)[:500] for url in source_urls[:4] if str(url).strip()],
            "confidence": item.get("confidence") or "medium",
        })
        if len(compact) >= MAX_OUTPUT_KEY_FACTS:
            break
    return compact


def _normalize_evidence_pack(
    parsed: dict[str, Any],
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return compact writing evidence while preserving backward compatibility."""
    existing = parsed.get("evidence_pack")
    if isinstance(existing, list) and existing:
        return [
            item for item in (_normalize_evidence_item(item, sources, documents) for item in existing)
            if item
        ]

    pack: list[dict[str, Any]] = []
    for evidence in parsed.get("evidences") or []:
        item = _normalize_evidence_item(evidence, sources, documents)
        if item:
            pack.append(item)

    for fact in parsed.get("key_facts") or []:
        if not isinstance(fact, dict):
            continue
        source_urls = fact.get("source_urls")
        if not isinstance(source_urls, list):
            source_urls = [fact.get("source_url")] if fact.get("source_url") else []
        for url in source_urls or [None]:
            item = _normalize_evidence_item(
                {
                    "claim": fact.get("claim"),
                    "evidence_text": fact.get("evidence") or fact.get("evidence_text"),
                    "source_url": url,
                    "confidence": fact.get("confidence"),
                },
                sources,
                documents,
            )
            if item:
                pack.append(item)

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in pack:
        key = (str(item.get("claim") or ""), str(item.get("source_url") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:MAX_OUTPUT_EVIDENCE_PACK]


def _normalize_evidence_item(
    item: Any,
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    claim = str(item.get("claim") or "").strip()
    evidence_text = str(item.get("evidence_text") or item.get("evidence") or "").strip()
    source_url = str(item.get("source_url") or item.get("url") or "").strip()
    if not claim and not evidence_text:
        return None
    source_title = str(item.get("source_title") or _title_for_url(source_url, sources, documents) or "").strip()
    suggested_titles = item.get("suggested_titles") or item.get("suggested_sections") or []
    if isinstance(suggested_titles, str):
        suggested_titles = [suggested_titles]
    if not isinstance(suggested_titles, list):
        suggested_titles = []
    return {
        "claim": (claim or evidence_text[:140])[:160],
        "evidence_text": evidence_text[:300],
        "source_url": source_url,
        "source_title": source_title[:180],
        "confidence": str(item.get("confidence") or "medium").strip() or "medium",
        "suggested_titles": [str(title).strip() for title in suggested_titles if str(title).strip()],
    }


def _title_for_url(
    url: str,
    sources: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> str | None:
    if not url:
        return None
    for item in [*sources, *documents]:
        item_url = str(item.get("url") or "").strip()
        if item_url == url:
            return str(item.get("title") or "").strip() or None
    return None


def _content_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text") or item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content)


def _json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _repo_task_denied_reason(task: str) -> str:
    lowered = task.lower()
    hard_web_markers = [
        "source_mode=web",
        "source mode=web",
        "source mode: web",
        "只使用外部",
        "只用外部",
        "仅使用外部",
        "外部网页资料",
        "不读取当前本地项目源码",
        "不读取本地项目源码",
        "不读取本地",
        "不要读取本地",
        "不要读本地",
        "不使用本地",
        "不要使用本地",
        "不要调用 repo_worker_tool",
        "do not use local",
        "do not read local",
        "do not call repo_worker_tool",
        "do not use repo_worker_tool",
    ]
    if any(marker in lowered for marker in hard_web_markers):
        return "Task explicitly requires web-only research or forbids local/repo reading."
    return ""


def build_knowledge_manager_agent() -> Any:
    """Build a fresh, one-shot Knowledge Manager Worker agent.

    See docs/modules/kb-qa-channel-bypass-v0.1.md 4.4. Structurally identical to how
    repo_worker_tool builds its own `create_agent(...)` inline: no
    checkpointer, no HITL middleware -- this worker is stateless and re-created per call, not a
    persistent session. It intentionally does NOT hold `delete_knowledge_base_index` in its
    tool list (that stays in main.py::build_agent()'s own tools, gated by
    HumanInTheLoopMiddleware); this is the "propose, don't execute" boundary from the design
    doc, enforced structurally (the tool literally isn't reachable from here) rather than only
    by prompt instruction.
    """
    return create_agent(
        model=_main_chat_model(),
        tools=[
            list_knowledge_bases,
            list_knowledge_base_status,
            inspect_knowledge_base,
            inspect_theme_tool,
            summarize_knowledge_base,
            check_knowledge_base_index,
            index_knowledge_base_lightrag,
            evaluate_knowledge_base_qa,
            set_active_knowledge_base,
            query_knowledge_base,
            bind_channel_to_topic_tool,
            unbind_channel_topic_tool,
            get_channel_topic_binding_tool,
            get_channel_push_config_tool,
            set_channel_push_config_tool,
            list_personal_knowledge_tree_tool,
            preview_knowledge_tree_change_tool,
            preview_research_archive_tool,
            query_personal_knowledge_scope_tool,
            bind_channel_to_scope_tool,
        ],
        system_prompt=KNOWLEDGE_MANAGER_AGENT_PROMPT,
        middleware=[ContextSnipMiddleware(max_chars=4000), LoopGuardMiddleware(max_repeats=1)],
    )


@tool
def knowledge_manager_worker_tool(task: str) -> str:
    """将知识库管理、诊断、索引维护类任务委派给 Knowledge Manager Worker。

    当主 Agent 需要查看知识库列表/状态、查看单个知识库的文档结构、生成只读管理概览、诊断
    LightRAG 索引、构建/刷新索引、跑问答质量评测、切换当前会话的 active knowledge base，或者
    需要判断"某个知识库的索引该不该删除"时使用此工具。

    重要边界：本 worker 没有 `delete_knowledge_base_index` 工具，不能、也无法执行删除索引这个
    动作——涉及删除的判断只会作为返回 JSON 里的 `delete_index_proposal` 建议出现，不是执行记录。
    如果你认可这个建议，需要你自己调用 `delete_knowledge_base_index`（该工具仍受
    HumanInTheLoopMiddleware 强制人工审批，不会因为委派给本 worker 而被绕过）。

    输入：
    - 具体的知识库管理/诊断任务描述，尽量包含涉及的 topic 名称和要确认的问题。

    输出：
    - JSON 字符串，包含 findings、actions_taken、delete_index_proposal（与删除无关时为 null）、
      uncertainties 和 suggested_next_step。
    """
    emit_event(
        "worker.knowledge_manager.started",
        "workers",
        "Knowledge manager worker started.",
        data={"task_chars": len(task)},
    )
    worker = build_knowledge_manager_agent()
    result = worker.invoke({"messages": [{"role": "user", "content": task[:1000]}]})
    text = _last_text(result)
    output = _ensure_jsonish("knowledge_manager", text)
    emit_event(
        "worker.knowledge_manager.finished",
        "workers",
        "Knowledge manager worker finished.",
        data={"output_chars": len(output)},
    )
    return output


def _ensure_jsonish(worker: str, text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    parsed = _json_from_text(stripped)
    if parsed:
        return json.dumps(parsed, ensure_ascii=False)
    return json.dumps(
        {
            "worker": worker,
            "status": "completed",
            "summary": stripped[:1200],
            "uncertainties": ["Worker output was not valid JSON; wrapped for coordinator use."],
        },
        ensure_ascii=False,
    )
