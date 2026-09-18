"""Per-question agentic retrieval tools: search_web / search_local / done.

Implements the tool contracts decided in
docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md section 6
("工具契约"). The three tools are built per-question by `build_question_agent_tools`, a factory
closure that captures `pool`/`material_store`/`state` so the model only ever sees a `query`
argument -- `job_id`/`question_id` are never model-fillable parameters (existing project
convention, see the "ID 注入方式" note in the same doc section).

Design points carried over verbatim from the doc (not reinvented here):
- `search_web` does discovery+select+fetch+clean+chunk only; it never extracts evidence itself.
  It always finishes by running the same query through `search_local`'s ranking+extraction logic
  so the agent sees newly-fetched content in the same turn.
- `search_local` is the only place extraction happens, scoped to exactly this one question (not
  the whole section) -- the v0.1 "chunk 打分" bug this fixes is described in the doc's section 6.
- Extraction (and the evidence commit that follows) is unconditional inside these two tools, not
  something the agent can choose to skip -- there is no separate `commit_evidence` tool.
- Any exception during extraction is retried up to `RESEARCH_AGENTIC_MODEL_MAX_ATTEMPTS` times
  (each retry naturally rotates credentials via `ResearchModelPool`'s round-robin cursor); once
  retries are exhausted the round is skipped -- no deterministic fallback extraction.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from .audit import submit_with_audit_context

from .evidence import (
    admit_evidence,
    apply_semantic_arbitration,
    evidence_semantic_arbitration_mode,
    merge_evidence_clusters,
)
from .material_store import JobMaterialStore
from .models import ResearchQuestionState
from .source_pool import GlobalSourcePool, SourceCandidate


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class AgenticBudget:
    """The three-layer safety valve from the decision doc's section 4.

    All three numbers are explicitly marked in the doc as "试跑起始参数" (starting values for a
    pilot run, not settled production constants) -- every one is read from an env var each time
    so they can be retuned without a code change once real job data comes back.
    """

    max_web_calls: int
    max_local_calls: int
    timeout_seconds: float

    @classmethod
    def for_depth(cls, depth: str) -> "AgenticBudget":
        is_deep = str(depth or "").strip().lower() == "deep"
        return cls(
            max_web_calls=_env_int("RESEARCH_AGENTIC_MAX_WEB_CALLS", 6),
            max_local_calls=_env_int("RESEARCH_AGENTIC_MAX_LOCAL_CALLS", 15 if is_deep else 12),
            timeout_seconds=_env_float(
                "RESEARCH_AGENTIC_DEEP_TIMEOUT_SECONDS" if is_deep else "RESEARCH_AGENTIC_NORMAL_TIMEOUT_SECONDS",
                600.0 if is_deep else 360.0,
            ),
        )

    @classmethod
    def for_recon(cls) -> "AgenticBudget":
        """Budget for `clarify_concept_tool`'s second-stage agentic topic exploration
        (docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md) -- a single continuous
        agent session whose own `done()` call is the convergence signal (决策: 单一完整 agent
        session + 终结合成总结), sized at roughly the "~3 rounds" the design doc describes rather
        than a full per-question research budget. This call is synchronous inside a chat turn
        (main Agent -> clarify_concept_tool -> here), so the timeout is deliberately shorter than
        `for_depth`'s -- there is no per-turn hard timeout elsewhere in the stack to catch a
        runaway session (see feishu_ws_service.py's watchdog, which only guards connection
        liveness, not per-turn latency), so this budget is the only backstop.
        """
        return cls(
            max_web_calls=_env_int("RESEARCH_RECON_AGENTIC_MAX_WEB_CALLS", 8),
            max_local_calls=_env_int("RESEARCH_RECON_AGENTIC_MAX_LOCAL_CALLS", 16),
            timeout_seconds=_env_float("RESEARCH_RECON_AGENTIC_TIMEOUT_SECONDS", 240.0),
        )

    @classmethod
    def for_section_focus(cls) -> "AgenticBudget":
        """Budget for `draft_research_plan_tool`'s redo-adjustment focused supplemental
        exploration (docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md 线1的延伸) --
        scoped to one section's direction, not the whole topic, so it needs less budget than
        `for_recon()`'s whole-topic pass while still being a real multi-round exploration rather
        than the single lightweight lookup the old redo path skipped straight past.
        """
        return cls(
            max_web_calls=_env_int("RESEARCH_RECON_SECTION_AGENTIC_MAX_WEB_CALLS", 5),
            max_local_calls=_env_int("RESEARCH_RECON_SECTION_AGENTIC_MAX_LOCAL_CALLS", 10),
            timeout_seconds=_env_float("RESEARCH_RECON_SECTION_AGENTIC_TIMEOUT_SECONDS", 150.0),
        )



@dataclass(slots=True)
class AgenticQuestionSession:
    """Mutable per-question run state shared by all three tool closures.

    One instance per question agent invocation; never shared across questions (each question's
    `search_web`/`search_local`/`done` closures capture their own session).
    """

    job_id: str
    user_request: str
    state: ResearchQuestionState
    pool: GlobalSourcePool
    material_store: JobMaterialStore
    budget: AgenticBudget
    started: float = field(default_factory=time.perf_counter)
    web_calls_used: int = 0
    local_calls_used: int = 0
    call_counter: int = 0
    model_calls: int = 0
    done_flag: bool = False
    done_reason: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def web_budget_remaining(self) -> int:
        return max(0, self.budget.max_web_calls - self.web_calls_used)

    def local_budget_remaining(self) -> int:
        return max(0, self.budget.max_local_calls - self.local_calls_used)

    def time_remaining(self) -> float:
        return max(0.0, self.budget.timeout_seconds - self.elapsed_seconds())

    def next_call_id(self) -> int:
        with self._lock:
            self.call_counter += 1
            return self.call_counter


def _trace_enabled() -> bool:
    return os.getenv("RESEARCH_DEBUG_AGENT_TRACE") == "1"


def _trace_dir(job_id: str) -> Path:
    return Path(os.getenv("RESEARCH_TRACE_DIR", "logs/research_jobs")) / job_id / "agentic_trace"


def _trace_event(session: AgenticQuestionSession, tool_name: str, payload: dict[str, Any]) -> None:
    """Record one step's full tool call/response, gated by RESEARCH_DEBUG_AGENT_TRACE.

    Mirrors workers.py's `_write_research_trace` (env-gated JSONL under `logs/research_jobs/`) --
    same on-disk convention, new per-question-agent shape. Never raises: tracing must not break
    the retrieval loop it is observing.
    """
    if not _trace_enabled():
        return
    try:
        from .observability import emit_event  # local import: observability lives at repo root
    except ImportError:  # pragma: no cover - script mode fallback
        from observability import emit_event  # type: ignore
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "job_id": session.job_id,
        "question_id": session.state.question_id,
        "tool": tool_name,
        "web_calls_used": session.web_calls_used,
        "local_calls_used": session.local_calls_used,
        "elapsed_seconds": round(session.elapsed_seconds(), 3),
        **payload,
    }
    session.trace.append(row)
    try:
        directory = _trace_dir(session.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{session.state.question_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    try:
        emit_event(
            "agentic_retrieval.tool.called",
            "agentic_tools",
            f"{tool_name} called for {session.state.question_id}",
            data={
                "job_id": session.job_id,
                "question_id": session.state.question_id,
                "tool": tool_name,
                "web_budget_remaining": session.web_budget_remaining(),
                "local_calls_used": session.local_calls_used,
            },
        )
    except Exception:
        pass


def _extract_with_retry(
    *,
    task: str,
    section: dict[str, Any],
    chunks: list[dict[str, Any]],
    max_attempts: int,
) -> tuple[list[dict[str, Any]], int]:
    """Retry `_extract_section_evidence_batch` on ANY exception, rotating credentials.

    See docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md
    section 3: unlike `invoke_research_chat`'s own rate-limit-only retry,
    this retries any failure. Each fresh call re-enters `ResearchModelPool.credential()`, whose
    round-robin cursor naturally hands out a different credential -- no extra plumbing needed.
    A fixed backoff sleep separates attempts (`RESEARCH_AGENTIC_RETRY_BACKOFF_SECONDS`, same env
    var `agentic_model.AgenticRetryChatModel` uses) -- under real concurrent load a failure here
    is usually the whole "evidence" role pool being saturated, not one bad key, so retrying
    instantly just re-hits the same congestion.
    Retry exhaustion returns an empty result (skip this round); the caller must not construct
    evidence directly from the selected chunks.
    """
    from workers import _extract_section_evidence_batch  # lazy: avoid workers.py's app-init cycle

    backoff_seconds = _env_float("RESEARCH_AGENTIC_RETRY_BACKOFF_SECONDS", 1.5)
    last_exc: Exception | None = None
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            return _extract_section_evidence_batch(task, section, chunks)
        except Exception as exc:  # pragma: no cover - exact provider failures vary
            last_exc = exc
            if attempt < attempts - 1 and backoff_seconds > 0:
                time.sleep(backoff_seconds)
            continue
    _ = last_exc
    return [], 0


def _rank_local_chunks(
    *,
    query: str,
    session: AgenticQuestionSession,
    page_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every chunk in the shared material pool against `query`, return top `page_size`.

    Uses workers.py's shared legacy/RRF ranker so batch and agentic retrieval cannot silently
    diverge. In RRF mode, chunk vectors are cached on the job-owned material store and only newly
    fetched chunks are embedded; each local query computes one new query vector.
    """
    from workers import (  # lazy: avoid workers.py's app-init import cycle
        _build_materials,
        _chunk_ranker_mode,
        _diverse_scored_chunks,
        _merge_protected_chunks,
        _rank_chunks_for_query,
        _score_terms,
        _scored_chunk_key,
    )

    documents = session.material_store.all_documents()
    materials = _build_materials(documents)
    state = session.state
    terms = list(dict.fromkeys([*_score_terms(query), *_score_terms(state.question)]))
    rerank_query = " ".join(part for part in [query, state.question, state.topic_anchor] if part)
    scored, metrics, diagnostics = _rank_chunks_for_query(
        materials=materials,
        terms=terms,
        query=rerank_query,
        topic_anchor=state.topic_anchor or state.question,
        question_kind=state.question_kind,
        cache_get=session.material_store.cached_chunk_embeddings,
        cache_set=session.material_store.store_chunk_embeddings,
    )
    picked = _diverse_scored_chunks(scored, limit=min(page_size, len(scored)))
    protected_reinsert_count = 0
    if _chunk_ranker_mode() == "rrf_live":
        picked, protected_reinsert_count = _merge_protected_chunks(
            picked,
            scored,
            metrics,
            limit=min(page_size, len(scored)),
        )
    diagnostics["protected_reinsert_count"] = protected_reinsert_count
    aggregate = state.diagnostics.setdefault("chunk_ranker", {
        "call_count": 0,
        "lexical_candidate_count": 0,
        "semantic_candidate_count": 0,
        "rrf_union_count": 0,
        "entity_rejected_count": 0,
        "strong_match_count": 0,
        "protected_reinsert_count": 0,
        "ranker_degraded": False,
        "ranker_degraded_reasons": [],
    })
    aggregate["mode"] = diagnostics.get("ranker_mode")
    aggregate["call_count"] += 1
    for key in (
        "lexical_candidate_count",
        "semantic_candidate_count",
        "rrf_union_count",
        "entity_rejected_count",
        "strong_match_count",
        "protected_reinsert_count",
    ):
        aggregate[key] += int(diagnostics.get(key) or 0)
    if diagnostics.get("ranker_degraded"):
        aggregate["ranker_degraded"] = True
        reason = str(diagnostics.get("ranker_degraded_reason") or "")
        if reason and reason not in aggregate["ranker_degraded_reasons"]:
            aggregate["ranker_degraded_reasons"].append(reason)
    chunks: list[dict[str, Any]] = []
    for score, material, c_index, chunk_text in picked:
        row_metrics = metrics.get(_scored_chunk_key(material, c_index), {})
        chunks.append({
            "id": f"pending",
            "material_id": material.get("id"),
            "section_id": state.section_id,
            "question_id": state.question_id,
            "question": state.question,
            "comparison_axis": state.comparison_axis,
            "text": chunk_text,
            "score": round(score, 3),
            "lexical_score": row_metrics.get("lexical_score"),
            "lexical_rank": row_metrics.get("lexical_rank"),
            "semantic_score": row_metrics.get("semantic_score"),
            "semantic_rank": row_metrics.get("semantic_rank"),
            "rrf_score": row_metrics.get("rrf_score"),
            "keyword_coverage_score": row_metrics.get("keyword_coverage_score"),
            "rerank_mode": row_metrics.get("rerank_mode") or "lexical",
            "entity_match_score": row_metrics.get("entity_match_score"),
            "canonical_project": row_metrics.get("canonical_project") or "",
            "entity_mismatch_reason": row_metrics.get("entity_mismatch_reason") or "",
            "strong_text_match": bool(row_metrics.get("strong_text_match")),
            "strong_match_reason": row_metrics.get("strong_match_reason") or "",
            "protected_after_rerank": bool(row_metrics.get("protected_after_rerank")),
            "shadow_rrf_rank": row_metrics.get("shadow_rrf_rank"),
            "shadow_rrf_score": row_metrics.get("shadow_rrf_score"),
            "source_url": material.get("url"),
            "source_title": material.get("title"),
            "source_type": material.get("source_type"),
            "chunk_index": c_index,
        })
    return chunks, materials


def _namespace_and_extract(
    *,
    session: AgenticQuestionSession,
    query: str,
    chunks: list[dict[str, Any]],
    call_id: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Namespace this call's chunk ids, extract evidence scoped to ONE question, admit it.

    `chunk_id` prefix is `{question_id}_call{N}_` (per-tool-call-sequence, not per-round -- see
    the doc's "数据结构兼容性" section for why this replaced the old `_r{round}_` scheme).
    `section["questions"]` intentionally contains only this question (the v0.1 fix the doc calls
    out in section 6) so the model's extraction context is not diluted by sibling questions.
    """
    state = session.state
    prefix = f"{state.question_id}_call{call_id}_"
    namespaced: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        namespaced.append({
            **chunk,
            "id": f"{prefix}chk_{index:04d}",
            "comparison_axis": chunk.get("comparison_axis") or state.comparison_axis,
        })
    if not namespaced:
        return [], []
    section = {
        "id": state.section_id,
        "title": state.section_title,
        "questions": [{
            "id": state.question_id,
            "text": state.question,
            "query": query,
            "comparison_axis": state.comparison_axis,
            "criteria": [
                {
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "query_hint": criterion.query_hint,
                    "level": criterion.level,
                }
                for criterion in state.criteria
            ],
        }],
    }
    task = session.user_request or state.purpose or state.question
    evidence_items, _rejected = _extract_with_retry(
        task=task,
        section=section,
        chunks=namespaced,
        max_attempts=max_attempts,
    )
    session.model_calls += 1
    if not evidence_items:
        return namespaced, []
    accepted, rejected = admit_evidence(
        evidence_items,
        chunks=namespaced,
        question_id=state.question_id,
        section_id=state.section_id,
        question=state.question,
        target_subject=state.topic_anchor,
        comparison_subjects=state.comparison_subjects,
        expected_comparison_axis=state.comparison_axis,
        valid_criterion_ids={criterion.criterion_id for criterion in state.criteria},
    )
    semantic_mode = evidence_semantic_arbitration_mode()
    conflict_items = [
        item for item in accepted
        if (item.get("semantic_arbitration") or {}).get("conflict_reasons")
    ]
    arbitration_result = {"status": "not_needed", "verdicts": {}, "reason": ""}
    arbitration_called = False
    if semantic_mode != "off" and conflict_items:
        if session.model_calls >= state.budget.max_model_calls or session.time_remaining() <= 0:
            arbitration_result = {
                "status": "unavailable",
                "verdicts": {},
                "reason": "question_model_call_or_time_budget_exhausted",
            }
        else:
            from workers import arbitrate_evidence_semantics

            session.model_calls += 1
            arbitration_called = True
            arbitration_result = arbitrate_evidence_semantics(
                question_id=state.question_id,
                section_id=state.section_id,
                question=state.question,
                purpose=state.purpose,
                criteria=[{
                    "criterion_id": criterion.criterion_id,
                    "description": criterion.description,
                    "level": criterion.level,
                } for criterion in state.criteria],
                target_subject=state.topic_anchor,
                comparison_subjects=state.comparison_subjects,
                comparison_axis=state.comparison_axis,
                evidence=conflict_items,
                timeout_seconds=max(1.0, min(
                    _env_float("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_TIMEOUT_SECONDS", 45.0),
                    session.time_remaining(),
                )),
            )
    if semantic_mode != "off":
        accepted, semantic_rejected = apply_semantic_arbitration(
            accepted,
            verdicts=arbitration_result.get("verdicts") or {},
            failure_status=(
                str(arbitration_result.get("status") or "")
                if conflict_items and arbitration_result.get("status") != "resolved"
                else ""
            ),
            failure_reason=str(arbitration_result.get("reason") or ""),
        )
        rejected.extend(semantic_rejected)
        state.diagnostics.setdefault("semantic_arbitration_rounds", []).append({
            "tool_call_id": call_id,
            "mode": semantic_mode,
            "conflict_count": len(conflict_items),
            "call_count": int(arbitration_called),
            "status": arbitration_result.get("status"),
            "reason": arbitration_result.get("reason"),
            "resolved_count": len(arbitration_result.get("verdicts") or {}),
            "quarantined_count": len(semantic_rejected),
        })
    state.accepted_evidence, _cluster_diagnostics = merge_evidence_clusters(state.accepted_evidence, accepted)
    state.rejected_evidence.extend(rejected)
    state.selected_chunks = [*state.selected_chunks, *namespaced]
    return namespaced, accepted


def _local_search_round(
    *,
    session: AgenticQuestionSession,
    query: str,
    call_id: int,
    page_size: int,
    max_attempts: int,
) -> dict[str, Any]:
    chunks, _materials = _rank_local_chunks(query=query, session=session, page_size=page_size)
    namespaced, accepted = _namespace_and_extract(
        session=session,
        query=query,
        chunks=chunks,
        call_id=call_id,
        max_attempts=max_attempts,
    )
    return {
        "candidate_chunks": [
            {
                "chunk_id": item["id"],
                "score": item.get("score"),
                "source_url": item.get("source_url"),
                "source_title": item.get("source_title"),
                "text_preview": str(item.get("text") or "")[:280],
            }
            for item in namespaced
        ],
        "newly_committed_evidence": [
            {
                "claim": item.get("atomic_claim"),
                "source_url": item.get("source_url"),
                "evidence_score": item.get("evidence_score"),
            }
            for item in accepted
        ],
    }


def build_question_agent_tools(
    *,
    job_id: str,
    user_request: str,
    state: ResearchQuestionState,
    pool: GlobalSourcePool,
    material_store: JobMaterialStore,
    depth: str,
    budget: "AgenticBudget | None" = None,
    include_fetch_url: bool = False,
) -> tuple[list[Any], AgenticQuestionSession]:
    """Build the (search_web, search_local, done[, fetch_url]) tool set for one agent session.

    Returns the tools plus the `AgenticQuestionSession` the orchestrator inspects afterward
    (`state.accepted_evidence` is mutated in place as tools run, so the session itself mainly
    carries budget/diagnostic bookkeeping the orchestrator needs for `_finalize_state`).

    `budget` lets a caller override `AgenticBudget.for_depth(depth)` -- clarify_concept_tool's
    second-stage recon session uses `AgenticBudget.for_recon()` instead (see
    docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md), since it is a synchronous
    chat-turn call with no per-question research depth involved. `include_fetch_url` adds a
    `fetch_url` tool so the recon agent can retrieve a specific URL the user pasted into their
    request verbatim, rather than only being able to hope `search_web`'s keyword search happens
    to resurface it (the per-question retrieval loop never needed this: questions are generated
    from the plan, not handed a user-supplied URL to chase).
    """
    session = AgenticQuestionSession(
        job_id=job_id,
        user_request=user_request,
        state=state,
        pool=pool,
        material_store=material_store,
        budget=budget or AgenticBudget.for_depth(depth),
    )
    max_attempts = _env_int("RESEARCH_AGENTIC_MODEL_MAX_ATTEMPTS", 3)
    web_fetch_per_call = max(1, _env_int("RESEARCH_AGENTIC_WEB_FETCH_PER_CALL", 3))
    local_page_size = max(1, _env_int("RESEARCH_AGENTIC_LOCAL_PAGE_SIZE", 8))
    ranker_mode = str(os.getenv("RESEARCH_SOURCE_RANKER_MODE", "scoped")).strip().lower()
    if ranker_mode not in {"legacy", "scoped"}:
        ranker_mode = "scoped"

    def _budget_status() -> dict[str, Any]:
        return {
            "web_budget_remaining": session.web_budget_remaining(),
            "local_calls_used": session.local_calls_used,
            "local_budget_remaining": session.local_budget_remaining(),
            "time_remaining_seconds": round(session.time_remaining(), 1),
        }

    def search_web(query: str) -> str:
        if session.time_remaining() <= 0:
            return json.dumps({"status": "denied", "reason": "timeout", **_budget_status()}, ensure_ascii=False)
        if session.web_calls_used >= session.budget.max_web_calls:
            return json.dumps(
                {"status": "denied", "reason": "web_budget_exhausted", **_budget_status()},
                ensure_ascii=False,
            )
        session.web_calls_used += 1
        call_id = session.next_call_id()

        from tools.web import broad_discovery, fetch_url_with_chain  # lazy: heavy provider module

        discovery = broad_discovery(query, num_results=5)
        sources = discovery.get("sources") or []
        pool.add(sources, state, match_states=[state])
        selected = pool.select([state], max_urls=web_fetch_per_call, mode=ranker_mode)
        fetched_titles: list[str] = []
        if selected:
            max_workers = max(1, min(len(selected), _env_int("RESEARCH_AGENTIC_FETCH_PARALLELISM", 4)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    submit_with_audit_context(executor, fetch_url_with_chain, candidate.url): candidate
                    for candidate in selected
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - provider dependent
                        result = {"document": None, "provider_attempts": [{
                            "capability": "url_fetch",
                            "provider": "agentic_search_web",
                            "status": "error",
                            "reason": str(exc)[:240],
                            "degraded": True,
                            "url": candidate.url,
                        }]}
                    material_store.record_fetch(candidate, result)
                    document = result.get("document") if isinstance(result, dict) else None
                    if isinstance(document, dict) and str(document.get("content") or "").strip():
                        fetched_titles.append(str(document.get("title") or candidate.url))
        state.sources = [*state.sources, *[item.to_dict() for item in selected]]
        state.attempted_fetch_urls = list(dict.fromkeys([*state.attempted_fetch_urls, *[item.url for item in selected]]))
        local_round = _local_search_round(
            session=session,
            query=query,
            call_id=call_id,
            page_size=local_page_size,
            max_attempts=max_attempts,
        )
        response = {
            "status": "ok",
            "query": query,
            "discovered_source_count": len(sources),
            "fetched_document_count": len(fetched_titles),
            "fetched_titles": fetched_titles[:10],
            **local_round,
            **_budget_status(),
        }
        _trace_event(session, "search_web", response)
        return json.dumps(response, ensure_ascii=False)

    def search_local(query: str) -> str:
        if session.time_remaining() <= 0:
            return json.dumps({"status": "denied", "reason": "timeout", **_budget_status()}, ensure_ascii=False)
        if session.local_calls_used >= session.budget.max_local_calls:
            return json.dumps(
                {"status": "denied", "reason": "local_budget_exhausted", **_budget_status()},
                ensure_ascii=False,
            )
        session.local_calls_used += 1
        call_id = session.next_call_id()
        local_round = _local_search_round(
            session=session,
            query=query,
            call_id=call_id,
            page_size=local_page_size,
            max_attempts=max_attempts,
        )
        response = {
            "status": "ok",
            "query": query,
            **local_round,
            **_budget_status(),
        }
        _trace_event(session, "search_local", response)
        return json.dumps(response, ensure_ascii=False)

    def done(reason: str = "") -> str:
        session.done_flag = True
        session.done_reason = reason or "agent_reported_sufficient"
        response = {"status": "ok", "message": "循环结束，等待服务端核验覆盖度。", **_budget_status()}
        _trace_event(session, "done", {**response, "reason": session.done_reason})
        return json.dumps(response, ensure_ascii=False)

    def fetch_url(url: str) -> str:
        """Directly fetch one specific URL (does not consume web_budget -- it is not a search,
        just a retrieval of a page the caller already knows the address of, mirroring how a
        user-supplied reference link is meant to be treated as already-known material rather than
        something to be rediscovered)."""
        clean_url = str(url or "").strip()
        if not clean_url.startswith(("http://", "https://")):
            return json.dumps({"status": "invalid_url", **_budget_status()}, ensure_ascii=False)
        if session.time_remaining() <= 0:
            return json.dumps({"status": "denied", "reason": "timeout", **_budget_status()}, ensure_ascii=False)
        call_id = session.next_call_id()

        from tools.web import fetch_url_with_chain  # lazy: heavy provider module

        candidate = SourceCandidate(url=clean_url, provider="agentic_fetch_url")
        try:
            result = fetch_url_with_chain(clean_url)
        except Exception as exc:  # pragma: no cover - provider dependent
            result = {"document": None, "provider_attempts": [{
                "capability": "url_fetch",
                "provider": "agentic_fetch_url",
                "status": "error",
                "reason": str(exc)[:240],
                "degraded": True,
                "url": clean_url,
            }]}
        material_store.record_fetch(candidate, result)
        document = result.get("document") if isinstance(result, dict) else None
        fetched = isinstance(document, dict) and str(document.get("content") or "").strip() != ""
        if fetched:
            candidate.title = str(document.get("title") or clean_url)
            candidate.fetched = True
            candidate.discovered_by_question_ids.add(state.question_id)
            state.sources = [*state.sources, candidate.to_dict()]
        state.attempted_fetch_urls = list(dict.fromkeys([*state.attempted_fetch_urls, clean_url]))
        local_round = (
            _local_search_round(
                session=session,
                query=clean_url,
                call_id=call_id,
                page_size=local_page_size,
                max_attempts=max_attempts,
            )
            if fetched
            else {"candidate_chunks": [], "newly_committed_evidence": []}
        )
        response = {
            "status": "ok" if fetched else "fetch_failed",
            "url": clean_url,
            "title": str(document.get("title") or "") if isinstance(document, dict) else "",
            **local_round,
            **_budget_status(),
        }
        _trace_event(session, "fetch_url", response)
        return json.dumps(response, ensure_ascii=False)

    tools = [
        StructuredTool.from_function(
            func=search_web,
            name="search_web",
            description=(
                "发起真实的网络搜索并抓取正文，消耗有限的网络预算。用于本地材料池已经没有新线索时，"
                "获取全新的候选来源。会自动清洗正文、切块，并顺带对同一个 query 触发一次本地检索+抽取。"
                "返回结果里的 web_budget_remaining 会持续下降到 0；到 0 后只能使用 search_local。"
            ),
        ),
        StructuredTool.from_function(
            func=search_local,
            name="search_local",
            description=(
                "在【全局共享材料池】（本问题以及其它并发问题已经抓取的所有正文）里换角度重新检索、"
                "重新打分排序，并对新出现的相关片段抽取证据、落库。不消耗网络预算，比 search_web 更省，"
                "换一种问法/角度通常能命中之前排不进结果的材料。"
            ),
        ),
        StructuredTool.from_function(
            func=done,
            name="done",
            description=(
                "认为已经拿到足够回答这个问题的证据时调用，结束本问题的检索循环。"
                "服务端仍会用机械覆盖门槛复核一遍，不是调用后立即视为通过。"
            ),
        ),
    ]
    if include_fetch_url:
        tools.append(
            StructuredTool.from_function(
                func=fetch_url,
                name="fetch_url",
                description=(
                    "直接抓取一个已知的具体 URL 正文（不发起搜索，不消耗 web 预算）。"
                    "当用户在请求里明确给出了一个参考链接时用这个工具去拿它的正文，"
                    "不要指望 search_web 的关键词搜索能凑巧搜到同一个页面。"
                ),
            )
        )
    return tools, session
