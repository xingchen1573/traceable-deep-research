"""Bounded question-level research orchestration."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .audit import submit_with_audit_context
from .coverage import assess_question_coverage, should_stop
from .evidence import (
    admit_evidence,
    apply_semantic_arbitration,
    authority_score,
    claim_similarity,
    comparison_subjects_compatible,
    evidence_semantic_arbitration_mode,
    is_direct_exact_evidence,
    merge_evidence_clusters,
    semantic_relation_allows_criterion,
)
from .models import ResearchQuestionState
from .source_policy import (
    boundary_query_terms,
    canonical_source_identity,
    canonical_project,
    compile_research_query,
    entity_match_metadata,
    source_query,
    targeted_query,
)
from .text_utils import extract_terms


RetrievalCallable = Callable[[ResearchQuestionState, list[str], int], dict[str, Any]]
BatchRetrievalCallable = Callable[[list[ResearchQuestionState], dict[str, list[str]], int], dict[str, Any]]
CoverageJudge = Callable[[ResearchQuestionState], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]


class QuestionResearchOrchestrator:
    def __init__(
        self,
        retrieval: RetrievalCallable,
        *,
        batch_retrieval: BatchRetrievalCallable | None = None,
        coverage_judge: CoverageJudge | None = None,
        progress_callback: ProgressCallback | None = None,
        parallelism: int = 3,
        agentic: bool = False,
        source_pool: Any = None,
        material_store: Any = None,
        job_id: str = "",
        user_request: str = "",
        research_depth: str = "normal",
        agentic_budget: Any = None,
        agentic_include_fetch_url: bool = False,
    ) -> None:
        self._retrieval = retrieval
        self._batch_retrieval = batch_retrieval
        self._coverage_judge = coverage_judge
        self._progress_callback = progress_callback
        self._parallelism = max(1, parallelism)
        # Opt-in per-question agentic retrieval path (see docs/modules/deep-research-step2-
        # retrieval-and-evidence-admission-discussion-v0.2.md). `agentic=False` (the default)
        # leaves `run()` on the exact same batch/simple retrieval branches as before this was
        # added -- these five fields are simply unused in that case.
        self._agentic = agentic
        self._agentic_source_pool = source_pool
        self._agentic_material_store = material_store
        self._agentic_job_id = job_id
        self._agentic_user_request = user_request
        self._agentic_research_depth = research_depth
        # `agentic_budget`/`agentic_include_fetch_url` let a caller override
        # `AgenticBudget.for_depth(research_depth)` and opt into the `fetch_url` tool -- used by
        # research_service.py's `_agentic_recon`/`_agentic_recon_for_section` (see
        # docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md), which run a single
        # synthetic "question" through `_run_question_agent` directly rather than through `run()`.
        # None/False here preserve every existing question_loop call site's behavior unchanged.
        self._agentic_budget = agentic_budget
        self._agentic_include_fetch_url = agentic_include_fetch_url


    def run(
        self,
        states: list[ResearchQuestionState],
        *,
        initial_queries: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        initial_queries = initial_queries or {}
        if self._agentic:
            completed = self._run_agentic(states)
            completed.sort(key=lambda item: states.index(item))
            return self.snapshot(completed, elapsed_ms=int((time.perf_counter() - started) * 1000))
        foundational, remaining = _execution_waves(states)
        completed: list[ResearchQuestionState] = []
        waves = (states,) if self._batch_retrieval else (foundational, remaining)
        for wave in waves:
            if not wave:
                continue
            if self._batch_retrieval:
                completed.extend(self._run_wave(wave, initial_queries))
            else:
                with ThreadPoolExecutor(max_workers=min(self._parallelism, len(wave))) as executor:
                    futures = {
                        submit_with_audit_context(
                            executor, self._run_question, state, initial_queries.get(state.question_id) or []
                        ): state
                        for state in wave
                    }
                    for future in as_completed(futures):
                        completed.append(future.result())
        completed.sort(key=lambda item: states.index(item))
        return self.snapshot(completed, elapsed_ms=int((time.perf_counter() - started) * 1000))

    def snapshot(
        self,
        states: list[ResearchQuestionState],
        *,
        elapsed_ms: int = 0,
    ) -> dict[str, Any]:
        """Serialize the current states after one or more bounded research passes."""
        completed = list(states)
        evidence_map = _global_evidence(completed)
        answer_status_counts = {"complete": 0, "partial": 0, "uncovered": 0}
        for state in completed:
            answer_status = str((state.coverage or {}).get("answer_status") or "").strip().lower()
            if answer_status not in answer_status_counts:
                answer_status = "complete" if state.status == "sufficient" else (
                    "partial" if state.accepted_evidence else "uncovered"
                )
            answer_status_counts[answer_status] += 1
        return {
            "question_states": [state.to_dict() for state in completed],
            "sources": _dedupe([item for state in completed for item in state.sources], ("url", "source_url")),
            "materials": _dedupe([item for state in completed for item in state.materials], ("url", "id")),
            "selected_chunks": _dedupe([item for state in completed for item in state.selected_chunks], ("id",)),
            "evidence_map": evidence_map,
            "rejected_evidence": [item for state in completed for item in state.rejected_evidence],
            "pipeline_diagnostics": {
                "mode": "agentic_retrieval" if self._agentic else "question_loop",
                "elapsed_ms": max(0, int(elapsed_ms)),
                "question_count": len(completed),
                "sufficient_count": sum(state.status == "sufficient" for state in completed),
                "exhausted_count": sum(state.status == "exhausted" for state in completed),
                "answer_status_counts": answer_status_counts,
                "round_count": sum(state.rounds_used for state in completed),
                "model_call_count": sum(state.model_calls for state in completed),
                "rejected_evidence_count": sum(len(state.rejected_evidence) for state in completed),
                **getattr(self, "_batch_diagnostics", {}),
            },
        }

    def _run_wave(
        self,
        wave: list[ResearchQuestionState],
        initial_queries: dict[str, list[str]],
    ) -> list[ResearchQuestionState]:
        if any(state.comparison_cells for state in wave):
            return self._run_comparison_wave(wave, initial_queries)
        pending = {
            state.question_id: _unique(initial_queries.get(state.question_id) or _criterion_queries(state))
            for state in wave
        }
        active = list(wave)
        for state in active:
            state.status = "researching"
        while active:
            queries_by_id: dict[str, list[str]] = {}
            contexts: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
            for state in active:
                state.rounds_used += 1
                queries = _next_queries(state, pending[state.question_id])
                _record_query_attempts(state, queries, state.rounds_used)
                state.attempted_queries.extend(query for query in queries if query not in state.attempted_queries)
                queries_by_id[state.question_id] = queries
                contexts[state.question_id] = (
                    [dict(item) for item in state.gap_actions if str(item.get("query") or "") in queries],
                    dict(state.coverage),
                )
            batch = self._batch_retrieval(active, queries_by_id, max(state.rounds_used for state in active))
            results = batch.get("results") or {}
            batch_diagnostics = batch.get("diagnostics") or {}
            previous_diagnostics = getattr(self, "_batch_diagnostics", {})
            retrieval_waves = list(previous_diagnostics.get("retrieval_waves") or [])
            retrieval_wave = batch_diagnostics.get("retrieval_wave")
            if isinstance(retrieval_wave, dict):
                retrieval_waves.append(dict(retrieval_wave))
            self._batch_diagnostics = {
                **previous_diagnostics,
                **batch_diagnostics,
                "retrieval_waves": retrieval_waves,
            }
            apply_started = time.perf_counter()
            round_results: dict[str, tuple[bool, list[str]]] = {}
            with ThreadPoolExecutor(max_workers=min(self._parallelism, len(active))) as executor:
                futures = {}
                for state in active:
                    executed_actions, coverage_before = contexts[state.question_id]
                    retrieval = results.get(state.question_id) or {}
                    chargeable = float((retrieval.get("_diagnostics") or {}).get("chargeable_elapsed_seconds") or 0.0)
                    elapsed = state.elapsed_seconds + max(0.0, chargeable)
                    futures[submit_with_audit_context(
                        executor,
                        self._apply_round,
                        state,
                        retrieval,
                        queries=queries_by_id[state.question_id],
                        executed_actions=executed_actions,
                        coverage_before=coverage_before,
                        elapsed=elapsed,
                    )] = state
                for future in as_completed(futures):
                    state = futures[future]
                    round_results[state.question_id] = future.result()
            apply_elapsed_ms = int((time.perf_counter() - apply_started) * 1000)
            if retrieval_waves:
                completed_wave = {
                    **retrieval_waves[-1],
                    "post_retrieval_apply_elapsed_ms": apply_elapsed_ms,
                }
                if "wave_elapsed_ms" in completed_wave:
                    completed_wave["total_wave_elapsed_ms"] = (
                        int(completed_wave["wave_elapsed_ms"]) + apply_elapsed_ms
                    )
                retrieval_waves[-1] = completed_wave
                self._batch_diagnostics["retrieval_wave"] = completed_wave
                self._batch_diagnostics["retrieval_waves"] = retrieval_waves
            next_active: list[ResearchQuestionState] = []
            for state in active:
                stop, suggestions = round_results[state.question_id]
                if stop:
                    self._finalize_state(state)
                    self._notify_progress(state)
                    continue
                pending[state.question_id] = _unique([*_gap_queries(state), *suggestions])
                scheduled_queries = set(pending[state.question_id][:2])
                state.generated_gap_action_count += sum(
                    str(item.get("query") or "") in scheduled_queries for item in state.gap_actions
                )
                next_active.append(state)
                self._notify_progress(state)
            active = next_active
        return wave

    def _run_comparison_wave(
        self,
        wave: list[ResearchQuestionState],
        initial_queries: dict[str, list[str]],
    ) -> list[ResearchQuestionState]:
        pending = {
            state.question_id: _unique(initial_queries.get(state.question_id) or [])
            for state in wave
        }
        active = list(wave)
        for state in active:
            state.status = "researching"
        while active:
            initial_pending = any(_comparison_initial_queries(state) for state in active)
            candidates_by_id: dict[str, list[str]] = {}
            for state in active:
                if initial_pending:
                    candidates = _comparison_initial_queries(state)
                else:
                    candidates = [
                        *_comparison_gap_queries(state),
                        *_remaining_comparative_queries(state),
                    ]
                candidates_by_id[state.question_id] = _unique(candidates)
            queries_by_id = _allocate_comparison_wave_queries(
                active,
                candidates_by_id,
                max_queries=_comparison_wave_query_limit(active),
            )
            scheduled = [state for state in active if queries_by_id.get(state.question_id)]
            if not scheduled:
                for state in active:
                    _update_comparison_cells(state, force_terminal=True)
                    self._finalize_state(state)
                    state.status = "sufficient" if _comparison_cells_complete(state) else "exhausted"
                    state.stop_reason = "comparison_cells_complete" if _comparison_cells_complete(state) else "budget_exhausted"
                    self._notify_progress(state)
                break

            contexts: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
            for state in scheduled:
                state.rounds_used += 1
                queries = queries_by_id[state.question_id]
                _record_query_attempts(state, queries, state.rounds_used)
                _record_comparison_query_attempts(state, queries, state.rounds_used)
                state.attempted_queries.extend(query for query in queries if query not in state.attempted_queries)
                contexts[state.question_id] = ([], dict(state.coverage))

            batch = self._batch_retrieval(
                scheduled,
                queries_by_id,
                max(state.rounds_used for state in scheduled),
            )
            results = batch.get("results") or {}
            batch_diagnostics = batch.get("diagnostics") or {}
            previous_diagnostics = getattr(self, "_batch_diagnostics", {})
            retrieval_waves = list(previous_diagnostics.get("retrieval_waves") or [])
            retrieval_wave = batch_diagnostics.get("retrieval_wave")
            if isinstance(retrieval_wave, dict):
                retrieval_waves.append(dict(retrieval_wave))
            self._batch_diagnostics = {
                **previous_diagnostics,
                **batch_diagnostics,
                "retrieval_waves": retrieval_waves,
            }

            round_results: dict[str, tuple[bool, list[str]]] = {}
            with ThreadPoolExecutor(max_workers=min(self._parallelism, len(scheduled))) as executor:
                futures = {}
                for state in scheduled:
                    executed_actions, coverage_before = contexts[state.question_id]
                    retrieval = results.get(state.question_id) or {}
                    chargeable = float((retrieval.get("_diagnostics") or {}).get("chargeable_elapsed_seconds") or 0.0)
                    futures[submit_with_audit_context(
                        executor,
                        self._apply_round,
                        state,
                        retrieval,
                        queries=queries_by_id[state.question_id],
                        executed_actions=executed_actions,
                        coverage_before=coverage_before,
                        elapsed=state.elapsed_seconds + max(0.0, chargeable),
                    )] = state
                for future in as_completed(futures):
                    state = futures[future]
                    round_results[state.question_id] = future.result()

            next_active: list[ResearchQuestionState] = []
            for state in active:
                if state not in scheduled:
                    next_active.append(state)
                    continue
                stop, _suggestions = round_results[state.question_id]
                if stop:
                    self._finalize_state(state)
                    self._notify_progress(state)
                else:
                    pending[state.question_id] = _unique([
                        *_comparison_gap_queries(state),
                        *_remaining_comparative_queries(state),
                    ])
                    next_active.append(state)
                    self._notify_progress(state)
            active = next_active
        return wave

    def _run_question(self, state: ResearchQuestionState, initial_queries: list[str]) -> ResearchQuestionState:
        started = time.perf_counter()
        state.status = "researching"
        pending_queries = _unique(initial_queries or [state.question])
        while True:
            state.rounds_used += 1
            queries = _next_queries(state, pending_queries)
            _record_query_attempts(state, queries, state.rounds_used)
            state.attempted_queries.extend(query for query in queries if query not in state.attempted_queries)
            executed_actions = [
                dict(item)
                for item in state.gap_actions
                if str(item.get("query") or "") in queries
            ]
            coverage_before = dict(state.coverage)
            retrieval = self._retrieval(state, queries, state.rounds_used)
            stop, suggestions = self._apply_round(
                state,
                retrieval,
                queries=queries,
                executed_actions=executed_actions,
                coverage_before=coverage_before,
                elapsed=time.perf_counter() - started,
            )
            if stop:
                self._finalize_state(state)
                self._notify_progress(state)
                break
            self._notify_progress(state)
            pending_queries = _unique([*_gap_queries(state), *suggestions])
            scheduled_queries = set(pending_queries[:2])
            state.generated_gap_action_count += sum(
                str(item.get("query") or "") in scheduled_queries for item in state.gap_actions
            )
        if state.status == "researching":
            self._finalize_state(state)
        return state

    def _run_agentic(self, states: list[ResearchQuestionState]) -> list[ResearchQuestionState]:
        """Per-question agentic retrieval, replacing `_run_wave`'s round-synchronized loop.

        See docs/modules/deep-research-step2-retrieval-and-evidence-admission-discussion-v0.2.md
        section 7 ("wave 调度整合"): production already runs every question as a single
        `_batch_retrieval` wave with no real foundational/remaining split, so there is no
        round-barrier concept to preserve here -- this reuses the exact same ThreadPoolExecutor
        concurrency shell the non-batch `run()` branch already uses (orchestrator.py's original
        `_run_question` dispatch), just calling `_run_question_agent` instead.
        """
        for state in states:
            state.status = "researching"
        completed: list[ResearchQuestionState] = []
        with ThreadPoolExecutor(max_workers=min(self._parallelism, len(states))) as executor:
            futures = {
                submit_with_audit_context(executor, self._run_question_agent, state): state
                for state in states
            }
            for future in as_completed(futures):
                completed.append(future.result())
        return completed

    def _run_question_agent(
        self,
        state: ResearchQuestionState,
        *,
        system_prompt: str = "",
        initial_message: str = "",
    ) -> ResearchQuestionState:
        """Run one question through the per-question agentic retrieval loop.

        `create_agent` gives the model a persistent, multi-turn ReAct session (unlike
        `workers.py`'s single-shot `web_worker_tool`) so it can remember which queries it already
        tried and what it has already committed across several `search_web`/`search_local`
        rounds -- see the doc's "落地设计" section 1/"bind_tools 封装的具体接口设计".

        `system_prompt`/`initial_message` let a caller override the question_loop defaults below
        -- used by `run_single_agent_session` (recon's synthetic "question" is not really a
        question_loop question, so it needs its own system prompt/briefing).
        """
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage

        from middleware import ContextCompactMiddleware, LoopGuardMiddleware, MicroCompactMiddleware
        from prompts import QUESTION_RESEARCH_AGENT_PROMPT

        from .agentic_model import build_agentic_model
        from .agentic_tools import build_question_agent_tools

        started = time.perf_counter()
        state.status = "researching"
        tools, session = build_question_agent_tools(
            job_id=self._agentic_job_id,
            user_request=self._agentic_user_request,
            state=state,
            pool=self._agentic_source_pool,
            material_store=self._agentic_material_store,
            depth=self._agentic_research_depth,
            budget=self._agentic_budget,
            include_fetch_url=self._agentic_include_fetch_url,
        )
        model = build_agentic_model()
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt or QUESTION_RESEARCH_AGENT_PROMPT,
            middleware=[
                MicroCompactMiddleware(),
                ContextCompactMiddleware(model=model),
                LoopGuardMiddleware(),
            ],
        )
        # 【硬性要求】thread_id must be passed explicitly -- without it, `middleware._thread_id`
        # silently falls back to the ambient `current_session_id()` contextvar, which does not
        # propagate across the ThreadPoolExecutor worker threads `_run_agentic` submits into. All
        # concurrent questions would then share one LoopGuardMiddleware/ContextCompactMiddleware
        # state key ("default") and cross-contaminate each other's circuit breaker/compaction
        # state -- see the doc's "上下文管理" section for the confirmed risk this guards against.
        thread_id = f"{self._agentic_job_id}:{state.question_id}"
        # Bound LangGraph's own step count as a second, model-independent safety net beneath the
        # tools' cooperative budget denials (a model that ignores "budget exhausted" responses
        # would otherwise only be stopped by LoopGuardMiddleware noticing identical repeats).
        recursion_limit = 2 * (session.budget.max_web_calls + session.budget.max_local_calls) + 20
        initial_message = initial_message or _question_agent_briefing(state, self._agentic_user_request)


        def _invoke(thread_id_for_attempt: str) -> Any:
            return agent.invoke(
                {"messages": [HumanMessage(content=initial_message)]},
                config={"configurable": {"thread_id": thread_id_for_attempt}, "recursion_limit": recursion_limit},
            )

        agent_error: Exception | None = None
        recovered_from_error = False
        for attempt in range(2):  # 1 original + at most 1 self-heal retry, see guard below
            # A retry attempt gets its own thread_id suffix, not a resume of the same one: the
            # first attempt failed before completing, so its LangGraph checkpoint (if any exists
            # at all) is a broken half-turn -- resuming into it risks replaying corrupted state
            # rather than starting the ReAct loop cleanly.
            attempt_thread_id = thread_id if attempt == 0 else f"{thread_id}:retry{attempt}"
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = submit_with_audit_context(executor, _invoke, attempt_thread_id)
                try:
                    future.result(timeout=session.time_remaining())
                    if agent_error is not None:
                        recovered_from_error = True
                    agent_error = None
                    break
                except Exception as exc:
                    # Covers both a hard wall-clock timeout (three-layer budget, section 4) and any
                    # exception the agent loop itself raised (e.g. the reasoning model's own retries
                    # exhausted). Either way, evidence already committed by completed search_web/
                    # search_local tool calls is safe on `state` -- only "decide the next step" was
                    # lost, never previously-admitted evidence (decision 6: extraction+admission is
                    # unconditional inside the tools, not something a failed turn can undo).
                    agent_error = exc
                    # Only worth a self-heal retry when the session made ZERO progress (never got
                    # past the agent's very first decision-step model call, so no tool ever ran)
                    # AND there is still time budget left -- a session that already burned rounds
                    # and then hit a late failure keeps whatever it committed; retrying that would
                    # just re-run a fresh conversation over work already done, wasting budget.
                    if session.call_counter > 0 or session.time_remaining() <= 0:
                        break

        state.rounds_used = session.call_counter
        state.model_calls += session.model_calls
        state.elapsed_seconds = round(time.perf_counter() - started, 3)
        # 服务端强制核验：done() 只是 agent 请求结束循环，真正的充分性判断仍由
        # assess_question_coverage（经 _finalize_state）机械核验一遍 -- 对应决议"coverage 双层"。
        self._finalize_state(state)
        state.diagnostics["agentic_session"] = {
            "web_calls_used": session.web_calls_used,
            "local_calls_used": session.local_calls_used,
            "done_called": session.done_flag,
            "done_reason": session.done_reason,
        }
        if agent_error is not None:
            # _finalize_state() above fully replaces state.diagnostics, so this must be set
            # after it, not before (same reason "agentic_session" above is set post-finalize).
            state.diagnostics["agent_error"] = {
                "type": type(agent_error).__name__,
                "message": str(agent_error)[:300],
            }
        elif recovered_from_error:
            # The first attempt died with zero progress but the self-heal retry succeeded --
            # worth keeping visible for future analysis of how often this actually happens, even
            # though the question itself ends up fine.
            state.diagnostics["agent_error_recovered"] = True
        state.status = "sufficient" if state.coverage.get("sufficient") else "exhausted"
        if session.done_flag:
            state.stop_reason = session.done_reason
        elif agent_error is not None and session.call_counter == 0:
            # Distinguish "the agent never even got to decide anything" from a genuine budget
            # exhaustion after real rounds -- "budget_exhausted" here would hide a real infra
            # failure (see state.diagnostics["agent_error"] for the underlying exception).
            state.stop_reason = "agent_invocation_failed"
        else:
            state.stop_reason = "budget_exhausted"
        self._notify_progress(state)
        return state

    def run_single_agent_session(
        self,
        state: ResearchQuestionState,
        *,
        system_prompt: str,
        initial_message: str,
    ) -> ResearchQuestionState:
        """Public entry point for a one-off agentic session over a synthetic "question" that is
        not part of a question_loop plan -- used by research_service.py's `_agentic_recon`/
        `_agentic_recon_for_section` (docs/modules/deep-research-plan-recon-agentic-upgrade-
        v0.1.md). Thin wrapper around `_run_question_agent` so recon code does not need to reach
        into a private method or duplicate its retry/thread-id/finalize logic.
        """
        return self._run_question_agent(state, system_prompt=system_prompt, initial_message=initial_message)

    def _notify_progress(self, state: ResearchQuestionState) -> None:
        if not self._progress_callback:
            return
        try:
            self._progress_callback(
                {
                    "question_id": state.question_id,
                    "section_id": state.section_id,
                    "section_title": state.section_title,
                    "question": state.question,
                    "rounds_used": state.rounds_used,
                    "status": state.status,
                    "evidence_count": len(state.accepted_evidence),
                    "source_count": len(state.sources),
                }
            )
        except Exception:
            pass

    def _apply_round(
        self,
        state: ResearchQuestionState,
        retrieval: dict[str, Any],
        *,
        queries: list[str],
        executed_actions: list[dict[str, Any]],
        coverage_before: dict[str, Any],
        elapsed: float,
    ) -> tuple[bool, list[str]]:
            apply_started = time.perf_counter()
            state.model_calls += int((retrieval.get("_diagnostics") or {}).get("model_call_count") or 1)
            chunks, evidence = _namespace_round(retrieval, state, state.rounds_used)
            state.sources = _merge(state.sources, retrieval.get("sources") or [], ("url", "source_url"))
            state.materials = _merge(state.materials, retrieval.get("materials") or [], ("url", "id"))
            attempted_urls = [str(item) for item in retrieval.get("attempted_urls") or [] if str(item).startswith("http")]
            if not attempted_urls:
                attempted_urls = [
                    str(item.get("url") or item.get("source_url") or "")
                    for item in retrieval.get("materials") or []
                    if str(item.get("url") or item.get("source_url") or "").startswith("http")
                ]
            state.attempted_fetch_urls = _unique([*state.attempted_fetch_urls, *attempted_urls])
            state.selected_chunks = _merge(state.selected_chunks, chunks, ("id",))
            previous_domains = _evidence_domains(state.accepted_evidence)
            previous_source_identities = _evidence_source_identities(state.accepted_evidence)
            accepted, rejected = admit_evidence(
                evidence,
                chunks=chunks,
                question_id=state.question_id,
                section_id=state.section_id,
                question=state.question,
                target_subject=state.topic_anchor,
                comparison_subjects=state.comparison_subjects,
                expected_comparison_axis=state.comparison_axis,
                valid_criterion_ids={criterion.criterion_id for criterion in state.criteria},
            )
            semantic_mode = evidence_semantic_arbitration_mode()
            arbitration_result = {"status": "not_needed", "verdicts": {}, "reason": ""}
            arbitration_called = False
            conflict_items = [
                item for item in accepted
                if (item.get("semantic_arbitration") or {}).get("conflict_reasons")
            ]
            if semantic_mode != "off" and conflict_items:
                if state.model_calls >= state.budget.max_model_calls:
                    arbitration_result = {
                        "status": "unavailable",
                        "verdicts": {},
                        "reason": "question_model_call_budget_exhausted",
                    }
                elif elapsed >= state.budget.timeout_seconds:
                    arbitration_result = {
                        "status": "unavailable",
                        "verdicts": {},
                        "reason": "question_timeout_budget_exhausted",
                    }
                else:
                    from workers import arbitrate_evidence_semantics

                    state.model_calls += 1
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
                            float(os.getenv("RESEARCH_EVIDENCE_SEMANTIC_ARBITRATION_TIMEOUT_SECONDS", "45")),
                            state.budget.timeout_seconds - elapsed,
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
                arbitration_rounds = state.diagnostics.setdefault("semantic_arbitration_rounds", [])
                arbitration_rounds.append({
                    "round_index": state.rounds_used,
                    "mode": semantic_mode,
                    "eligible_count": len(accepted) + len(semantic_rejected),
                    "conflict_count": len(conflict_items),
                    "call_count": int(arbitration_called),
                    "status": arbitration_result.get("status"),
                    "reason": arbitration_result.get("reason"),
                    "resolved_count": len(arbitration_result.get("verdicts") or {}),
                    "downgraded_count": sum(
                        (item.get("semantic_arbitration") or {}).get("fallback", {}).get("support_relation")
                        == "inference_support"
                        for item in [*accepted, *semantic_rejected]
                    ),
                    "quarantined_count": len(semantic_rejected),
                })
            state.accepted_evidence, cluster_diagnostics = merge_evidence_clusters(state.accepted_evidence, accepted)
            state.rejected_evidence.extend(rejected)
            new_domains = _evidence_domains(state.accepted_evidence) - previous_domains
            new_source_identities = _evidence_source_identities(state.accepted_evidence) - previous_source_identities
            if cluster_diagnostics["novel_cluster_count"] == 0 and not new_source_identities:
                state.stagnant_rounds += 1
            else:
                state.stagnant_rounds = 0
            state.access_issues = _merge_access_issues(state.access_issues, retrieval.get("provider_attempts") or [])
            _complete_comparison_query_attempts(state, queries, retrieval)
            state.diagnostics["last_cluster_merge"] = cluster_diagnostics
            chunk_ranker = (retrieval.get("_diagnostics") or {}).get("chunk_ranker")
            if isinstance(chunk_ranker, dict) and chunk_ranker:
                state.diagnostics["chunk_ranker"] = _merge_chunk_ranker_diagnostics(
                    state.diagnostics.get("chunk_ranker"),
                    chunk_ranker,
                )
            suggestions = _suggested_queries(retrieval)
            if self._coverage_judge and state.accepted_evidence:
                semantic = self._coverage_judge(state)
                state.model_calls += 1
                if semantic:
                    state.diagnostics["semantic_coverage"] = semantic
                    suggestions = [*suggestions, *[str(item) for item in semantic.get("suggested_queries") or []]]
            _update_criterion_ledger(state, semantic=state.diagnostics.get("semantic_coverage"))
            _update_comparison_cells(state)
            state.suggested_queries = _unique([*state.suggested_queries, *suggestions])[-6:]
            elapsed += time.perf_counter() - apply_started
            state.elapsed_seconds = round(elapsed, 3)
            stop, reason = should_stop(state, elapsed_seconds=elapsed)
            if state.comparison_cells:
                pending_initial = bool(_comparison_initial_queries(state))
                pending_followup = bool(
                    _comparison_gap_queries(state) or _remaining_comparative_queries(state)
                )
                if pending_initial:
                    stop, reason = False, ""
                elif pending_followup and state.rounds_used < state.budget.max_rounds:
                    stop, reason = False, ""
                elif _comparison_cells_complete(state):
                    stop, reason = True, "comparison_cells_complete"
                else:
                    stop, reason = True, "budget_exhausted"
            state.gap_actions = _plan_gap_actions(state)
            state.round_budgets.append({
                "round_index": state.rounds_used,
                "query_count": len(queries),
                "attempted_url_count": len(attempted_urls),
                "total_attempted_url_count": len(state.attempted_fetch_urls),
                "material_count": len(state.materials),
                "model_calls": state.model_calls,
            })
            for action in executed_actions:
                state.gap_execution_log.append({
                    **action,
                    "round_index": state.rounds_used,
                    "coverage_before": coverage_before,
                    "coverage_after": dict(state.coverage),
                    "new_source_count": len(new_source_identities),
                    "new_domain_count": len(new_domains),
                    "new_claim_cluster_count": cluster_diagnostics["novel_cluster_count"],
                })
            if stop:
                state.stop_reason = reason
                state.status = "sufficient" if (
                    reason.startswith("sufficient_evidence") or reason == "comparison_cells_complete"
                ) else "exhausted"
            return stop, suggestions

    def _finalize_state(self, state: ResearchQuestionState) -> None:
        semantic_retry_attempted = False
        semantic_coverage = state.diagnostics.get("semantic_coverage")
        if self._coverage_judge and state.accepted_evidence and not isinstance(semantic_coverage, dict):
            # Per-round semantic judging may occasionally return no JSON during provider pressure.
            # Make one final bounded attempt so a genuinely answerable question is not left at the
            # conservative unjudged=partial fallback solely because an earlier judge call failed.
            semantic_retry_attempted = True
            state.model_calls += 1
            semantic_coverage = self._coverage_judge(state)
            if semantic_coverage:
                state.diagnostics["semantic_coverage"] = semantic_coverage
                _update_criterion_ledger(state, semantic=semantic_coverage)
        state.coverage = assess_question_coverage(state)
        _update_comparison_cells(state, force_terminal=True)
        if state.comparison_cells:
            state.coverage["comparison_cells"] = [
                {
                    "cell_id": cell.cell_id,
                    "subject": cell.subject,
                    "axis": cell.axis,
                    "status": cell.status,
                    "evidence_ids": list(cell.evidence_ids),
                    "criteria": [
                        {
                            "criterion_id": criterion.criterion_id,
                            "description": criterion.description,
                            "status": criterion.status,
                            "evidence_ids": list(criterion.evidence_ids),
                        }
                        for criterion in cell.criteria
                    ],
                    "attempt_count": len(cell.attempts),
                    "unavailable_reasons": list(cell.unavailable_reasons),
                }
                for cell in state.comparison_cells
            ]
        state.covered_aspects = list(state.coverage.get("covered_aspects") or [])
        state.weak_aspects = list(state.coverage.get("weak_aspects") or [])
        state.gap_actions = _plan_gap_actions(state)
        semantic_coverage = state.diagnostics.get("semantic_coverage")
        last_cluster_merge = state.diagnostics.get("last_cluster_merge")
        chunk_ranker = state.diagnostics.get("chunk_ranker")
        coverage_evidence_selection = state.diagnostics.get("coverage_evidence_selection")
        semantic_arbitration_rounds = state.diagnostics.get("semantic_arbitration_rounds")
        state.diagnostics = {
            "rounds_used": state.rounds_used,
            "elapsed_ms": int(state.elapsed_seconds * 1000),
            "accepted_evidence_count": len(state.accepted_evidence),
            "rejected_evidence_count": len(state.rejected_evidence),
            "attempted_query_count": len(state.attempted_queries),
            "semantic_coverage": semantic_coverage,
            "semantic_coverage_final_retry_attempted": semantic_retry_attempted,
            "semantic_coverage_unavailable": semantic_retry_attempted and not isinstance(semantic_coverage, dict),
            "coverage_evidence_selection": coverage_evidence_selection,
            "semantic_arbitration_rounds": semantic_arbitration_rounds,
            "last_cluster_merge": last_cluster_merge,
            "chunk_ranker": chunk_ranker,
            "stagnant_rounds": state.stagnant_rounds,
            "claim_cluster_count": len({str(item.get("claim_cluster_id") or "") for item in state.accepted_evidence if item.get("claim_cluster_id")}),
            "access_issue_count": len(state.access_issues),
            "attempted_fetch_url_count": len(state.attempted_fetch_urls),
            "targeted_followup_executed_count": len(state.gap_execution_log),
            "generated_gap_action_count": state.generated_gap_action_count,
            "criterion_count": len(state.criteria),
            "hard_criterion_count": sum(item.level == "hard" for item in state.criteria),
            "covered_criterion_count": sum(item.status == "covered" for item in state.criteria),
        }


def _merge_chunk_ranker_diagnostics(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    previous = previous if isinstance(previous, dict) else {}
    totals = {
        key: int(previous.get(key) or 0) + int(current.get(key) or 0)
        for key in (
            "question_count",
            "call_count",
            "lexical_candidate_count",
            "semantic_candidate_count",
            "rrf_union_count",
            "entity_rejected_count",
            "strong_match_count",
            "protected_reinsert_count",
        )
    }
    modes = [str(value) for value in (previous.get("mode"), current.get("mode")) if str(value or "")]
    reasons: list[str] = []
    for item in (previous.get("ranker_degraded_reasons") or []):
        if str(item) and str(item) not in reasons:
            reasons.append(str(item))
    for item in (current.get("ranker_degraded_reasons") or []):
        if str(item) and str(item) not in reasons:
            reasons.append(str(item))
    questions = {
        **(previous.get("questions") or {}),
        **(current.get("questions") or {}),
    }
    return {
        "mode": modes[0] if modes and all(mode == modes[0] for mode in modes) else ("mixed" if modes else ""),
        **totals,
        "ranker_degraded": bool(previous.get("ranker_degraded") or current.get("ranker_degraded")),
        "ranker_degraded_reasons": reasons,
        "questions": questions,
    }


def _execution_waves(states: list[ResearchQuestionState]) -> tuple[list[ResearchQuestionState], list[ResearchQuestionState]]:
    foundational = [state for state in states if state.section_id == "sec_01"]
    if len(foundational) > 2:
        foundational = foundational[:2]
    foundational_ids = {id(state) for state in foundational}
    return foundational, [state for state in states if id(state) not in foundational_ids]


def _comparison_wave_query_limit(states: list[ResearchQuestionState]) -> int:
    for state in states:
        if state.comparison_budget:
            return max(1, int(state.comparison_budget.get("max_queries_per_wave") or 1))
    return 1


def _allocate_comparison_wave_queries(
    states: list[ResearchQuestionState],
    candidates_by_id: dict[str, list[str]],
    *,
    max_queries: int,
) -> dict[str, list[str]]:
    queues = {
        state.question_id: [
            query for query in candidates_by_id.get(state.question_id) or []
            if query and query not in state.attempted_queries
        ]
        for state in states
    }
    output = {state.question_id: [] for state in states}
    while sum(len(items) for items in output.values()) < max_queries:
        progressed = False
        for state in states:
            queue = queues[state.question_id]
            if not queue:
                continue
            output[state.question_id].append(queue.pop(0))
            progressed = True
            if sum(len(items) for items in output.values()) >= max_queries:
                break
        if not progressed:
            break
    return output


def _comparison_initial_queries(state: ResearchQuestionState) -> list[str]:
    target = max(1, int(state.comparison_budget.get("initial_queries_per_cell") or 1))
    return [
        cell.initial_query
        for cell in state.comparison_cells
        if cell.initial_query
        and sum(item.get("attempt_type") == "initial" for item in cell.attempts) < target
    ]


def _comparison_gap_query_rows(
    state: ResearchQuestionState,
) -> list[tuple[str, Any, Any]]:
    max_gap = max(0, int(state.comparison_budget.get("max_gap_queries_per_cell") or 0))
    rows: list[tuple[str, Any, Any]] = []
    for cell in state.comparison_cells:
        if not cell.attempts or cell.status == "complete":
            continue
        gap_attempts = sum(item.get("attempt_type") == "gap" for item in cell.attempts)
        if gap_attempts >= max_gap:
            continue
        criterion = next((item for item in cell.criteria if item.status != "covered"), None)
        if criterion is None:
            continue
        query = targeted_query(
            topic_anchor=cell.subject,
            missing_fact=(
                f"{cell.axis} {criterion.query_hint or criterion.description} "
                f"independent evidence followup {gap_attempts + 1}"
            ),
            source_type=(criterion.preferred_source_types or state.preferred_source_types or [""])[0],
        )
        if query:
            rows.append((query, cell, criterion))
    return rows


def _comparison_gap_queries(state: ResearchQuestionState) -> list[str]:
    return [query for query, _cell, _criterion in _comparison_gap_query_rows(state)]


def _remaining_comparative_queries(state: ResearchQuestionState) -> list[str]:
    return [query for query in state.comparative_queries if query not in state.attempted_queries]


def _record_comparison_query_attempts(
    state: ResearchQuestionState,
    queries: list[str],
    round_index: int,
) -> None:
    gap_rows = {query: (cell, criterion) for query, cell, criterion in _comparison_gap_query_rows(state)}
    for query in queries:
        initial_cell = next((cell for cell in state.comparison_cells if cell.initial_query == query), None)
        if initial_cell is not None:
            initial_cell.attempts.append({
                "round_index": round_index,
                "query": query,
                "attempt_type": "initial",
            })
            continue
        if query in state.comparative_queries:
            state.comparative_query_attempts.append({
                "round_index": round_index,
                "query": query,
                "attempt_type": "comparative",
            })
            continue
        row = gap_rows.get(query)
        if row is None:
            continue
        cell, criterion = row
        attempt = {
            "round_index": round_index,
            "query": query,
            "attempt_type": "gap",
            "criterion_id": criterion.criterion_id,
        }
        cell.attempts.append(attempt)
        criterion.attempts.append(dict(attempt))


def _complete_comparison_query_attempts(
    state: ResearchQuestionState,
    queries: list[str],
    retrieval: dict[str, Any],
) -> None:
    if not state.comparison_cells:
        return
    success, reasons = _retrieval_operational_status(retrieval)
    query_set = set(queries)
    for cell in state.comparison_cells:
        for attempt in cell.attempts:
            if attempt.get("query") in query_set and "operational_success" not in attempt:
                attempt["operational_success"] = success
                attempt["failure_reasons"] = reasons
        if not success:
            cell.unavailable_reasons = list(dict.fromkeys([*cell.unavailable_reasons, *reasons]))
    for attempt in state.comparative_query_attempts:
        if attempt.get("query") in query_set and "operational_success" not in attempt:
            attempt["operational_success"] = success
            attempt["failure_reasons"] = reasons


def _retrieval_operational_status(retrieval: dict[str, Any]) -> tuple[bool, list[str]]:
    if any(retrieval.get(key) for key in ("sources", "materials", "selected_chunks", "evidence_map", "attempted_urls")):
        return True, []
    attempts = [item for item in retrieval.get("provider_attempts") or [] if isinstance(item, dict)]
    failure_statuses = {"error", "failed", "denied", "timeout", "unavailable"}
    success_statuses = {"ok", "success", "succeeded", "completed", "empty"}
    if any(str(item.get("status") or "").lower() in success_statuses for item in attempts):
        return True, []
    reasons = [
        str(item.get("reason") or item.get("status") or "provider_unavailable")[:240]
        for item in attempts
        if str(item.get("status") or "").lower() in failure_statuses
    ]
    return False, list(dict.fromkeys(reasons or ["retrieval_unavailable"]))


def _update_comparison_cells(state: ResearchQuestionState, *, force_terminal: bool = False) -> None:
    if not state.comparison_cells:
        return
    semantic_live = evidence_semantic_arbitration_mode() == "live"
    for cell in state.comparison_cells:
        matching = [
            evidence for evidence in state.accepted_evidence
            if _comparison_subject_matches(str(evidence.get("subject") or ""), cell.subject)
            and _comparison_axis_matches(str(evidence.get("comparison_axis") or ""), cell.axis)
            and (not semantic_live or semantic_relation_allows_criterion(evidence))
            and str(evidence.get("criterion_binding_status") or "") in {
                "verified_target",
                "verified_target_component",
                "verified_comparison",
            }
            and str(evidence.get("comparison_axis_binding_status") or "") == "verified_axis"
        ]
        for evidence in matching:
            evidence["matrix_cell_id"] = cell.cell_id
        evidence_ids = list(dict.fromkeys(
            str(item.get("id") or item.get("_coverage_evidence_id") or "")
            for item in matching
            if str(item.get("id") or item.get("_coverage_evidence_id") or "")
        ))
        cell.evidence_ids = evidence_ids
        for criterion in cell.criteria:
            supporting = [
                item for item in matching
                if criterion.criterion_id in {str(value) for value in item.get("criterion_ids") or []}
            ]
            if not semantic_live:
                criterion.evidence_ids = list(dict.fromkeys(
                    str(item.get("id") or item.get("_coverage_evidence_id") or "")
                    for item in supporting
                    if str(item.get("id") or item.get("_coverage_evidence_id") or "")
                ))
                criterion.status = "covered" if criterion.evidence_ids else "uncovered"
                continue
            direct_ids = [
                str(item.get("id") or item.get("_coverage_evidence_id") or "")
                for item in supporting
                if is_direct_exact_evidence(item)
                and str(item.get("id") or item.get("_coverage_evidence_id") or "")
            ]
            inference_ids = [
                str(item.get("id") or item.get("_coverage_evidence_id") or "")
                for item in supporting
                if not is_direct_exact_evidence(item)
                and str(item.get("id") or item.get("_coverage_evidence_id") or "")
            ]
            criterion.evidence_ids = list(dict.fromkeys([*direct_ids, *inference_ids]))
            criterion.status = "covered" if direct_ids else "partial" if inference_ids else "uncovered"

        if cell.criteria and all(item.status == "covered" for item in cell.criteria):
            cell.status = "complete"
            continue
        if matching:
            cell.status = "partial"
            continue
        completed_attempts = [item for item in cell.attempts if "operational_success" in item]
        if completed_attempts and all(not bool(item.get("operational_success")) for item in completed_attempts):
            cell.status = "unavailable"
            continue
        max_gap = max(0, int(state.comparison_budget.get("max_gap_queries_per_cell") or 0))
        gap_attempts = sum(item.get("attempt_type") == "gap" for item in cell.attempts)
        if force_terminal or (completed_attempts and gap_attempts >= max_gap):
            cell.status = "exhausted"
        else:
            cell.status = "uncovered"


def _comparison_subject_matches(actual: str, expected: str) -> bool:
    if not actual or not expected:
        return False
    if comparison_subjects_compatible(actual, expected):
        return True
    return float(entity_match_metadata(title=actual, snippet=actual, topic=expected).get("entity_match_score") or 0.0) >= 0.75


def _comparison_axis_matches(actual: str, expected: str) -> bool:
    left = "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", actual.lower()))
    right = "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", expected.lower()))
    return bool(left and right and (left in right or right in left))


def _comparison_cells_complete(state: ResearchQuestionState) -> bool:
    return bool(state.comparison_cells) and all(cell.status == "complete" for cell in state.comparison_cells)


def _next_queries(state: ResearchQuestionState, pending: list[str]) -> list[str]:
    queries = [query for query in pending if query and query not in state.attempted_queries]
    if not queries:
        queries = _gap_queries(state)
    return _unique(queries)[:2]


def _question_agent_briefing(state: ResearchQuestionState, user_request: str) -> str:
    """Build the per-question agent's initial human message: what to research and what "done"
    means, so the model has criteria to check itself against (decision 4: agent self-judgment is
    a first layer, not a replacement for the server-side coverage gate)."""
    lines = [
        f"用户的整体研究需求：{user_request}" if user_request else "",
        f"你负责的研究问题：{state.question}",
        f"这个问题在报告里的目的：{state.purpose}",
        "研究标准（hard 是完成本问题所必需，soft/optional 是增强项）：",
        *[f"- [{item.level}] {item.description}" for item in state.criteria],
    ]
    if state.preferred_source_types:
        lines.append(f"优先来源类型：{', '.join(state.preferred_source_types)}")
    if state.topic_anchor:
        lines.append(f"主题锚点（检索时可以带上，避免跑题）：{state.topic_anchor}")
    lines.append(
        f"证据数量参考：{state.min_evidence} 条独立证据、{state.min_sources} 个来源；"
        "这些是广度提示，不得在 hard 标准已回答后继续为凑数量检索。"
    )
    return "\n".join(line for line in lines if line)


def _criterion_queries(state: ResearchQuestionState) -> list[str]:
    criteria = [item for item in state.criteria if item.level == "hard"]
    criteria.extend(item for item in state.criteria if item.level == "soft")
    criteria.extend(item for item in state.criteria if item.level == "optional")
    queries = [
        compile_research_query(
            topic_anchor=state.topic_anchor or state.question,
            content=criterion.query_hint or criterion.description,
            source_type=(criterion.preferred_source_types or state.preferred_source_types or [""])[0],
        )
        for criterion in criteria[:2]
    ]
    return _unique([item for item in queries if item] or [state.question])


def _record_query_attempts(state: ResearchQuestionState, queries: list[str], round_index: int) -> None:
    for query in queries:
        best = max(
            state.criteria,
            key=lambda item: (
                _query_contains_hint(query, item.query_hint or item.description),
                _text_overlap(query, item.query_hint or item.description),
                -len(item.attempts),
            ),
            default=None,
        )
        if best is None:
            continue
        best.attempts.append({
            "round_index": round_index,
            "query": query,
            "attempt_type": "targeted" if round_index > 1 else "direct",
        })


def _query_contains_hint(query: str, hint: str) -> int:
    normalized_query = re.sub(r"\s+", "", str(query or "")).lower()
    normalized_hint = re.sub(r"\s+", "", str(hint or "")).lower()
    return int(bool(normalized_hint) and normalized_hint in normalized_query)


def _update_criterion_ledger(state: ResearchQuestionState, *, semantic: Any) -> None:
    semantic = semantic if isinstance(semantic, dict) else {}
    semantic_rows = [
        item for item in semantic.get("criterion_coverage") or [] if isinstance(item, dict)
    ]
    status_by_id = {
        str(item.get("criterion_id") or ""): str(item.get("status") or "").strip().lower()
        for item in semantic_rows
        if str(item.get("criterion_id") or "")
        and str(item.get("status") or "").strip().lower() in {"covered", "partial", "uncovered"}
    }
    judged_evidence_ids_by_criterion = {
        str(item.get("criterion_id") or ""): [
            str(value) for value in item.get("supporting_evidence_ids") or [] if str(value)
        ]
        for item in semantic_rows
        if str(item.get("criterion_id") or "")
        and "supporting_evidence_ids" in item
    }
    selection = state.diagnostics.get("coverage_evidence_selection")
    selection = selection if isinstance(selection, dict) else {}
    evidence_ids_by_criterion = {
        str(item.get("criterion_id") or ""): [
            str(value) for value in item.get("evidence_ids") or [] if str(value)
        ]
        for item in selection.get("criterion_evidence_groups") or []
        if isinstance(item, dict) and str(item.get("criterion_id") or "")
    }
    accepted_ids = {
        str(item.get("id") or item.get("_coverage_evidence_id") or "")
        for item in state.accepted_evidence
        if str(item.get("id") or item.get("_coverage_evidence_id") or "")
    }
    for criterion in state.criteria:
        was_covered = criterion.status == "covered"
        if criterion.criterion_id in status_by_id:
            criterion.status = status_by_id[criterion.criterion_id]
        evidence_ids = (
            judged_evidence_ids_by_criterion[criterion.criterion_id]
            if criterion.criterion_id in judged_evidence_ids_by_criterion
            else evidence_ids_by_criterion.get(criterion.criterion_id, [])
        )
        criterion.evidence_ids = list(dict.fromkeys(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in accepted_ids
        ))
        criterion.coverage_delta = 1.0 if criterion.status == "covered" and not was_covered else 0.0


def _text_overlap(left: str, right: str) -> float:
    left_terms, right_terms = extract_terms(left), extract_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(2, min(len(left_terms), len(right_terms)))


def _gap_queries(state: ResearchQuestionState) -> list[str]:
    queries = [str(item.get("query") or "").strip() for item in state.gap_actions if str(item.get("query") or "").strip()]
    if queries:
        return _unique(queries)[:3]
    return [
        source_query(state.topic_anchor or state.question, source_type)
        for source_type in state.preferred_source_types[:2]
    ] or [state.question]


def _plan_gap_actions(state: ResearchQuestionState) -> list[dict[str, Any]]:
    # `existing_domains` is applied to the "source_floor" gap action only. `_evidence_domains`
    # dedupes by bare netloc (urlparse().netloc), not by repo/author path, so a multi-tenant host
    # (github.com, medium.com, ...) gets excluded wholesale the moment ANY accepted source happens
    # to live there. For "source_type"/"semantic_coverage" gaps -- whose goal is a SPECIFIC missing
    # type or fact, not domain diversity -- that exclusion is actively counterproductive: in a
    # dual/multi-compare topic each subject's own official repo commonly shares a netloc with an
    # already-fetched, unrelated repo, so the exclusion permanently starves the gap of the exact
    # source it needs. In `case-gap-domain-exclusion`, an architecture gap query excluded an
    # entire shared repository host after another subject's repository had already covered that
    # domain, so the target subject's official repository could never be found. Only
    # "source_floor" (diversify_domains) genuinely wants to
    # avoid domains already represented, since its entire purpose is domain diversity.
    actions: list[dict[str, Any]] = []
    existing_domains = sorted(_evidence_domains(state.accepted_evidence))
    for gap in state.coverage.get("gaps") or []:
        # Automatic follow-up is reserved for a blocking core gap. Source/evidence floors,
        # preferred source types and soft criteria remain visible diagnostics but do not
        # consume another round once the question's hard criteria are covered.
        if not gap.get("blocking"):
            continue
        gap_type = str(gap.get("gap_type") or "")
        if gap_type not in {"semantic_coverage", "evidence_floor", "source_floor"}:
            continue
        if gap_type == "source_type":
            requested_type = str(gap.get("source_type") or "web_article")
            fallback_used = False
            if state.access_issues and state.rounds_used >= 1 and gap.get("fallback_types"):
                requested_type = str((gap.get("fallback_types") or [requested_type])[0])
                fallback_used = True
            actions.append({
                "action": "retrieve_source_type",
                "gap_type": gap_type,
                "source_type": requested_type,
                "fallback_used": fallback_used,
                "blocking": True,
                "query": source_query(state.topic_anchor or state.question, requested_type),
            })
        elif gap_type == "source_floor":
            actions.append({
                "action": "diversify_domains",
                "gap_type": gap_type,
                "blocking": True,
                "query": source_query(
                    state.topic_anchor or state.question, "web_article", excluded_domains=existing_domains
                ),
            })
            # Implementation/definition questions use canonical document identities as the
            # floor. If the topic is a concrete GitHub project, the normal domain-diversification
            # query would exclude github.com and starve the exact README/docs/issues pages that
            # can satisfy the floor. Add a same-project fallback without exclusions; the source
            # identity metric, not domain diversity, is the hard requirement for these kinds.
            if (
                state.coverage.get("source_floor_mode") == "source_identity"
                and _github_project_evidence(state.accepted_evidence)
            ):
                actions.append({
                    "action": "retrieve_same_project_documents",
                    "gap_type": gap_type,
                    "blocking": True,
                    "fallback_for": "diversify_domains",
                    "query": compile_research_query(
                        topic_anchor=state.topic_anchor or state.question,
                        content="GitHub README docs issues pulls commits contributors releases",
                    ),
                })
        elif gap_type == "boundary_evidence":
            actions.append({
                "action": "find_boundary_evidence",
                "gap_type": gap_type,
                "blocking": True,
                "query": compile_research_query(
                    topic_anchor=state.topic_anchor or state.question,
                    content=boundary_query_terms(state.question),
                ),
            })
        elif gap_type == "semantic_coverage":
            missing_fact = str(gap.get("missing_fact") or gap.get("aspect") or "").strip()
            preferred_types = [str(item) for item in gap.get("preferred_source_types") or [] if str(item)]
            criterion_id = str(gap.get("criterion_id") or "")
            missing_subjects = _missing_comparison_subjects(state, criterion_id=criterion_id)
            query_subjects = missing_subjects or [state.topic_anchor or state.question]
            for subject in query_subjects:
                action = {
                    "action": "cover_success_criterion",
                    "gap_type": gap_type,
                    "criterion_id": gap.get("criterion_id"),
                    "gap_code": gap.get("gap_code"),
                    "missing_fact": missing_fact,
                    "blocking": True,
                    "query": targeted_query(
                        topic_anchor=subject,
                        missing_fact=missing_fact,
                        source_type=preferred_types[0] if preferred_types else "",
                    ),
                }
                if missing_subjects:
                    action["missing_subject"] = subject
                actions.append(action)
        elif gap_type == "evidence_floor":
            # A low evidence count does not tell us WHICH content is missing. Prefer concrete,
            # unattempted criteria over a generic "implementation details evidence" query. This
            # keeps soft/optional criteria as enrichment rather than hard gates, while still using
            # the remaining bounded rounds to make the answer fuller. Generic evidence search is
            # only a last resort after every criterion has received a real query attempt.
            unattempted = sorted(
                (
                    criterion for criterion in state.criteria
                    if criterion.status != "covered" and not criterion.attempts
                ),
                key=lambda criterion: (
                    {"hard": 0, "soft": 1, "optional": 2}.get(criterion.level, 3),
                    criterion.criterion_id,
                ),
            )
            for criterion in unattempted[:3]:
                actions.append({
                    "action": "cover_unattempted_criterion",
                    "gap_type": gap_type,
                    "criterion_id": criterion.criterion_id,
                    "missing_fact": criterion.description,
                    "blocking": True,
                    "query": targeted_query(
                        topic_anchor=state.topic_anchor or state.question,
                        missing_fact=criterion.query_hint or criterion.description,
                        source_type=(criterion.preferred_source_types or state.preferred_source_types or [""])[0],
                    ),
                })
            if not unattempted:
                actions.append({
                    "action": "find_novel_claim",
                    "gap_type": gap_type,
                    "blocking": True,
                    "query": compile_research_query(
                        topic_anchor=state.topic_anchor or state.question,
                        content="additional documented facts examples limitations",
                    ),
                })
        elif gap_type == "required_source_group":
            # Mirrors the "source_type" branch above -- see coverage.py's `_official_source_check`
            # docstring (case-source-signal-starvation-b) for why this gap stays non-blocking. That
            # incident was about the gate being too eager to conclude "missing" from a
            # classification artifact, not about whether it's worth actually searching once --
            # docs/modules/deep-research-plan-recon-agentic-upgrade-v0.1.md 线2 restores exactly
            # that one real search so "缺" is a conclusion backed by an actual attempt, not silence.
            # Only the first type in the group is queried (same simplification `source_type`'s
            # branch makes for its own single `requested_type`): the group is satisfied by ANY one
            # of its listed types, so one representative query is enough to test the gap for real.
            group_types = [str(item) for item in gap.get("source_types") or [] if str(item)]
            requested_type = group_types[0] if group_types else "official_docs"
            actions.append({
                "action": "retrieve_required_source_group",
                "gap_type": gap_type,
                "source_types": group_types,
                "blocking": False,
                "query": source_query(state.topic_anchor or state.question, requested_type),
            })
    actions.sort(key=lambda item: _gap_action_priority(state, item))
    output: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for action in actions:
        query = str(action.get("query") or "")
        if not query or query in seen_queries or query in state.attempted_queries:
            continue
        seen_queries.add(query)
        output.append(action)
    return output[:4]


def _missing_comparison_subjects(
    state: ResearchQuestionState,
    *,
    criterion_id: str,
) -> list[str]:
    """Return requested comparison subjects without criterion-bound evidence.

    A comparison query that keeps naming every subject can repeatedly rank the already-covered
    subject first. Use the existing structured subject contract to make follow-up retrieval target
    only the missing side. Evidence on another comparison axis does not count: admission clears
    incompatible criterion bindings, so the check stays scoped to the current criterion.
    """
    subjects = [str(item).strip() for item in state.comparison_subjects if str(item).strip()]
    if len(subjects) < 2:
        return []
    covered: set[str] = set()
    for evidence in state.accepted_evidence:
        if not isinstance(evidence, dict):
            continue
        evidence_criteria = {str(item) for item in evidence.get("criterion_ids") or [] if str(item)}
        if criterion_id and criterion_id not in evidence_criteria:
            continue
        evidence_subject = str(evidence.get("subject") or "").strip()
        if not evidence_subject:
            continue
        for subject in subjects:
            match = entity_match_metadata(
                title=evidence_subject,
                snippet=evidence_subject,
                topic=subject,
            )
            if float(match.get("entity_match_score") or 0.0) >= 0.75:
                covered.add(subject)
    return [subject for subject in subjects if subject not in covered]


def _gap_action_priority(state: ResearchQuestionState, action: dict[str, Any]) -> tuple[int, int, str]:
    gap_type = str(action.get("gap_type") or "")
    if gap_type == "boundary_evidence":
        return (0, 0, "")
    if gap_type == "semantic_coverage":
        criterion_id = str(action.get("criterion_id") or "")
        criterion = next((item for item in state.criteria if item.criterion_id == criterion_id), None)
        attempts = len(criterion.attempts) if criterion else 0
        return (1, attempts, criterion_id)
    if gap_type in {"evidence_floor", "source_floor", "grounding"}:
        return (2, 0, gap_type)
    return (3, 0, gap_type)


def _suggested_queries(retrieval: dict[str, Any]) -> list[str]:
    suggestions = retrieval.get("suggested_queries") or []
    if isinstance(retrieval.get("suggested_next_step"), str):
        suggestions = [*suggestions, retrieval["suggested_next_step"]]
    return [str(item).strip() for item in suggestions if str(item).strip()][:3]


def _namespace_round(
    retrieval: dict[str, Any],
    state: ResearchQuestionState,
    round_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefix = f"{state.question_id}_r{round_index}"
    chunks: list[dict[str, Any]] = []
    chunk_ids: dict[str, str] = {}
    for index, raw in enumerate(retrieval.get("selected_chunks") or [], start=1):
        if not isinstance(raw, dict):
            continue
        old_id = str(raw.get("id") or f"chk_{index:04d}")
        new_id = f"{prefix}_{old_id}"
        chunk_ids[old_id] = new_id
        chunks.append({
            **raw,
            "id": new_id,
            "question_id": state.question_id,
            "section_id": state.section_id,
            "question": state.question,
            "comparison_axis": state.comparison_axis,
        })
    evidence: list[dict[str, Any]] = []
    raw_evidence = retrieval.get("evidence_map") or retrieval.get("evidence_pack") or []
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            continue
        old_chunk_id = str(raw.get("chunk_id") or "")
        evidence.append({
            **raw,
            "id": "",
            "chunk_id": chunk_ids.get(old_chunk_id, old_chunk_id),
            "question_id": state.question_id,
            "section_id": state.section_id,
            "comparison_axis": raw.get("comparison_axis") or "",
            "source_type": raw.get("source_type"),
        })
    return chunks, evidence


def _global_evidence(states: list[ResearchQuestionState]) -> list[dict[str, Any]]:
    items = [(state, item) for state in states for item in state.accepted_evidence]
    items.sort(
        key=lambda pair: (
            float(pair[1].get("evidence_score") or 0),
            float(
                pair[1].get("authority_score")
                or authority_score(str(pair[1].get("source_url") or ""))
            ),
        ),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    evidence_id_map: dict[tuple[str, str], str] = {}
    for state, item in items:
        previous_id = str(item.get("id") or item.get("_coverage_evidence_id") or "")
        scoped_id = (state.question_id, previous_id)
        duplicate = next(
            (
                existing
                for existing in output
                if str(existing.get("source_url") or "") == str(item.get("source_url") or "")
                and claim_similarity(
                    str(existing.get("atomic_claim") or existing.get("claim") or ""),
                    str(item.get("atomic_claim") or item.get("claim") or ""),
                ) >= 0.82
            ),
            None,
        )
        if duplicate:
            item["id"] = duplicate["id"]
            if previous_id:
                evidence_id_map[scoped_id] = str(duplicate["id"])
            item.pop("_coverage_evidence_id", None)
            duplicate["question_ids"] = list(dict.fromkeys([
                *[str(value) for value in duplicate.get("question_ids") or [duplicate.get("question_id")] if value],
                *[str(value) for value in item.get("question_ids") or [item.get("question_id")] if value],
            ]))
            duplicate["criterion_ids"] = list(dict.fromkeys([
                *[str(value) for value in duplicate.get("criterion_ids") or [] if value],
                *[str(value) for value in item.get("criterion_ids") or [] if value],
            ]))[:2]
            continue
        item["id"] = f"ev_{len(output) + 1:03d}"
        if previous_id:
            evidence_id_map[scoped_id] = str(item["id"])
        item.pop("_coverage_evidence_id", None)
        item["used_by_documents"] = []
        output.append(dict(item))
    _remap_coverage_evidence_ids(states, evidence_id_map)
    return output


def _remap_coverage_evidence_ids(
    states: list[ResearchQuestionState],
    evidence_id_map: dict[tuple[str, str], str],
) -> None:
    if not evidence_id_map:
        return

    for state in states:
        def remap(values: Any) -> list[str]:
            return list(dict.fromkeys(
                evidence_id_map.get((state.question_id, str(value)), str(value))
                for value in values or []
                if str(value)
            ))

        for criterion in state.criteria:
            criterion.evidence_ids = remap(criterion.evidence_ids)
        for cell in state.comparison_cells:
            cell.evidence_ids = remap(cell.evidence_ids)
            for criterion in cell.criteria:
                criterion.evidence_ids = remap(criterion.evidence_ids)
        selection = state.diagnostics.get("coverage_evidence_selection")
        if not isinstance(selection, dict):
            continue
        selection["unassigned_evidence_ids"] = remap(selection.get("unassigned_evidence_ids"))
        for group in selection.get("criterion_evidence_groups") or []:
            if isinstance(group, dict):
                group["evidence_ids"] = remap(group.get("evidence_ids"))
        for attribution in selection.get("evidence_attribution") or []:
            if isinstance(attribution, dict):
                evidence_id = str(attribution.get("evidence_id") or "")
                attribution["evidence_id"] = evidence_id_map.get(
                    (state.question_id, evidence_id), evidence_id
                )


def _merge(existing: list[dict[str, Any]], incoming: list[Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return _dedupe([*existing, *[item for item in incoming if isinstance(item, dict)]], keys)


def _dedupe(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = "||".join(str(item.get(name) or "") for name in keys)
        if not key.strip("|"):
            key = repr(sorted(item.items()))[:500]
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _evidence_domains(items: list[dict[str, Any]]) -> set[str]:
    from urllib.parse import urlparse

    return {urlparse(str(item.get("source_url") or "")).netloc for item in items if urlparse(str(item.get("source_url") or "")).netloc}


def _evidence_source_identities(items: list[dict[str, Any]]) -> set[str]:
    return {
        identity
        for item in items
        for identity in [canonical_source_identity(item.get("source_url") or "")]
        if identity
    }


def _github_project_evidence(items: list[dict[str, Any]]) -> bool:
    """Whether accepted evidence already identifies a concrete GitHub repository."""
    projects = {
        canonical_project(item.get("source_url") or "")
        for item in items
        if "github.com/" in str(item.get("source_url") or "").lower()
    }
    return any(project.startswith("github.com/") and project.count("/") >= 2 for project in projects)


def _merge_access_issues(existing: list[dict[str, Any]], attempts: list[Any]) -> list[dict[str, Any]]:
    output = list(existing)
    seen = {(str(item.get("provider") or ""), str(item.get("status") or ""), str(item.get("reason") or "")) for item in output}
    resolved_capabilities = {
        str(item.get("capability") or "")
        for item in attempts
        if isinstance(item, dict)
        and item.get("selected")
        and str(item.get("status") or "") == "success"
        and not item.get("degraded")
    }
    for raw in attempts:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "")
        if str(raw.get("capability") or "") in resolved_capabilities:
            continue
        if status not in {"error", "timeout", "blocked", "skipped"} and not raw.get("degraded"):
            continue
        item = {
            "capability": raw.get("capability"),
            "provider": raw.get("provider"),
            "status": status or "degraded",
            "reason": str(raw.get("reason") or "")[:300],
            "url": str(raw.get("url") or "")[:500],
        }
        key = (str(item["provider"] or ""), str(item["status"] or ""), str(item["reason"] or ""))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output[-12:]
