"""Job-level candidate source ownership and criterion-aware pre-fetch ranking."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .evidence import authority_score
from .models import ResearchCriterionState, ResearchQuestionState
from .source_policy import entity_match_metadata, normalize_source_type
from .text_utils import extract_key_terms as _key_terms
from .text_utils import extract_terms as _terms
from .text_utils import to_simplified as _to_simplified


# Semantic rescue for the RETRIEVAL/source-pool layer -- mirrors the already-shipped
# backend/research/evidence.py:_semantic_rescue_mask pattern. When the pure bigram lexical
# classifier in `_recompute_relationship` would leave a genuinely-topical candidate as "generic"
# (real case-validator-context-parity sec_03_q1: a `diversify_domains` gap-fill query returned 10
# non-github Serper results -- neo4j.com, arxiv.org, hackernoon, medium... -- that ARE about
# LightRAG/GraphRAG, but ALL 10 were classified `generic` because the English snippets had zero
# bigram overlap with the Chinese question; none were fetched, `source_floor`'s 3-distinct-domain
# gate stayed blocked forever), a single batched embedding-cosine check per question rescues the
# topical ones to fetchable "adjacent" without admitting noise (verified with real
# BAAI/bge-large-zh-v1.5 embeddings: the 10 topical sources scored 0.52-0.64, genuinely
# irrelevant ones 0.15-0.32, against a 0.38 threshold).
_QUESTION_EMBEDDING_CACHE: dict[str, list[float]] = {}


def _semantic_rescue_enabled() -> bool:
    raw = os.getenv("RESEARCH_SOURCE_SEMANTIC_RELEVANCE_ENABLED", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _semantic_rescue_threshold() -> float:
    try:
        return float(os.getenv("RESEARCH_SOURCE_SEMANTIC_RELEVANCE_THRESHOLD", "0.38"))
    except (TypeError, ValueError):
        return 0.38


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right))
    norm_left = sum(x * x for x in left) ** 0.5
    norm_right = sum(y * y for y in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass(slots=True)
class SourceCandidate:
    url: str
    title: str = ""
    snippet: str = ""
    provider: str = ""
    source_type: str = "web_article"
    discovered_by_question_ids: set[str] = field(default_factory=set)
    matched_criterion_ids: set[str] = field(default_factory=set)
    relevance_by_criterion: dict[str, float] = field(default_factory=dict)
    legacy_relevance_by_criterion: dict[str, float] = field(default_factory=dict)
    relevance_by_question: dict[str, float] = field(default_factory=dict)
    relationship_by_question: dict[str, str] = field(default_factory=dict)
    # Question-ids whose "generic" lexical verdict was upgraded to "adjacent" by the batched
    # embedding-rescue pass (`apply_semantic_rescue`). Mirrors evidence.py's
    # `admission_tier="semantic_rescue"` audit tag: makes the rescue auditable in `to_dict()`
    # output and in the shadow-comparison diagnostics so an operator can see how many candidates
    # the lexical ranker was mis-classifying.
    semantic_rescued_by_question: set[str] = field(default_factory=set)
    entity_match_score: float = 1.0
    canonical_project: str = ""
    entity_mismatch_reason: str = ""
    entity_match_by_question: dict[str, float] = field(default_factory=dict)
    canonical_project_by_question: dict[str, str] = field(default_factory=dict)
    entity_mismatch_reason_by_question: dict[str, str] = field(default_factory=dict)
    # Question-ids for which a Context7 library candidate's post-fetch document body has
    # already been used to rescore it (see `_apply_context7_body_boost` in
    # research_service.py). Prevents re-scoring the same (candidate, question) pair on every
    # wave once its real content has already been factored in.
    body_scored_by_question: set[str] = field(default_factory=set)
    fetched: bool = False
    fetch_count: int = 0
    library_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "provider": self.provider,
            "source_type": self.source_type,
            "discovered_by_question_ids": sorted(self.discovered_by_question_ids),
            "matched_criterion_ids": sorted(self.matched_criterion_ids),
            "relevance_by_criterion": dict(self.relevance_by_criterion),
            "legacy_relevance_by_criterion": dict(self.legacy_relevance_by_criterion),
            "relevance_by_question": dict(self.relevance_by_question),
            "relationship_by_question": dict(self.relationship_by_question),
            "semantic_rescued_by_question": sorted(self.semantic_rescued_by_question),
            "entity_match_score": self.entity_match_score,
            "canonical_project": self.canonical_project,
            "entity_mismatch_reason": self.entity_mismatch_reason,
            "entity_match_by_question": dict(self.entity_match_by_question),
            "canonical_project_by_question": dict(self.canonical_project_by_question),
            "entity_mismatch_reason_by_question": dict(self.entity_mismatch_reason_by_question),
            "body_scored_by_question": sorted(self.body_scored_by_question),
            "fetched": self.fetched,
            "fetch_count": self.fetch_count,
            "library_id": self.library_id,
        }


class GlobalSourcePool:
    def __init__(self) -> None:
        self._candidates: dict[str, SourceCandidate] = {}
        self.discovered_count = 0
        self.deduplicated_count = 0
        self.selection_rounds = 0
        # Per-job pool instances are shared across concurrent per-question agent threads in the
        # agentic retrieval path (each question's `search_web` tool calls `add`/`select`
        # independently, unlike the question_loop wave path where these were only ever called
        # from one thread at a time). `add`'s "check candidate dict, insert-or-update" and
        # `select`'s "check `fetched`, then set it" are both classic check-then-write races
        # without this -- see docs/modules/deep-research-step2-retrieval-and-evidence-admission-
        # discussion-v0.2.md decision 5.
        self._lock = Lock()
        # Question-ids whose context7 fetch slot was claimed in a PRIOR wave of the same
        # job. `_select_context7_libraries_for_fetch` orders these LAST so a question
        # starved in round 1 (cap < active-question count) gets a slot in round 2
        # before any round-1 question claims a second. Lives on the pool because the
        # selector is a pure function without a job-scoped home; reset per-job by
        # constructing a fresh pool.
        self.context7_served_questions: set[str] = set()

    def add(
        self,
        sources: list[dict[str, Any]],
        state: ResearchQuestionState,
        *,
        match_states: list[ResearchQuestionState] | None = None,
    ) -> None:
        with self._lock:
            self._add_locked(sources, state, match_states=match_states)

    def _add_locked(
        self,
        sources: list[dict[str, Any]],
        state: ResearchQuestionState,
        *,
        match_states: list[ResearchQuestionState] | None = None,
    ) -> None:
        for raw in sources:
            url = normalize_url(str(raw.get("url") or raw.get("source_url") or ""))
            raw_provider = str(raw.get("provider") or "")
            raw_url = str(raw.get("url") or raw.get("source_url") or "")
            library_id = ""
            # Context7's /api/v2/libs/search returns items whose only URL-shaped field is a
            # library ID path like `/crewaiinc/crewai` -- not a fetchable URL. `normalize_url`
            # rejects non-http(s) values so these were silently dropped at this exact point,
            # meaning the official docs they point at never reached the candidate pool (see
            # project_sec03_official_sources_discovered_not_fetched_20260804). Synthesize a
            # `context7://library/<id>` key so the candidate enters the pool carrying its
            # `library_id`; a separate docs-channel fetch (`fetch_context7_context`) later fills
            # the material store without going through `url_fetch` (which cannot fetch a
            # library ID).
            if not url and raw_provider == "context7" and raw_url.startswith("/") and raw_url.count("/") >= 2:
                library_id = raw_url
                url = f"context7://library{raw_url}"
            if not url:
                continue
            self.discovered_count += 1
            candidate = self._candidates.get(url)
            if candidate is None:
                candidate = SourceCandidate(
                    url=url,
                    title=str(raw.get("title") or raw.get("source_title") or "")[:300],
                    snippet=str(raw.get("snippet") or raw.get("description") or "")[:1000],
                    provider=str(raw.get("provider") or ""),
                    source_type=normalize_source_type(
                        raw.get("source_type"), url=url, title=str(raw.get("title") or ""), text=str(raw.get("snippet") or ""),
                    ),
                )
                if library_id:
                    candidate.library_id = library_id
                    # Context7 library candidates stand in for official docs (the library ID
                    # resolves to docs via /api/v2/context, not a URL fetch). Force
                    # `official_docs` so the candidate clears `required_source_group` coverage
                    # and the docs-channel fetcher picks it up, regardless of lexical
                    # relationship to the (typically Chinese) criterion text.
                    candidate.source_type = "official_docs"
                self._candidates[url] = candidate
            else:
                self.deduplicated_count += 1
                if len(str(raw.get("snippet") or "")) > len(candidate.snippet):
                    candidate.snippet = str(raw.get("snippet") or "")[:1000]
            candidate.discovered_by_question_ids.add(state.question_id)
            for target in match_states or [state]:
                self._match(candidate, target)

    def select(
        self,
        states: list[ResearchQuestionState],
        *,
        max_urls: int,
        mode: str | None = None,
    ) -> list[SourceCandidate]:
        with self._lock:
            self.selection_rounds += 1
            # Exclude context7 library candidates (URL `context7://library/<id>`, carrying
            # `library_id`): they're not http-fetchable, so admitting one here wastes a
            # `max_urls` slot and `fetch_url_with_chain` would just record a `denied`
            # attempt. Their docs are fetched out-of-band by the context7 docs channel
            # (`_retrieve_question_wave`), not by `select` + `_fetch_source_candidates`.
            available = [
                item for item in self._candidates.values()
                if not item.fetched and not item.library_id
            ]
            if max_urls <= 0 or not available:
                return []
            selected = self._rank_candidates(available, states, max_urls=max_urls, mode=mode)
            for item in selected:
                item.fetched = True
                item.fetch_count += 1
            return selected

    def preview_select(
        self,
        states: list[ResearchQuestionState],
        *,
        max_urls: int,
        mode: str,
    ) -> list[SourceCandidate]:
        available = [
            item for item in self._candidates.values()
            if not item.fetched and not item.library_id
        ]
        return self._rank_candidates(available, states, max_urls=max_urls, mode=mode)

    def unfetched_candidates(self) -> list[SourceCandidate]:
        """Snapshot of not-yet-fetched candidates, for an external scorer (e.g. the LLM boost
        pass) to read titles/snippets from without reaching into the pool's private dict."""
        return [item for item in self._candidates.values() if not item.fetched and not item.library_id]

    def context7_library_candidates(self) -> list[SourceCandidate]:
        """Candidates that stand in for a Context7 library (URL is a synthetic
        `context7://library/<id>` and carry a `library_id`). Their docs are fetched via
        /api/v2/context, not `url_fetch`, so they live outside the http URL candidate space
        and must not enter `select`'s ranking competition (they'd dilute `max_urls` slots
        without being fetchable). Returns unfetched ones so a docs-channel pass can fetch
        them and mark them fetched out-of-band."""
        return [
            item for item in self._candidates.values()
            if item.library_id and not item.fetched
        ]

    def fetched_context7_candidates(self) -> list[SourceCandidate]:
        """Library candidates whose docs are already fetched (real body content sits in
        `JobMaterialStore`, keyed by `library_id`), for a post-fetch body-content rescoring
        pass to consume. `_match()` only ever scored these against the short library-level
        title+description available at discovery time (before `/api/v2/context` was called);
        this is the complement of `context7_library_candidates()` -- the fetched half, not
        the unfetched half."""
        return [
            item for item in self._candidates.values()
            if item.library_id and item.fetched
        ]

    def preview_llm_boost(
        self,
        scores: dict[str, dict[str, float]],
        states: list[ResearchQuestionState],
    ) -> dict[str, Any]:
        """Read-only diagnostics for "what would `apply_llm_boost` change" -- shadow mode calls
        this instead of `apply_llm_boost` so it can observe the effect without ever mutating the
        pool or influencing this round's real selection."""
        boosted_pair_count = 0
        newly_admitted_count = 0
        newly_admitted_urls: set[str] = set()
        for state in states:
            for criterion in state.criteria:
                key = f"{state.question_id}:{criterion.criterion_id}"
                for candidate in self._candidates.values():
                    llm_score = (scores.get(candidate.url) or {}).get(key)
                    if llm_score is None:
                        continue
                    current = candidate.relevance_by_criterion.get(key, 0.0)
                    boosted = max(current, min(1.0, float(llm_score)))
                    if boosted > current:
                        boosted_pair_count += 1
                        if current < 0.12 <= boosted:
                            newly_admitted_count += 1
                            newly_admitted_urls.add(candidate.url)
        return {
            "boosted_pair_count": boosted_pair_count,
            "newly_admitted_count": newly_admitted_count,
            "newly_admitted_urls": sorted(newly_admitted_urls)[:20],
        }

    def apply_llm_boost(
        self,
        scores: dict[str, dict[str, float]],
        states: list[ResearchQuestionState],
    ) -> int:
        """Merge externally computed (candidate_url -> {question_id:criterion_id -> score})
        LLM relevance scores into the pool, boost-only.

        Only ever raises `relevance_by_criterion` (via ``max``) -- a candidate the lexical
        scorer already admitted keeps its lexical score untouched; this can only pull a
        lexically-invisible-but-semantically-relevant candidate (e.g. an English "recursion
        limits" source against a Chinese "最大迭代次数" criterion -- see
        project_source_pool_extract_key_terms_relevance_fix_20260801 / the LLM-boost follow-up)
        up past the 0.12 admission bar it could never reach on lexical overlap alone. Never
        touches `legacy_relevance_by_criterion` -- the legacy ranker mode must stay a pure,
        unmodified baseline for the existing legacy/scoped shadow comparison
        (`_source_selection_shadow`) to remain meaningful. Returns the number of candidates
        whose relevance actually changed, for shadow-mode diagnostics.
        """
        boosted_count = 0
        for state in states:
            prefix = f"{state.question_id}:"
            for criterion in state.criteria:
                key = f"{state.question_id}:{criterion.criterion_id}"
                for candidate in self._candidates.values():
                    candidate_scores = scores.get(candidate.url) or {}
                    llm_score = candidate_scores.get(key)
                    if llm_score is None:
                        continue
                    current = candidate.relevance_by_criterion.get(key, 0.0)
                    boosted = max(current, min(1.0, float(llm_score)))
                    if boosted > current:
                        candidate.relevance_by_criterion[key] = boosted
                        if boosted >= 0.12:
                            candidate.matched_criterion_ids.add(key)
                        boosted_count += 1
            for candidate in self._candidates.values():
                self._recompute_relationship(candidate, state, prefix=prefix)
        return boosted_count

    def apply_semantic_rescue(self, states: list[ResearchQuestionState]) -> int:
        """Batched embedding rescue: promote lexically-"generic" candidates to fetchable
        "adjacent" when their title+snippet is semantically close to the question.

        Mirrors `backend/research/evidence.py:_semantic_rescue_mask`'s proven pattern:
        batched (one `_embed_texts` call per question, not per candidate), env-gated
        (`RESEARCH_SOURCE_SEMANTIC_RELEVANCE_ENABLED`, default on), fail-open on any embedding
        error (keeps the lexical-only verdict -- an embedding outage must never crash or block
        source selection; the provider is external/SiliconFlow and goes down), and tagged for
        audit (`semantic_rescued_by_question`, mirroring evidence.py's
        `admission_tier="semantic_rescue"`).

        Why batched, not inline in `_recompute_relationship`: `_recompute_relationship` is called
        per-candidate per-state from `_match` (as candidates stream in during `add`) and again
        from `apply_llm_boost` (post-boost). Inline rescue there would be N embedding calls per
        question per round (N = generic-candidate count, often 10+) and would be silently
        overwritten by the next `_match`/`apply_llm_boost` call. Batching here does one call per
        question and writes a tag the (already-floor-protected) `_recompute_relationship` honors
        on re-runs.

        Promotes to "adjacent" (NOT "direct"): rescue admits to fetchability, but "direct" wins
        `_rank_candidates`'s round-robin and would starve genuine lexical direct matches.
        "adjacent" is the right tier -- fetchable but lower-priority.
        """
        if not _semantic_rescue_enabled():
            return 0
        try:
            from workers import _embed_texts  # lazy: avoid workers.py's app-init import cycle
        except Exception:
            # workers.py import failure degrades to no-rescue rather than crashing source selection.
            return 0
        rescued_count = 0
        for state in states:
            question = str(state.question or "").strip()
            if not question:
                continue
            # Collect generic candidates for THIS question. A candidate already direct/adjacent
            # lexically needs no rescue; one already rescued for this question (re-run across
            # waves) needs no re-embed; context7 library candidates are fetched out-of-band.
            generic = [
                candidate for candidate in self._candidates.values()
                if candidate.relationship_by_question.get(state.question_id) == "generic"
                and candidate.entity_match_by_question.get(state.question_id, 1.0) >= 0.5
                and state.question_id not in candidate.semantic_rescued_by_question
                and not candidate.library_id
            ]
            if not generic:
                continue
            # Embed title + " " + snippet only -- NOT the url. url tokens like "neo4j.com"
            # pollute the embedding and would inflate similarity for any question about a domain
            # that happens to share a TLD. The question side is state.question alone; purpose is
            # frequently empty or generic and adding it dilutes the signal.
            texts = [f"{candidate.title} {candidate.snippet}" for candidate in generic]
            try:
                question_embedding = _QUESTION_EMBEDDING_CACHE.get(question)
                if question_embedding is None:
                    embeddings = _embed_texts([question, *texts])
                    question_embedding, text_embeddings = embeddings[0], embeddings[1:]
                    _QUESTION_EMBEDDING_CACHE[question] = question_embedding
                else:
                    text_embeddings = _embed_texts(texts)
            except Exception:
                # Fail-open per question: one question's batch failing must not poison the next
                # question's batch. Keep the lexical-only "generic" verdict for this batch.
                continue
            threshold = _semantic_rescue_threshold()
            for candidate, embedding in zip(generic, text_embeddings):
                if _cosine_similarity(question_embedding, embedding) >= threshold:
                    candidate.semantic_rescued_by_question.add(state.question_id)
                    candidate.relationship_by_question[state.question_id] = "adjacent"
                    rescued_count += 1
        return rescued_count

    def _recompute_relationship(
        self, candidate: SourceCandidate, state: ResearchQuestionState, *, prefix: str
    ) -> None:
        """Re-derive `relevance_by_question`/`relationship_by_question` for one candidate from
        whatever is currently in `relevance_by_criterion` -- the same logic `_match()` runs
        inline, factored out so `apply_llm_boost()` can re-run it after merging in scores that
        weren't known at discovery time."""
        source_text = f"{candidate.title} {candidate.snippet} {candidate.url}"
        entity_score = candidate.entity_match_by_question.get(state.question_id, 1.0)
        if entity_score < 0.5:
            candidate.relationship_by_question[state.question_id] = "generic"
            return
        candidate.relevance_by_question[state.question_id] = max(
            candidate.relevance_by_question.get(state.question_id, 0.0),
            max((score for key, score in candidate.relevance_by_criterion.items() if key.startswith(prefix)), default=0.0),
        )
        object_overlap = _object_relevance(source_text, f"{state.question} {state.purpose}")
        scoped_max = max(
            (score for key, score in candidate.relevance_by_criterion.items() if key.startswith(prefix)),
            default=0.0,
        )
        if object_overlap >= 0.12 and scoped_max >= 0.12:
            relationship = "direct"
        elif scoped_max >= 0.12 or max(
            (score for key, score in candidate.legacy_relevance_by_criterion.items() if key.startswith(prefix)),
            default=0.0,
        ) >= 0.2:
            relationship = "adjacent"
        else:
            relationship = "generic"
        # Preserve a prior semantic-rescue upgrade across re-runs. `_recompute_relationship` is
        # called from `_match` (per-candidate at discovery) AND from `apply_llm_boost` (per-
        # candidate after merging LLM scores). Without this floor, the second call would
        # overwrite a rescue-promoted "adjacent" back to "generic" if the boost didn't itself
        # clear the 0.12 bar -- silently un-rescuing a candidate the rescue pass had already
        # admitted. "adjacent" is the floor, never "direct": rescue admits to fetchability, it
        # does not jump the round-robin queue ahead of genuine lexical direct matches.
        if state.question_id in candidate.semantic_rescued_by_question and relationship == "generic":
            relationship = "adjacent"
        candidate.relationship_by_question[state.question_id] = relationship

    def _rank_candidates(
        self,
        available: list[SourceCandidate],
        states: list[ResearchQuestionState],
        *,
        max_urls: int,
        mode: str | None,
    ) -> list[SourceCandidate]:
        mode = str(mode or os.getenv("RESEARCH_SOURCE_RANKER_MODE", "scoped")).strip().lower()
        if mode not in {"legacy", "scoped"}:
            mode = "scoped"
        unresolved = [
            (state, criterion)
            for state in states
            for criterion in state.criteria
            if criterion.status != "covered"
        ]
        if mode == "legacy":
            return sorted(
                available,
                key=lambda item: self._legacy_score(item, unresolved),
                reverse=True,
            )[:max_urls]
        selected: list[SourceCandidate] = []
        served_subjects: set[tuple[str, str]] = set()
        max_subject_count = max((len(state.comparison_subjects) for state in states), default=0)
        for subject_index in range(max_subject_count):
            for state in states:
                if subject_index >= len(state.comparison_subjects):
                    continue
                state_criteria = [
                    (candidate_state, criterion)
                    for candidate_state, criterion in unresolved
                    if candidate_state.question_id == state.question_id
                ]
                focus = state_criteria[0] if state_criteria else None
                subject = state.comparison_subjects[subject_index]
                subject_key = _subject_key(subject)
                served_key = (state.question_id, subject_key)
                if not subject_key or served_key in served_subjects:
                    continue
                ranked_subject_candidates = sorted(
                    (
                        item for item in available
                        if item not in selected
                        and _candidate_matches_subject(item, subject)
                        and (
                            self._relationship(item, state) in {"direct", "adjacent"}
                            or state.question_id in item.discovered_by_question_ids
                        )
                    ),
                    key=lambda item: self._score(item, unresolved, focus=focus),
                    reverse=True,
                )
                if not ranked_subject_candidates:
                    continue
                selected.append(ranked_subject_candidates[0])
                served_subjects.add(served_key)
                if len(selected) >= max_urls:
                    return selected
        # A hard slot prefers direct evidence and only falls back to adjacent material.
        # Interleave round-robin across questions instead of draining one question's hard
        # criteria before moving to the next: a flat scan let questions earlier in `states`
        # (e.g. a section's first question) claim every slot in a capped round before a later
        # question's turn ever arrived, even though its own candidates were fully matched and
        # waiting in the pool (see coverage.json attempted_fetch_url_count staying 0 for a
        # question with many search hits already discovered).
        for state, criterion in _round_robin_hard_criteria(unresolved):
            remaining = [item for item in available if item not in selected]
            direct = [item for item in remaining if self._relationship(item, state) == "direct"]
            adjacent = [item for item in remaining if self._relationship(item, state) == "adjacent"]
            ranked = sorted(
                direct or adjacent,
                key=lambda item: self._score(item, unresolved, focus=(state, criterion)),
                reverse=True,
            )
            if ranked and self._criterion_relevance(ranked[0], state, criterion) >= 0.12:
                selected.append(ranked[0])
            if len(selected) >= max_urls:
                break
        if len(selected) < max_urls:
            for relation in ("direct", "adjacent"):
                ranked = sorted(
                    (
                        item for item in available
                        if item not in selected
                        and any(self._relationship(item, state) == relation for state, _criterion in unresolved)
                    ),
                    key=lambda item: self._score(item, unresolved),
                    reverse=True,
                )
                for item in ranked:
                    if self._score(item, unresolved) < 0.18:
                        continue
                    selected.append(item)
                    if len(selected) >= max_urls:
                        return selected
        return selected

    def diagnostics(self) -> dict[str, Any]:
        candidates = list(self._candidates.values())
        return {
            "candidate_count": len(candidates),
            "discovered_count": self.discovered_count,
            "deduplicated_count": self.deduplicated_count,
            "fetched_count": sum(item.fetched for item in candidates),
            "fetch_count": sum(item.fetch_count for item in candidates),
            "multi_question_candidate_count": sum(len(_matched_question_ids(item)) > 1 for item in candidates),
            "multi_criterion_candidate_count": sum(len(item.matched_criterion_ids) > 1 for item in candidates),
            "selection_rounds": self.selection_rounds,
        }

    def relevant_candidates(
        self,
        state: ResearchQuestionState,
        *,
        exclude_urls: list[str] | None = None,
        limit: int = 6,
        mode: str | None = None,
    ) -> list[SourceCandidate]:
        excluded = {normalize_url(item) for item in (exclude_urls or [])}
        prefix = f"{state.question_id}:"
        mode = str(mode or os.getenv("RESEARCH_SOURCE_RANKER_MODE", "scoped")).strip().lower()
        ranked = sorted(
            (
                item for item in self._candidates.values()
                if item.fetched
                and normalize_url(item.url) not in excluded
                and (
                    (
                        max((score for key, score in item.legacy_relevance_by_criterion.items() if key.startswith(prefix)), default=0.0) >= 0.08
                        if mode == "legacy"
                        else (
                            self._relationship(item, state) in {"direct", "adjacent"}
                            or any(_candidate_matches_subject(item, subject) for subject in state.comparison_subjects)
                            or (
                                bool(item.library_id)
                                and state.question_id in item.discovered_by_question_ids
                                and item.entity_match_by_question.get(state.question_id, 1.0) >= 0.5
                            )
                        )
                    )
                )
            ),
            key=lambda item: max(
                (
                    score for key, score in (
                        item.legacy_relevance_by_criterion if mode == "legacy" else item.relevance_by_criterion
                    ).items() if key.startswith(prefix)
                ),
                default=0.0,
            ),
            reverse=True,
        )
        return _balance_candidates_by_subject(ranked, state.comparison_subjects, limit=max(1, limit))

    def _match(self, candidate: SourceCandidate, state: ResearchQuestionState) -> None:
        source_text = f"{candidate.title} {candidate.snippet} {candidate.url}"
        entity = entity_match_metadata(
            url=candidate.url,
            title=candidate.title,
            snippet=candidate.snippet,
            topic=state.question,
        )
        score = float(entity.get("entity_match_score") or 0.0)
        candidate.entity_match_by_question[state.question_id] = score
        candidate.canonical_project_by_question[state.question_id] = str(entity.get("canonical_project") or "")
        candidate.entity_mismatch_reason_by_question[state.question_id] = str(entity.get("entity_mismatch_reason") or "")
        if not candidate.canonical_project or score > candidate.entity_match_score:
            candidate.entity_match_score = score
            candidate.canonical_project = str(entity.get("canonical_project") or "")
            candidate.entity_mismatch_reason = str(entity.get("entity_mismatch_reason") or "")
        domain_relevance = _domain_relevance(source_text, f"{state.question} {state.purpose}")
        candidate.relevance_by_question[state.question_id] = max(
            candidate.relevance_by_question.get(state.question_id, 0.0),
            domain_relevance,
        )
        for criterion in state.criteria:
            key = f"{state.question_id}:{criterion.criterion_id}"
            legacy_relevance = _lexical_relevance(source_text, f"{state.question} {criterion.description}")
            if candidate.source_type in criterion.preferred_source_types:
                legacy_relevance = min(1.0, legacy_relevance + 0.18)
            candidate.legacy_relevance_by_criterion[key] = max(
                candidate.legacy_relevance_by_criterion.get(key, 0.0),
                legacy_relevance,
            )
            criterion_relevance = _key_term_relevance(source_text, criterion.description)
            if domain_relevance >= 0.08:
                relevance = domain_relevance * 0.6 + criterion_relevance * 0.4
            elif criterion_relevance >= 0.3:
                relevance = criterion_relevance * 0.7
            else:
                relevance = 0.0
            candidate.relevance_by_criterion[key] = max(candidate.relevance_by_criterion.get(key, 0.0), relevance)
            if relevance >= 0.12:
                candidate.matched_criterion_ids.add(key)
        self._recompute_relationship(candidate, state, prefix=f"{state.question_id}:")

    def _criterion_relevance(
        self,
        candidate: SourceCandidate,
        state: ResearchQuestionState,
        criterion: ResearchCriterionState,
    ) -> float:
        return candidate.relevance_by_criterion.get(f"{state.question_id}:{criterion.criterion_id}", 0.0)

    def _score(
        self,
        candidate: SourceCandidate,
        criteria: list[tuple[ResearchQuestionState, ResearchCriterionState]],
        *,
        focus: tuple[ResearchQuestionState, ResearchCriterionState] | None = None,
    ) -> float:
        active_keys = {f"{state.question_id}:{criterion.criterion_id}" for state, criterion in criteria}
        relevance = (
            self._criterion_relevance(candidate, focus[0], focus[1])
            if focus
            else max((score for key, score in candidate.relevance_by_criterion.items() if key in active_keys), default=0.0)
        )
        utility = min(len(candidate.matched_criterion_ids & active_keys), 4) * 0.08
        source_fit = 0.12 if any(candidate.source_type in item.preferred_source_types for _state, item in criteria) else 0.0
        authority = authority_score(candidate.url) * 0.25
        return relevance * 0.55 + authority + utility + source_fit

    def _legacy_score(
        self,
        candidate: SourceCandidate,
        criteria: list[tuple[ResearchQuestionState, ResearchCriterionState]],
    ) -> float:
        active_keys = {f"{state.question_id}:{criterion.criterion_id}" for state, criterion in criteria}
        relevance = max(
            (score for key, score in candidate.legacy_relevance_by_criterion.items() if key in active_keys),
            default=0.0,
        )
        return relevance * 0.7 + authority_score(candidate.url) * 0.3

    @staticmethod
    def _relationship(candidate: SourceCandidate, state: ResearchQuestionState) -> str:
        return candidate.relationship_by_question.get(state.question_id, "generic")


def normalize_url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return ""
    parsed = urlparse(value.strip())
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    path = path.rstrip("/") or "/"
    query = urlencode([
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ])
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def _lexical_relevance(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = left_terms & right_terms
    return min(1.0, len(overlap) / max(2, min(len(left_terms), len(right_terms))))


def _key_term_relevance(left: str, right: str) -> float:
    """TF-IDF-ranked term overlap, for questions dense with parenthetical examples.

    `_lexical_relevance`'s character-bigram splitting truncates multi-character words at
    arbitrary boundaries (e.g. "显式状态检查" fragments into "如显"/"式状"/"态检"/"查超"), so a long
    Chinese question packed with bracketed examples ("如状态不一致、循环爆炸..." "如显式状态检查、超时机制...")
    never produces bigrams that overlap a source's real vocabulary -- see
    project_source_pool_extract_key_terms_relevance_fix_20260801: a production job's `sec_03_q3`
    (陷阱/循环爆炸/幂等性 question) scored every one of 14 discovered candidates below the 0.12
    fetch-admission bar and never fetched a single URL, while `extract_key_terms`'s phrase-ranked
    tokens ("陷阱", "循环", "幂等性") let the same candidates clear it. Falls back to
    `_lexical_relevance` when either side yields no key terms (very short/generic text), same
    guard `extract_key_terms` itself uses relative to `extract_terms`.
    """
    left_terms = set(_key_terms(left, top_k=32))
    right_terms = set(_key_terms(right, top_k=32))
    if not left_terms or not right_terms:
        return _lexical_relevance(left, right)
    return min(1.0, len(left_terms & right_terms) / max(2, min(len(left_terms), len(right_terms))))


def _domain_relevance(left: str, right: str) -> float:
    generic = {
        "agent", "agents", "multi", "system", "systems", "workflow", "workflows", "production",
        "architecture", "implementation", "example", "examples", "智能体", "多智能体", "系统", "工作流",
        "架构", "实现", "生产", "案例", "如何", "哪些", "什么",
    }
    left_terms = set(_key_terms(left, top_k=32)) - generic
    right_terms = set(_key_terms(right, top_k=32)) - generic
    if len(right_terms) < 3:
        return _lexical_relevance(left, right)
    return min(1.0, len(left_terms & right_terms) / max(2, min(len(left_terms), len(right_terms))))


def _object_relevance(left: str, right: str) -> float:
    generic = {
        "what", "which", "how", "common", "used", "using", "include", "provide", "describe", "identify",
        "agent", "agents", "multi", "system", "systems", "production", "architecture", "workflow", "workflows",
        "implementation", "role", "roles", "example", "examples", "official", "documentation", "case", "study",
        "如何", "哪些", "什么", "常见", "使用", "包括", "提供", "描述", "识别", "智能体", "多智能体", "系统",
        "生产", "架构", "工作流", "实现", "角色", "案例", "文档", "研究", "分析",
    }
    left_terms = set(_key_terms(left, top_k=32)) - generic
    right_terms = set(_key_terms(right, top_k=32)) - generic
    if not left_terms or not right_terms:
        return 0.0
    return min(1.0, len(left_terms & right_terms) / max(2, min(len(left_terms), len(right_terms))))


def _subject_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", _to_simplified(str(value or "").lower()))


def _candidate_matches_subject(candidate: SourceCandidate, subject: str) -> bool:
    if not _subject_key(subject):
        return False
    match = entity_match_metadata(
        url=candidate.url,
        title=candidate.title,
        snippet=candidate.snippet,
        topic=subject,
    )
    return float(match.get("entity_match_score") or 0.0) >= 0.75


def _balance_candidates_by_subject(
    ranked: list[SourceCandidate],
    subjects: list[str],
    *,
    limit: int,
) -> list[SourceCandidate]:
    if limit <= 0 or len(subjects) < 2:
        return ranked[: max(0, limit)]
    selected: list[SourceCandidate] = []
    for subject in subjects:
        candidate = next(
            (
                item for item in ranked
                if item not in selected and _candidate_matches_subject(item, subject)
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
        if len(selected) >= limit:
            return selected
    for candidate in ranked:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _round_robin_hard_criteria(
    unresolved: list[tuple[ResearchQuestionState, ResearchCriterionState]],
) -> list[tuple[ResearchQuestionState, ResearchCriterionState]]:
    """Order hard-criterion slots question-by-question first, then criterion-by-criterion.

    Preserves each question's own criterion order but ensures every question's first hard
    criterion is considered before any question's second, so a capped `max_urls` spends its
    slots across questions instead of exhausting them on whichever question sorts first.
    """
    by_question: dict[str, list[tuple[ResearchQuestionState, ResearchCriterionState]]] = {}
    for state, criterion in unresolved:
        if criterion.level != "hard":
            continue
        by_question.setdefault(state.question_id, []).append((state, criterion))
    return [
        item
        for row in zip_longest(*by_question.values())
        for item in row
        if item is not None
    ]


def _matched_question_ids(candidate: SourceCandidate) -> set[str]:
    return {key.split(":", 1)[0] for key in candidate.matched_criterion_ids if ":" in key}
