"""Internal data models for the question-centric research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .source_policy import (
    build_source_requirements,
    coverage_requirements,
    criterion_requires_source_diversity,
    normalize_question_kind,
)
from .text_utils import is_non_research_section_title


CRITERION_LEVEL_POLICY_V1 = "hard_soft_optional_v1"
LEGACY_CRITERION_LEVEL_POLICY = "legacy_unversioned"


@dataclass(slots=True)
class ResearchCriterionState:
    criterion_id: str
    description: str
    # Short, search-ready terms for this criterion (round-1 query content), planner-authored via
    # `criteria_queries` (aligned by index to success_criteria). Empty when the planner omits it
    # (older/authored plans) -- query compilation then falls back to `description` itself.
    # `description` stays untouched for its other consumers (coverage-ledger text-overlap,
    # source-relevance scoring, ranker/judge prompt payloads) -- this field exists only for the
    # search-query call sites.
    query_hint: str = ""
    level: str = "hard"
    # `explicit` means callers deliberately supplied the level. Plans normalized by the current
    # planner carry a versioned source; old checkpoints without level metadata are marked
    # `legacy_inferred` so coverage can apply a narrow compatibility rule without weakening an
    # explicitly hard criterion.
    level_source: str = "explicit"
    status: str = "uncovered"
    preferred_source_types: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    coverage_delta: float = 0.0


@dataclass(slots=True)
class ComparisonBudget:
    max_subjects: int
    max_axes: int
    max_required_cells: int
    initial_queries_per_cell: int
    max_gap_queries_per_cell: int
    comparative_queries_per_axis: int
    max_queries_per_wave: int
    max_fetch_urls_per_wave: int
    max_rounds: int

    @classmethod
    def for_depth(cls, depth: str) -> "ComparisonBudget":
        if depth == "deep":
            return cls(5, 5, 25, 1, 2, 2, 12, 10, 4)
        return cls(3, 4, 12, 1, 1, 1, 8, 10, 3)


@dataclass(slots=True)
class ComparisonCellState:
    cell_id: str
    subject: str
    axis: str
    criteria: list[ResearchCriterionState] = field(default_factory=list)
    initial_query: str = ""
    status: str = "uncovered"
    evidence_ids: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    unavailable_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QuestionBudget:
    max_rounds: int
    max_sources: int
    timeout_seconds: float
    max_model_calls: int
    max_materials: int = 0
    initial_fetch_limit: int = 0
    followup_fetch_limit: int = 0
    max_fetch_urls: int = 0

    def __post_init__(self) -> None:
        # max_sources remains as a compatibility alias for older callers.
        self.max_materials = self.max_materials or self.max_sources
        self.initial_fetch_limit = self.initial_fetch_limit or max(1, min(4, self.max_materials))
        self.followup_fetch_limit = self.followup_fetch_limit or max(1, min(2, self.max_materials))
        self.max_fetch_urls = self.max_fetch_urls or max(self.max_materials, self.max_sources)

    @classmethod
    def for_depth(cls, depth: str) -> "QuestionBudget":
        if depth == "deep":
            return cls(
                max_rounds=4,
                max_sources=14,
                max_materials=14,
                initial_fetch_limit=6,
                followup_fetch_limit=3,
                max_fetch_urls=20,
                timeout_seconds=300.0,
                max_model_calls=12,
            )
        return cls(
            max_rounds=3,
            max_sources=8,
            max_materials=8,
            initial_fetch_limit=4,
            followup_fetch_limit=2,
            max_fetch_urls=12,
            timeout_seconds=180.0,
            max_model_calls=8,
        )


@dataclass(slots=True)
class ResearchQuestionState:
    question_id: str
    section_id: str
    section_title: str
    question: str
    purpose: str
    success_criteria: list[str]
    preferred_source_types: list[str]
    budget: QuestionBudget
    criteria: list[ResearchCriterionState] = field(default_factory=list)
    question_kind: str = "definition"
    coverage_requirements: dict[str, Any] = field(default_factory=dict)
    source_requirements: list[dict[str, Any]] = field(default_factory=list)
    # Job-level stable topic anchor (e.g. "Harness Engineering", "LightRAG GraphRAG"), extracted
    # ONCE per job via _extract_topic_anchor(user_request) and stamped onto plan["topic_anchor"]
    # -- every question state in the job shares the same value, rather than each question/round
    # re-deriving its own phrasing (which risked drift across the 4-6+ separate model calls a job
    # makes). compile_research_query() prepends this to every query; empty when extraction fails
    # or the plan predates this field, in which case callers fall back to `question` as the anchor.
    topic_anchor: str = ""
    # Explicit subjects for comparison jobs. This is the admissible subject set, not a search
    # keyword list: evidence about an unrelated comparator must not satisfy the question merely
    # because the extractor labelled it `subject_relation=comparison`.
    comparison_subjects: list[str] = field(default_factory=list)
    # Planner-authored comparison dimension for this question. Downstream stages use the exact
    # structured value to keep, for example, a general usefulness score from satisfying a
    # citation-quality criterion. Empty for non-comparison or genuinely cross-axis questions.
    comparison_axis: str = ""
    comparison_cells: list[ComparisonCellState] = field(default_factory=list)
    comparison_budget: dict[str, Any] = field(default_factory=dict)
    comparative_queries: list[str] = field(default_factory=list)
    comparative_query_attempts: list[dict[str, Any]] = field(default_factory=list)
    min_evidence: int = 4
    min_sources: int = 3
    criterion_level_policy: str = "explicit"
    # Whether this planner question is part of the user's completion contract or useful
    # enrichment. Soft questions are still researched and written, but their remaining gaps do
    # not make the whole report incomplete.
    answer_contract_level: str = "hard"
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    stop_reason: str = ""
    rounds_used: int = 0
    model_calls: int = 0
    elapsed_seconds: float = 0.0
    attempted_queries: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    materials: list[dict[str, Any]] = field(default_factory=list)
    selected_chunks: list[dict[str, Any]] = field(default_factory=list)
    accepted_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_evidence: list[dict[str, Any]] = field(default_factory=list)
    covered_aspects: list[str] = field(default_factory=list)
    weak_aspects: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    gap_actions: list[dict[str, Any]] = field(default_factory=list)
    access_issues: list[dict[str, Any]] = field(default_factory=list)
    attempted_fetch_urls: list[str] = field(default_factory=list)
    round_budgets: list[dict[str, Any]] = field(default_factory=list)
    gap_execution_log: list[dict[str, Any]] = field(default_factory=list)
    generated_gap_action_count: int = 0
    stagnant_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def question_states_from_plan(plan: dict[str, Any], *, depth: str) -> list[ResearchQuestionState]:
    """Build executable question states while deriving overview from later findings."""
    candidates: list[ResearchQuestionState] = []
    budget = QuestionBudget.for_depth(depth)
    topic_anchor = str(plan.get("topic_anchor") or "").strip()[:40]
    comparison_subjects = _strings(plan.get("subjects"))
    comparison_axes = _strings(plan.get("comparison_axes"))
    criterion_level_policy = str(
        plan.get("criterion_level_policy") or LEGACY_CRITERION_LEVEL_POLICY
    ).strip()
    for section in plan.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or f"sec_{len(candidates):02d}")
        section_title = str(section.get("title") or section_id)
        if section_id == "sec_00" or _is_non_research_section(section_title):
            continue
        raw_questions = section.get("questions") or [section_title]
        question_count = max(1, sum(bool(_question_text(item)) for item in raw_questions))
        for index, raw in enumerate(raw_questions):
            if isinstance(raw, dict):
                text = str(raw.get("text") or raw.get("question") or raw.get("title") or "").strip()
                question_id = str(raw.get("id") or raw.get("question_id") or f"{section_id}_q{index + 1}")
                purpose = str(raw.get("purpose") or section.get("intent") or section_title)
                criteria = _strings(raw.get("success_criteria"))
                preferred = _strings(raw.get("preferred_source_types"))
                dependencies = _strings(raw.get("dependencies"))
                criteria_queries = _strings(raw.get("criteria_queries"))
                criteria_are_contract_dimensions = str(
                    raw.get("criteria_contract") or ""
                ).strip().lower() in {
                    "question_dimensions",
                    "explicit_scope_obligations",
                }
                answer_contract_level = (
                    "soft" if str(raw.get("answer_contract_level") or "").strip().lower() == "soft" else "hard"
                )
            else:
                text = str(raw or "").strip()
                question_id = f"{section_id}_q{index + 1}"
                purpose = str(section.get("intent") or section_title)
                criteria, preferred, dependencies = [], [], []
                criteria_queries = []
                criteria_are_contract_dimensions = False
                answer_contract_level = "hard"
            if not text:
                continue
            comparison_axis = _question_comparison_axis(
                raw,
                canonical_axes=comparison_axes,
            )
            evidence_budget = section.get("evidence_budget") if isinstance(section.get("evidence_budget"), dict) else {}
            criteria_value = (
                criteria
                if criterion_level_policy == CRITERION_LEVEL_POLICY_V1
                else (criteria or _strings(section.get("success_criteria")) or [f"回答：{text}"])
            )
            raw_question_kind = raw.get("question_kind") if isinstance(raw, dict) else None
            question_kind = normalize_question_kind(
                raw_question_kind or section.get("question_kind"),
                question=text,
                purpose=purpose,
                success_criteria=criteria_value,
            )
            requirements = coverage_requirements(question_kind, question=text)
            planned_evidence = _positive_int(evidence_budget.get("min_evidence"))
            planned_sources = _positive_int(evidence_budget.get("min_sources"))
            per_question_evidence = _ceil_div(planned_evidence, question_count) if planned_evidence else 0
            per_question_sources = _ceil_div(planned_sources, question_count) if planned_sources else 0
            source_requirements = build_source_requirements(
                question=text,
                purpose=purpose,
                success_criteria=criteria_value,
                preferred_source_types=preferred or _strings(section.get("preferred_source_types")) or [],
                question_kind=question_kind,
            )
            preferred_types = [item["source_type"] for item in source_requirements]
            candidates.append(
                ResearchQuestionState(
                    question_id=question_id,
                    section_id=section_id,
                    section_title=section_title,
                    question=text,
                    purpose=purpose,
                    success_criteria=criteria_value,
                    criteria=_criterion_states(
                        criteria_value,
                        preferred_types,
                        question=text,
                        query_hints=criteria_queries,
                        level_policy=criterion_level_policy,
                        answer_contract_level=answer_contract_level,
                        criteria_are_contract_dimensions=criteria_are_contract_dimensions,
                    ),
                    preferred_source_types=preferred_types,
                    source_requirements=source_requirements,
                    dependencies=dependencies,
                    topic_anchor=topic_anchor,
                    comparison_subjects=list(comparison_subjects),
                    comparison_axis=comparison_axis,
                    budget=QuestionBudget(**asdict(budget)),
                    question_kind=question_kind,
                    coverage_requirements=requirements,
                    min_evidence=max(int(requirements["min_claim_clusters"]), per_question_evidence),
                    min_sources=max(int(requirements["min_domains"]), per_question_sources),
                    criterion_level_policy=criterion_level_policy,
                    answer_contract_level=answer_contract_level,
                )
            )
    # Preserve section breadth instead of allowing the first sections to consume the cap.
    by_section: dict[str, list[ResearchQuestionState]] = {}
    for state in candidates:
        by_section.setdefault(state.section_id, []).append(state)
    states: list[ResearchQuestionState] = []
    # A comparison plan can legitimately expand one explicitly requested matrix into one question
    # per object. Keep room for that breadth even though the planner's normal target is 4-8, so a
    # valid final row is not silently discarded after the user confirms the plan.
    research_type = str(plan.get("research_type") or "").strip().lower()
    limit = 10 if depth == "deep" or research_type == "comparison" else 6
    reached_limit = False
    per_section_limit = 5 if research_type == "comparison" else 4
    for offset in range(per_section_limit):
        for section_states in by_section.values():
            if offset < len(section_states):
                states.append(section_states[offset])
            if len(states) >= limit:
                reached_limit = True
                break
        if reached_limit:
            break
    _attach_comparison_cells(states, plan)
    return states


def _attach_comparison_cells(states: list[ResearchQuestionState], plan: dict[str, Any]) -> None:
    matrix_cells = [item for item in plan.get("matrix_cells") or [] if isinstance(item, dict)]
    if not states or not matrix_cells:
        return
    comparison_budget = plan.get("comparison_budget") if isinstance(plan.get("comparison_budget"), dict) else {}
    exact_by_axis: dict[str, list[ResearchQuestionState]] = {}
    open_states: list[ResearchQuestionState] = []
    for state in states:
        if state.comparison_axis:
            exact_by_axis.setdefault(_normalize_axis_label(state.comparison_axis), []).append(state)
        elif state.comparison_subjects:
            open_states.append(state)
        state.comparison_budget = dict(comparison_budget)
        if comparison_budget:
            state.budget.max_rounds = max(1, int(comparison_budget.get("max_rounds") or state.budget.max_rounds))

    owner_by_axis: dict[str, ResearchQuestionState] = {}
    claimed_open_owner_ids: set[str] = set()
    for raw in matrix_cells:
        axis = str(raw.get("axis") or raw.get("comparison_axis") or "").strip()
        axis_key = _normalize_axis_label(axis)
        if not axis_key or axis_key in owner_by_axis:
            continue
        owners = exact_by_axis.get(axis_key) or []
        if not owners:
            owners = [
                state
                for state in open_states
                if axis_key in _normalize_axis_label(
                    " ".join((state.question, state.purpose, state.section_title))
                )
            ]
        if len(owners) != 1:
            raise ValueError(
                f"comparison axis {axis!r} requires exactly one Axis Question; found {len(owners)}"
            )
        owner = owners[0]
        if not owner.comparison_axis:
            if owner.question_id in claimed_open_owner_ids:
                raise ValueError(
                    f"comparison Axis Question {owner.question_id!r} cannot own multiple required axes"
                )
            claimed_open_owner_ids.add(owner.question_id)
            owner.comparison_axis = axis
        owner_by_axis[axis_key] = owner

    for raw in matrix_cells:
        subject = str(raw.get("subject") or "").strip()
        axis = str(raw.get("axis") or raw.get("comparison_axis") or "").strip()
        if not subject or not axis:
            continue
        owner = owner_by_axis[_normalize_axis_label(axis)]
        hard_criteria = [item for item in owner.criteria if item.level == "hard"]
        if not hard_criteria:
            raise ValueError(
                f"comparison axis {axis!r} Axis Question {owner.question_id!r} has no hard criterion"
            )
        cell_criteria = [
            ResearchCriterionState(**asdict(item))
            for item in hard_criteria
        ]
        content = " ".join(
            [axis, *[(item.query_hint or item.description) for item in cell_criteria]]
        ).strip()
        owner.comparison_cells.append(ComparisonCellState(
            cell_id=str(raw.get("id") or f"{owner.question_id}_cell_{len(owner.comparison_cells) + 1:03d}"),
            subject=subject,
            axis=axis,
            criteria=cell_criteria,
            initial_query=_compact_query(subject, content),
        ))

    comparative_count = max(0, int(comparison_budget.get("comparative_queries_per_axis") or 0))
    for state in states:
        if not state.comparison_cells or comparative_count <= 0:
            continue
        subjects = " ".join(dict.fromkeys(cell.subject for cell in state.comparison_cells))
        axes = " ".join(dict.fromkeys(cell.axis for cell in state.comparison_cells))
        state.comparative_queries = [
            _compact_query(subjects, f"{axes} comparison controversy limitations third party evaluation {index + 1}")
            for index in range(comparative_count)
        ]


def _compact_query(anchor: str, content: str) -> str:
    return re.sub(r"\s+", " ", f"{anchor} {content}").strip()[:140]


def _is_non_research_section(title: str) -> bool:
    return is_non_research_section_title(title)


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _question_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("text") or raw.get("question") or raw.get("title") or "").strip()
    return str(raw or "").strip()


def _question_comparison_axis(
    raw: Any,
    *,
    canonical_axes: list[str] | None = None,
) -> str:
    """Use the structured axis contract and align it with the plan's declared axis set."""
    if not isinstance(raw, dict):
        return ""
    mode = str(raw.get("comparison_axis_mode") or "").strip().lower()
    if mode in {"open", "not_applicable"}:
        return ""
    axis = str(raw.get("comparison_axis") or raw.get("axis") or "").strip()
    normalized_axis = _normalize_axis_label(axis)
    matches = [
        candidate
        for candidate in canonical_axes or []
        if _normalize_axis_label(candidate)
        and _normalize_axis_label(candidate) in normalized_axis
    ]
    if len(matches) == 1:
        return matches[0]
    return axis


def _normalize_axis_label(value: Any) -> str:
    return "".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(value or "").lower()))


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _criterion_states(
    criteria: list[str],
    preferred_source_types: list[str],
    *,
    question: str = "",
    query_hints: list[str] | None = None,
    level_policy: str = "explicit",
    answer_contract_level: str = "hard",
    criteria_are_contract_dimensions: bool = False,
) -> list[ResearchCriterionState]:
    output: list[ResearchCriterionState] = []
    legacy_unversioned = level_policy == LEGACY_CRITERION_LEVEL_POLICY
    planner_v1 = level_policy == CRITERION_LEVEL_POLICY_V1
    level_source = "legacy_inferred" if legacy_unversioned else (
        "question_dimensions_v2" if criteria_are_contract_dimensions else "planner_v1" if planner_v1 else "explicit"
    )
    if planner_v1 and not criteria_are_contract_dimensions:
        output.append(ResearchCriterionState(
            criterion_id="criterion_01",
            description=f"实质回答研究问题：{question}" if question else "实质回答研究问题",
            query_hint=question,
            level="soft" if answer_contract_level == "soft" else "hard",
            level_source="planner_v1",
            preferred_source_types=list(preferred_source_types),
        ))
    for index, description in enumerate(criteria, start=1 + len(output)):
        lowered = description.lower()
        if planner_v1 and any(marker in lowered for marker in (
            "可视化", "图表", "截图", "信息图", "visualization", "chart", "可选",
        )):
            level = "optional"
        elif planner_v1 and criteria_are_contract_dimensions:
            # v5 question dimensions and v6 explicit scope obligations are accepted question
            # contracts. Do not also prepend a synthetic whole-question criterion: overlapping
            # ledgers let a broad covered label hide a concrete partial dimension.
            level = "soft" if answer_contract_level == "soft" else "hard"
        elif planner_v1:
            # The question shown to the user is the answer contract. Planner-expanded criteria
            # guide retrieval and enrichment, but a missing detail must not fail an otherwise
            # useful answer unless that detail was explicitly part of the question itself.
            level = "soft"
        elif criterion_requires_source_diversity(description):
            # Explicit corroboration/case breadth in the user-confirmed plan is substantive.
            # It is the sole path by which source diversity becomes a hard requirement.
            level = "hard"
        elif any(marker in lowered for marker in ("可视化", "图表", "截图", "信息图", "visualization", "chart", "可选")):
            level = "optional"
        elif any(marker in lowered for marker in (
            "至少", "来源数", "链接", "可追溯", "引用", "官方文档", "不确定性",
            "补充", "示例", "参数枚举", "活动数据", "背景信息", "更多案例",
            "source count", "citation count", "cite ", "citation", "source for", "official documentation",
            "supplementary", "example", "parameter list", "background",
        )):
            level = "soft"
        elif legacy_unversioned and _legacy_scope_expansion(description, question):
            level = "soft"
        else:
            level = "hard"
        query_index = index - 1 - (1 if planner_v1 else 0)
        query_hint = query_hints[query_index] if query_hints and query_index < len(query_hints) else ""
        output.append(ResearchCriterionState(
            criterion_id=f"criterion_{index:02d}",
            description=description,
            query_hint=query_hint,
            level=level,
            level_source=level_source,
            preferred_source_types=list(preferred_source_types),
        ))
    if (
        output
        and not any(item.level == "hard" for item in output)
        and not (planner_v1 and answer_contract_level == "soft")
    ):
        output.append(ResearchCriterionState(
            criterion_id=f"criterion_{len(output) + 1:02d}",
            description=f"实质回答研究问题：{question}" if question else "实质回答研究问题",
            level="hard",
            level_source="synthetic_core",
            preferred_source_types=list(preferred_source_types),
        ))
    return output


def _legacy_scope_expansion(description: str, question: str) -> bool:
    """Recognize old criteria that add a workflow the research question did not request.

    Old plans stored only strings, so every criterion was later defaulted to hard. This deliberately
    small rule covers obvious scope expansion (for example, asking for extension-development steps
    in a question about architecture) and does not attempt topic-specific semantic classification.
    """
    criterion = str(description or "").lower()
    prompt = str(question or "").lower()
    expansion_markers = (
        "开发步骤", "实现步骤", "编写步骤", "扩展步骤", "发布机制", "共享机制",
        "development steps", "implementation steps", "publishing mechanism", "sharing mechanism",
    )
    requested_markers = (
        "如何", "怎么", "步骤", "开发", "实现", "扩展", "发布", "共享",
        "how to", "steps", "develop", "implement", "extend", "publish", "share",
    )
    return any(marker in criterion for marker in expansion_markers) and not any(
        marker in prompt for marker in requested_markers
    )
