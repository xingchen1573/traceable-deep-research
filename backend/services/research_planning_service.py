"""Shared preflight planning for conversational and Web research entry points.

The main Agent tools and Research Studio must use the same clarification, plan generation,
and plan-revision implementation. This module deliberately has no LangChain tool decorators
and creates no jobs; callers decide how to present the result and when to start execution.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from typing import Any


_LOGGER = logging.getLogger(__name__)
_RECON_TOKEN_VERSION = "v2"
_PLAN_TOKEN_VERSION = "p1"


def _load_recon_token_secret() -> bytes:
    """Load the shared signing key, or create one for this process only.

    The generated fallback keeps a local single-process runtime usable without shipping a
    forgeable key in source. It deliberately does not guarantee cross-process or restart
    continuity: multi-worker deployments must configure the same high-entropy
    ``RECON_TOKEN_SECRET`` for every worker.
    """
    configured = os.getenv("RECON_TOKEN_SECRET", "").strip()
    if configured:
        return configured.encode("utf-8")
    _LOGGER.warning(
        "RECON_TOKEN_SECRET is not configured; using an ephemeral process-local signing key. "
        "Recon and plan tokens will not survive a restart and are not guaranteed to verify "
        "on another worker. Configure the same high-entropy RECON_TOKEN_SECRET for every "
        "worker in a multi-process deployment."
    )
    return secrets.token_bytes(32)


_RECON_TOKEN_SECRET = _load_recon_token_secret()


def _research_service_module():
    from . import research_service

    return research_service


def legacy_recon_token(recon_context: str) -> str:
    """Return the v1 token shape signed by the current process or configured secret.

    This supports same-secret in-flight conversations only. Tokens signed with the removed
    public default key are intentionally not accepted.
    """
    digest = hmac.new(
        _RECON_TOKEN_SECRET,
        recon_context.strip().encode("utf-8"),
        hashlib.sha256,
    )
    return digest.hexdigest()[:32]


def recon_token(recon_context: str) -> str:
    """Mint a signed, self-contained credential for a canonical recon context."""
    payload = base64.urlsafe_b64encode(recon_context.strip().encode("utf-8")).decode("ascii").rstrip("=")
    signed_payload = f"{_RECON_TOKEN_VERSION}.{payload}"
    signature = hmac.new(
        _RECON_TOKEN_SECRET,
        signed_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return f"{signed_payload}.{signature}"


def recon_context_from_token(value: str) -> str | None:
    """Verify a v2 token and recover its canonical context."""
    try:
        version, payload, signature = value.strip().split(".", 2)
    except ValueError:
        return None
    if version != _RECON_TOKEN_VERSION or not payload or not signature:
        return None
    signed_payload = f"{version}.{payload}"
    expected = hmac.new(
        _RECON_TOKEN_SECRET,
        signed_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()[:32]
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padding = "=" * (-len(payload) % 4)
        return base64.urlsafe_b64decode(f"{payload}{padding}").decode("utf-8").strip()
    except (UnicodeDecodeError, ValueError):
        return None


def resolve_research_strategy(user_request: str, requested_strategy: str = "auto") -> str:
    """Resolve the internal planning strategy consistently across Web and conversational entry points."""
    requested = str(requested_strategy or "auto").strip().lower()
    if requested == "hv_analysis":
        return "hv_analysis"
    if requested == "standard":
        return "standard"
    text = str(user_request or "").lower()
    comparison_markers = (
        "对比",
        "比较",
        "横向",
        "纵向",
        "演化",
        "竞品",
        "多个对象",
        "多对象",
        " vs ",
        " versus ",
    )
    return "hv_analysis" if any(marker in f" {text} " for marker in comparison_markers) else "standard"


def _canonical_plan_json(confirmed_plan: dict[str, Any]) -> str:
    return json.dumps(confirmed_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def plan_id(confirmed_plan: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_plan_json(confirmed_plan).encode("utf-8")).hexdigest()[:24]


def plan_token(confirmed_plan: dict[str, Any]) -> str:
    identifier = plan_id(confirmed_plan)
    signed_payload = f"{_PLAN_TOKEN_VERSION}.{identifier}"
    signature = hmac.new(
        _RECON_TOKEN_SECRET,
        f"{signed_payload}.{_canonical_plan_json(confirmed_plan)}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]
    return f"{signed_payload}.{signature}"


def verify_plan_token(confirmed_plan: dict[str, Any], value: str) -> bool:
    try:
        version, identifier, signature = str(value or "").strip().split(".", 2)
    except ValueError:
        return False
    if version != _PLAN_TOKEN_VERSION or identifier != plan_id(confirmed_plan):
        return False
    expected = plan_token(confirmed_plan).rsplit(".", 1)[-1]
    return hmac.compare_digest(signature, expected)


def render_research_plan_markdown(sections: list[dict[str, Any]]) -> str:
    """Render the exact structured plan without model-authored summarization."""
    clean_sections = [item for item in sections if isinstance(item, dict) and str(item.get("title") or "").strip()]
    lines = ["## 调研规划", "", f"共 {len(clean_sections)} 个章节。", ""]
    for section in clean_sections:
        title = str(section.get("title") or "").strip()
        intent = str(section.get("intent") or "").strip()
        lines.append(f"### {title}")
        if intent:
            lines.append(intent)
        questions = [item for item in section.get("questions") or [] if isinstance(item, (dict, str))]
        if questions:
            lines.extend(["", "研究问题："])
        for index, question in enumerate(questions, start=1):
            if isinstance(question, dict):
                text_value = str(question.get("text") or question.get("question") or "").strip()
                criteria = [str(item).strip() for item in question.get("success_criteria") or [] if str(item).strip()]
            else:
                text_value = str(question).strip()
                criteria = []
            if not text_value:
                continue
            lines.append(f"{index}. {text_value}")
            for criterion in criteria:
                lines.append(f"   - {criterion}")
        lines.append("")
    lines.append("确认后请回复“开始”；需要调整时直接说明要保留、重做或新增的章节。")
    return "\n".join(lines).strip()


def ready_plan_result(
    sections: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
    user_request: str = "",
    strategy: str = "auto",
) -> dict[str, Any]:
    clean_sections = [item for item in sections if isinstance(item, dict)]
    question_count = sum(len(item.get("questions") or []) for item in clean_sections)
    source_plan = plan if isinstance(plan, dict) else {}
    confirmed_plan = {
        "version": "v1",
        "user_request": str(user_request or source_plan.get("user_request") or "").strip(),
        "topic": str(source_plan.get("topic") or "").strip(),
        "strategy": resolve_research_strategy(
            str(user_request or source_plan.get("user_request") or ""),
            str(source_plan.get("strategy") or strategy),
        ),
        "research_type": str(source_plan.get("research_type") or "").strip(),
        "subjects": [str(item).strip() for item in source_plan.get("subjects") or [] if str(item).strip()],
        "comparison_axes": [
            str(item).strip() for item in source_plan.get("comparison_axes") or [] if str(item).strip()
        ],
        "sections": clean_sections,
    }
    return {
        "status": "ready",
        "planned_sections": clean_sections,
        "confirmed_plan": confirmed_plan,
        "plan_id": plan_id(confirmed_plan),
        "plan_token": plan_token(confirmed_plan),
        "strategy": confirmed_plan["strategy"],
        "plan_markdown": render_research_plan_markdown(clean_sections),
        "section_count": len(clean_sections),
        "question_count": question_count,
    }


def clarify_concept(
    user_request: str,
    *,
    research_service_module: Any | None = None,
) -> dict[str, Any]:
    """Run the existing lightweight concept reconnaissance used before research planning."""
    research_service = research_service_module or _research_service_module()
    reconnaissance = research_service._research_reconnaissance(
        user_request,
        fetch_document_limit=int(os.getenv("RESEARCH_PLAN_RECON_FETCH_DOCUMENTS", "0")),
        timeout_seconds=float(os.getenv("RESEARCH_PLAN_RECON_TIMEOUT_SECONDS", "12")),
    )
    if reconnaissance.get("verification_status") != "verified":
        return {
            "status": "unavailable",
            "failure_reason": reconnaissance.get("failure_reason") or "concept_verification_unavailable",
            "message": "概念核验暂时不可用，请稍后重试；在完成核验前不会继续生成调研计划。",
            "recommended_strategy": resolve_research_strategy(user_request),
        }
    ambiguities = reconnaissance.get("ambiguities") or []
    if ambiguities:
        context = str(reconnaissance.get("summary") or "")
        return {
            "status": "ambiguous",
            "ambiguities": ambiguities,
            "entity_type": reconnaissance.get("entity_type") or "",
            "suggested_question": (
                "你说的「" + user_request.strip() + "」里，"
                + "；".join(str(item) for item in ambiguities)
                + "。请确认具体指的是哪一个，以免调研跑偏。"
            ),
            "recon_context": context,
            "recon_token": recon_token(context),
            "recommended_strategy": resolve_research_strategy(user_request),
        }

    deep_recon = (
        research_service._agentic_recon(user_request)
        if os.getenv("RESEARCH_PLAN_AGENTIC_RECON", "0").strip().lower() in {"1", "true", "yes"}
        else {}
    )
    context = str(deep_recon.get("summary") or "").strip() or str(reconnaissance.get("summary") or "")
    entity_type = str(deep_recon.get("entity_type") or "").strip() or str(reconnaissance.get("entity_type") or "")
    disambiguation_terms = deep_recon.get("disambiguation_terms") or reconnaissance.get("disambiguation_terms") or []
    return {
        "status": "clear",
        "entity_type": entity_type,
        "recon_context": context,
        "recon_token": recon_token(context),
        "disambiguation_terms": disambiguation_terms,
        "recommended_strategy": resolve_research_strategy(user_request),
    }


def draft_research_plan(
    user_request: str,
    *,
    topic: str = "",
    recon_context: str = "",
    recon_token_value: str = "",
    strategy: str = "standard",
    adjustment: dict[str, Any] | None = None,
    research_service_module: Any | None = None,
) -> dict[str, Any]:
    """Generate or revise a user-reviewable research plan using the production planner."""
    research_service = research_service_module or _research_service_module()
    clean_recon_token = recon_token_value.strip()
    canonical_recon_context = recon_context_from_token(clean_recon_token)
    if canonical_recon_context is None and recon_context.strip() and hmac.compare_digest(
        clean_recon_token,
        legacy_recon_token(recon_context),
    ):
        canonical_recon_context = recon_context.strip()
    if not clean_recon_token and not recon_context.strip():
        return {
            "status": "blocked",
            "reason": "missing_recon_context",
            "message": "缺少概念澄清凭证：请先调用 clarify_concept_tool，再把其返回的 recon_token 传入本工具，不能跳过概念澄清直接规划。",
        }
    if not canonical_recon_context:
        return {
            "status": "blocked",
            "reason": "invalid_recon_token",
            "message": "recon_token 无效或已被修改：请把 clarify_concept_tool 最近一次返回的 recon_token 原样传入，不需要重新复制或改写 recon_context。",
        }

    context = canonical_recon_context
    clean_strategy = resolve_research_strategy(user_request, strategy)
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    resolved_topic = research_service._suggest_topic({"topic": clean_topic}, user_request)
    timeout_seconds = float(os.getenv("PLAN_DRAFT_TIMEOUT_SECONDS", "45"))

    keep_sections: list[dict[str, Any]] = []
    stub_sections: list[dict[str, Any]] = []
    redo_sections: list[dict[str, Any]] = []
    if isinstance(adjustment, dict) and (adjustment.get("keep") or adjustment.get("redo") or adjustment.get("add")):
        for item in adjustment.get("keep") or []:
            if isinstance(item, dict) and item.get("title"):
                keep_sections.append(item)
        for item in adjustment.get("redo") or []:
            if isinstance(item, dict) and item.get("title"):
                stub_sections.append(item)
                redo_sections.append(item)
        for item in adjustment.get("add") or []:
            if isinstance(item, dict) and item.get("title"):
                stub_sections.append(item)
        if not stub_sections:
            normalized = research_service._normalize_research_plan(
                {"sections": keep_sections},
                user_request=user_request,
                topic=resolved_topic,
                strategy=clean_strategy,
                planner_mode="authored_plan",
            )
            if not normalized:
                return {"status": "unavailable", "failure_reason": "keep_only_adjustment_invalid"}
            return ready_plan_result(
                normalized["sections"], plan=normalized, user_request=user_request, strategy=clean_strategy
            )

        augmented_recon_context = context
        for item in redo_sections:
            supplemental = research_service._agentic_recon_for_section(
                user_request=user_request,
                section_title=str(item.get("title") or ""),
                section_axis=str(item.get("axis") or item.get("intent") or ""),
            )
            if supplemental:
                augmented_recon_context = f"{augmented_recon_context}\n{supplemental}"
        enriched = research_service._enrich_explicit_plan_questions(
            plan={"sections": stub_sections},
            user_request=user_request,
            recon_context=augmented_recon_context,
            timeout_seconds=timeout_seconds,
            strict=True,
        )
        if enriched.get("_planning_unavailable"):
            return {"status": "unavailable", "failure_reason": enriched.get("_failure_reason") or "unknown"}
        merged_sections = keep_sections + (enriched.get("sections") or [])
        normalized = research_service._normalize_research_plan(
            {"sections": merged_sections},
            user_request=user_request,
            topic=resolved_topic,
            strategy=clean_strategy,
            planner_mode="authored_plan",
        )
        if not normalized:
            return {"status": "unavailable", "failure_reason": "normalize_failed"}
        return ready_plan_result(
            normalized["sections"], plan=normalized, user_request=user_request, strategy=clean_strategy
        )

    model_plan = research_service._model_research_plan(
        user_request=user_request,
        topic=resolved_topic,
        strategy=clean_strategy,
        recon_context=context,
        timeout_seconds=timeout_seconds,
        strict=True,
    )
    if model_plan.get("_planning_unavailable"):
        return {"status": "unavailable", "failure_reason": model_plan.get("_failure_reason") or "unknown"}
    normalized = research_service._normalize_research_plan(
        model_plan,
        user_request=user_request,
        topic=resolved_topic,
        strategy=clean_strategy,
        planner_mode="model",
    )
    if not normalized:
        return {"status": "unavailable", "failure_reason": "normalize_failed"}
    return ready_plan_result(
        normalized["sections"], plan=normalized, user_request=user_request, strategy=clean_strategy
    )
