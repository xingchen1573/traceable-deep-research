"""Dispatcher tools that hand web-source research off to the backend job pipeline.

See docs/modules/agent-dispatcher-role-v0.1.md for the full design rationale. In short:
the main Agent no longer runs its own multi-round web retrieval + save_knowledge_set loop
for source_mode=web research (that stays reserved for source_mode=repo, see prompts.py).
Instead it dispatches to backend/services/research_service.py's question_loop job pipeline
via these tools and reports back to the user: start_research_job_tool (create),
check_research_job_tool (progress / waiting-approval lookup), approve_research_job_tool
(commit to the hierarchical knowledge tree), revise_report_section_tool (post-delivery section rewrite).

start_single_url_brief_tool is a separate, independent path -- see
docs/modules/single-url-brief-pipeline-v0.1.md. It dispatches to the same job_service
infrastructure but a different pipeline_mode ("single_url"), a short fetch-one-URL ->
two-stage-generate -> deliver flow that skips question_loop's multi-source evidence chain,
coverage gate, and HITL approval entirely. Its jobs report pipeline_mode="single_url" and
are never revisable (check_research_job_tool's `revisable` flag stays False for them, same
as any non-question_loop job) and never reach waiting_approval -- approve_research_job_tool
does not apply to them; saving to knowledge_base afterwards, once the user explicitly asks
for it, goes through save_single_url_brief_tool -- a separate HITL-gated tool below (mirrors
approve_research_job_tool's role for question_loop jobs), keeping "generate brief" and "save
to knowledge base" two decoupled actions.

revise_report_section_tool folds in what used to be a router-layer pre-filter
(channel_router_service.py's find_revisable_job_for_channel + extract_section_revision_intent)
-- see 3.8 of the design doc for why that moved into the Agent's own reasoning instead of
staying a standalone classifier. check_research_job_tool's `revisable` flag is what lets the
Agent (and this tool) tell a question_loop waiting_approval job apart from a legacy-pipeline
one before attempting a revision.

Hard constraints (do not relax without re-opening the design doc):
- start_research_job_tool always forces pipeline_mode="question_loop". Deep research has one
  production pipeline; historical jobs from removed modes remain readable but cannot be revised
  (see the feishu-channel-router-legacy-pipeline-bug lesson referenced throughout
  research_service.py).
- approve_research_job_tool must remain registered under HumanInTheLoopMiddleware's
  interrupt_on in main.py::build_agent(). It creates and immediately confirms a hierarchy
  archive preview after that approval, and must never be reachable without human approval.
- revise_report_section_tool must NOT rebuild section_id validation or job-state checks; it
  must keep calling revise_research_sections() and surfacing its ValueError as-is, so the
  existing defenses (invalid section_id rejected, non-question_loop pipeline_mode rejected,
  wrong job status rejected) stay the single source of truth.
- start_single_url_brief_tool must NOT be added to HumanInTheLoopMiddleware's interrupt_on --
  single_url jobs are meant to deliver without approval (design doc 7). It must also never
  gain a code path that calls save_knowledge/save_knowledge_set itself; that stays a decoupled
  follow-up action, kept as its own tool (save_single_url_brief_tool below) rather than folded
  into this one, so "generate brief" and "save to knowledge base" remain two separate calls.
- save_single_url_brief_tool (the *only* tool that may commit a single_url job's brief into
  knowledge_base) must remain registered under HumanInTheLoopMiddleware's interrupt_on in
  main.py::build_agent(), exactly like save_knowledge_set/approve_research_job_tool -- it
  ultimately calls the same internal knowledge-writing path and must never be reachable
  without human approval, even though the single_url *generation* step above it isn't gated.

research_service / job_service are imported lazily inside each function (not at module
import time) to avoid a load-time import cycle: tools/__init__.py -> this module ->
backend.services.research_service -> ...workers -> ...tools (partially-initialized).
Deferring the import until the tool actually runs sidesteps the ordering problem entirely
(the same pattern backend/services/feishu_service.py already uses for job_service).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

try:
    from ._registry import conversation_channel_id, current_thread_id
except ImportError:  # pragma: no cover - script mode fallback
    from tools._registry import conversation_channel_id, current_thread_id


def _research_service():
    try:
        from ..backend.services import research_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_service
    return research_service


def _job_service_module():
    try:
        from ..backend.services.job_service import job_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services.job_service import job_service
    return job_service


def _research_planning_service():
    try:
        from ..backend.services import research_planning_service
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_planning_service
    return research_planning_service


# Alias kept for this module's existing call sites -- the real implementation now lives in
# tools/_registry.py so tools/knowledge_manager.py's channel <-> topic binding tools can share
# it too (see docs/VISION.md 第3节).
_conversation_channel_id = conversation_channel_id


def list_conversation_jobs(
    *, thread_id: str | None = None, status: str | None = None, limit: int = 10
) -> list[dict[str, Any]]:
    """Plain helper (not an agent tool) listing research jobs tagged to a conversation.

    Used by main.py's CLI loop to check for unread job notifications before prompting for
    input, and available for other surfaces that want the same lookup without going through
    a tool call.
    """
    job_service = _job_service_module()
    channel_id = _conversation_channel_id(thread_id)
    return job_service.list(job_type="research", channel_id=channel_id, status=status, limit=limit)


@tool
def start_research_job_tool(
    user_request: str,
    research_strategy: str = "auto",
    depth: str = "normal",
    topic: str = "",
    planned_sections: list[dict[str, Any]] | None = None,
    confirmed_plan: dict[str, Any] | None = None,
    plan_id: str = "",
    plan_token: str = "",
    disambiguation_terms: list[str] | None = None,
) -> str:
    """立项：把一个 source_mode=web 的调研请求交给工程化调研 job pipeline 异步执行。

    这是主 Agent 作为"调度者"分发调研任务的唯一入口，取代了旧的
    web_worker_tool 多轮检索 + research_coverage_check + save_knowledge_set 自助流程。
    调用前应确认：
    - 请求确实是需要落库的 source_mode=web 调研（一次性事实核查不要走这里，直接用
      web_worker_tool）。
    - 已完成必要的需求理解：范围、深度、输出形式不模糊；模糊时应先向用户追问，不要盲目立项。
    - 已依次走完 clarify_concept_tool → draft_research_plan_tool → 多轮打磨，并且用户已明确
      说"直接开始"确认了规划草稿——调用本工具时必须把最后一次返回的 confirmed_plan、plan_id、
      plan_token 原样带上，不要在没有走完规划确认流程的情况下裸调本工具。

    参数：
    - user_request：完整的调研请求描述，尽量保留用户原话中的范围/深度/输出形式信息。
    - research_strategy：优先原样传 draft_research_plan_tool 返回的 strategy；未传时工具会按统一
      规则推断，避免 Web 与飞书入口各自判断。
    - depth："normal"（默认）或 "deep"（用户明确要求深度、全面、系统性调研时使用）。
    - topic：用户明确指定的知识库归档 topic/文件夹名；用户没有明确指定时留空，不要自行提炼。
    - confirmed_plan/plan_id/plan_token：draft_research_plan_tool 最后一次返回、并经用户确认定稿的
      完整结构和签名凭证，三者原样传入，不要手动改写或精简。planned_sections 仅保留为旧调用方
      的显式兼容输入；对话入口应使用带签名的 confirmed_plan。
    - disambiguation_terms：clarify_concept_tool 返回的消歧词列表，跟 planned_sections 一起原样
      传入（同一次规划确认的一部分，不要单独省略）；job 会用它替代自己重新调用概念侦察，不会
      展示给用户，只用于防止后续检索跑偏到同名实体。

    调用后请把返回的 job_id 原样告诉用户（例如"已创建调研任务，编号 job_xxx，完成后会在这里
    提示你确认"），不要编造任务已经完成或已经生成报告。
    """
    research_service = _research_service()
    planning_service = _research_planning_service()
    channel_id = _conversation_channel_id()
    is_feishu = channel_id.startswith("feishu:")
    plan_strategy = str((confirmed_plan or {}).get("strategy") or research_strategy)
    strategy = planning_service.resolve_research_strategy(user_request, plan_strategy)
    depth_value = depth if depth in {"normal", "deep"} else "normal"
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    clean_planned_sections = [item for item in planned_sections or [] if isinstance(item, dict)]
    clean_confirmed_plan = confirmed_plan if isinstance(confirmed_plan, dict) else None
    try:
        from ..backend.services import research_plan_store
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_plan_store
    latest = research_plan_store.get_latest_plan(current_thread_id())
    if (
        latest
        and latest.get("persistence_status") == "started"
        and str(latest.get("plan_id") or "") == str(plan_id or "").strip()
        and latest.get("started_job_id")
    ):
        return json.dumps(
            {
                "status": "already_started",
                "job_id": latest["started_job_id"],
                "message": "这份确认计划已经创建过调研任务，未重复立项。",
            },
            ensure_ascii=False,
        )
    if clean_confirmed_plan is None or not str(plan_id or "").strip() or not str(plan_token or "").strip():
        return json.dumps(
            {
                "status": "blocked",
                "reason": "confirmed_plan_required",
                "message": "缺少带签名的用户确认计划，已拒绝创建任务。请先生成计划；若服务刚重启，请调用 get_latest_research_plan_tool 恢复最近草稿。",
                "recoverable_plan_id": (latest or {}).get("plan_id"),
            },
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {
        "user_request": user_request,
        "source_mode": "web",
        "research_strategy": strategy,
        "depth": depth_value,
        "topic": clean_topic or None,
        "output_mode": "document_set",
        "auto_save": False,
        "channel_id": channel_id,
        "delivery_target": "feishu_doc" if is_feishu else "web",
        # HARD CONSTRAINT: always question_loop. Never default/legacy -- see module docstring.
        "pipeline_mode": "question_loop",
        "plan_policy": "confirmed_required",
        "planned_sections": clean_planned_sections,
        "confirmed_plan": clean_confirmed_plan,
        "plan_id": str(plan_id or "").strip() or None,
        "plan_token": str(plan_token or "").strip() or None,
        "disambiguation_terms": [str(item).strip() for item in disambiguation_terms or [] if str(item).strip()],
    }
    # Deliberately no max_rounds: question_loop treats it as a per-question retrieval cap
    # (QuestionBudget.for_depth already picks a depth-appropriate default), not the legacy
    # pipeline's single global round cap. Setting one here would silently shrink a "deep"
    # request's per-question budget -- same reasoning channel_router_service.py used to avoid
    # this for Feishu before question_loop became the only pipeline_mode this tool ever uses.
    try:
        job = research_service.create_research_job(payload)
    except ValueError as exc:
        return json.dumps(
            {"status": "blocked", "reason": "invalid_confirmed_plan", "message": str(exc)},
            ensure_ascii=False,
        )
    research_plan_store.mark_plan_started(current_thread_id(), str(job.get("input", {}).get("plan_id") or plan_id), job["job_id"])
    result = {
        "status": "research_started",
        "job_id": job["job_id"],
        "job_type": job.get("job_type"),
        "job_status": job.get("status"),
        "message": "调研任务已创建，正在后台执行，可用 check_research_job_tool 查询进度。",
    }
    return json.dumps(result, ensure_ascii=False)


def _legacy_recon_token(recon_context: str) -> str:
    return _research_planning_service().legacy_recon_token(recon_context)


def _recon_token(recon_context: str) -> str:
    return _research_planning_service().recon_token(recon_context)


def _recon_context_from_token(recon_token: str) -> str | None:
    return _research_planning_service().recon_context_from_token(recon_token)


def _render_research_plan_markdown(sections: list[dict[str, Any]]) -> str:
    return _research_planning_service().render_research_plan_markdown(sections)


def _ready_plan_result(sections: list[dict[str, Any]]) -> str:
    return json.dumps(_research_planning_service().ready_plan_result(sections), ensure_ascii=False)


@tool
def clarify_concept_tool(user_request: str) -> str:
    """概念澄清（立项前必调）：对一次 source_mode=web 调研请求先做轻量 recon
    （默认浅搜 5 条并只看摘要；可配置少量正文抓取）判断请求里的概念/实体是否有歧义
    （比如撞上一个同名但不相关的真实公司/产品）。默认到这里结束并生成 recon_context，避免
    用户只是想立项时先等待一轮分钟级深检索；正式的多轮检索留给调研任务执行阶段。

    只有显式设置 RESEARCH_PLAN_AGENTIC_RECON=1 时，无歧义路径才会额外做一轮 agentic
    主题探索。该开关用于专项评测或确实需要“先探索再规划”的场景，不是普通立项默认步骤。

    如果用户在 user_request 里直接给了参考链接/资料（比如"这是最新发布说明：<url>"），原样
    保留在 user_request 文本里传入即可，不需要单独摘出来——深挖阶段的 agent 会自己识别并
    直接抓取该链接正文，视为高置信度材料。

    调用时机（硬约束）：任何一次要落库的 web 调研，立项前必须先调这个工具，不能凭感觉跳过；
    这是 draft_research_plan_tool 的固定前置和数据源。

    返回结构：
    - status="ambiguous"：命中歧义，返回 ambiguities（冲突说明）和 suggested_question
      （建议直接转述给用户的追问）。这一轮不要再调 draft_research_plan_tool，等用户回答后
      下一轮把回答并入 user_request 再重新调用本工具（或直接把澄清结果转述给
      draft_research_plan_tool 的 recon_context）。命中歧义时不会触发第二段深挖。
    - status="unavailable"：搜索或判断暂时不可用；停止规划并如实告诉用户稍后重试。
    - status="clear"：无歧义，同一轮直接接着调 draft_research_plan_tool，把这里返回的
      recon_token 原样传过去（recon_context 可以一并传入，但规划工具会从带签名的 token 恢复
      本工具生成的原始上下文，不依赖模型逐字复制自然语言）。中间不要插入过渡回执，也不需要
      对用户说"我确认是XX"这类空话，规划草稿本身就是确认。不要编造或修改 recon_token。

    不挂人工审批（只读，无副作用，失败重试零成本）。
    """
    result = _research_planning_service().clarify_concept(
        user_request,
        research_service_module=_research_service(),
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def draft_research_plan_tool(
    user_request: str,
    topic: str = "",
    recon_context: str = "",
    recon_token: str = "",
    strategy: str = "auto",
    adjustment: dict[str, Any] | None = None,
) -> str:
    """规划草稿：调研究模型池，产一份完整 sections+questions 结构的调研规划草稿，
    交给用户过目/打磨，确认后把返回的 confirmed_plan、plan_id、plan_token 原样传给
    start_research_job_tool。

    调用前置（硬约束）：必须先调 clarify_concept_tool。
    - 无歧义：同一轮直接接着调本工具，把 clarify_concept_tool 返回的 recon_token 原样传入。
      recon_context 可以一并传入，规划工具会从 token 恢复澄清工具生成的原始上下文，因此模型
      对空格、标点或换行的无意规范化不会破坏调用；伪造或修改 token 仍会被拒绝。
    - 有歧义：这一轮不要调本工具，等用户回答歧义问题后，下一轮再把确认结果重新走一次
      clarify_concept_tool（拿新的 recon_context/recon_token 配对）再调本工具。
    - 多轮打磨（adjustment）复用同一轮 clarify 的 recon_context/recon_token 即可，不需要每次
      改一版草稿都重新走 clarify_concept_tool；但如果这次打磨里有 redo（要求"这节不对，重新
      查/重新做"），本工具会自动针对该 section 的方向先触发一轮聚焦的补充 agentic 探索（真的
      再去检索新材料，不是在旧材料上换个说法），拿到新证据后再重新生成该 section，因此 redo
      场景下这次调用会比 keep-only/add-only 更慢，属预期行为。

    首次出草稿：不传 adjustment，工具从零生成一份完整规划。

    多轮打磨（用户用自由语言说想怎么改，比如"总览和技术架构不用改，市场格局部分重新做，
    再加一节讲成本"）：主 agent 自己把这句话解析成 adjustment 约束字典再调用本工具，格式：
    - keep：用户要保留不变的 section，必须是上一轮草稿里的**完整 section 对象**（原样复制
      粘贴，不要只传 id 或标题——工具是无状态的，不知道上一轮草稿内容，只能靠这里传来的
      完整对象保证"原样不变"）。
    - redo：要重新生成的 section，传 {"title": "..."} 即可（也可以带 axis/intent 提示）。
    - add：要新增的 section，同样传 {"title": "..."}。
    调用后返回全新一整份重出的草稿（keep 的部分原样收录，redo/add 的部分重新生成），不是
    diff、不是补丁——主 agent 只负责把这份新草稿完整展示给用户，不需要自己合并新旧内容。

    解析用户改法意图时如果拿不准（比如说不清具体要改哪个 section），先追问用户，不要猜。

    失败处理：规划模型不可用/超时/限流时，本工具如实返回 status="unavailable" 和失败原因，
    不会编造或退回模板草稿；请把这个情况直接告知用户，不要假装规划成功。

    不挂人工审批（只读，无副作用）。
    """
    result = _research_planning_service().draft_research_plan(
        user_request=user_request,
        topic=topic,
        recon_context=recon_context,
        recon_token_value=recon_token,
        strategy=strategy,
        adjustment=adjustment,
        research_service_module=_research_service(),
    )
    if result.get("status") == "ready":
        try:
            from ..backend.services import research_plan_store
        except ImportError:  # pragma: no cover - script mode fallback
            from backend.services import research_plan_store
        research_plan_store.save_latest_plan(current_thread_id(), result)
    return json.dumps(result, ensure_ascii=False)


@tool
def get_latest_research_plan_tool() -> str:
    """恢复当前对话最近一次结构化调研计划，供服务重启或 checkpoint 丢失后继续确认立项。"""
    try:
        from ..backend.services import research_plan_store
    except ImportError:  # pragma: no cover - script mode fallback
        from backend.services import research_plan_store
    result = research_plan_store.get_latest_plan(current_thread_id())
    if not result:
        return json.dumps(
            {"status": "not_found", "message": "当前对话没有可恢复的结构化调研计划，请重新生成计划。"},
            ensure_ascii=False,
        )
    status = "already_started" if result.get("persistence_status") == "started" else "ready"
    return json.dumps({**result, "status": status}, ensure_ascii=False)


@tool
def start_single_url_brief_tool(url: str, user_request: str = "") -> str:
    """针对单个 URL 生成一份结构化简报（摘要/核心论点与数据/关键原文引用/点评），交给
    pipeline_mode="single_url" 的短流程 job 异步执行，完成后直接交付（不经过人工审批，不
    自动落知识库）。

    仅当用户消息里明确带了一个具体 URL，并且明确要求"总结/分析/解读这个链接"这类单链接分析
    意图时才调用本工具；不要用于：
    - 纯粹分享链接、闲聊引用、没有要求分析的场景（不做隐式全自动识别，避免误判）。
    - 需要多源交叉检索、对比、系统性调研的请求——那属于「八、调研调度规则」的
      start_research_job_tool，即使请求里恰好带了一个 URL 作为参考起点也一样。
    - 一次性事实核查（例如"这个链接说的版本号对不对"）——直接用 web_worker_tool，不要立项。

    调用前不需要额外追问确认，用户这句话里的意图已经足够明确（"总结一下这个链接"本身就是完整
    指令）；调用后请把返回的 job_id 原样告诉用户，简报生成完成后会在这里给出摘要和文档链接，
    不要编造任务已经完成或已经生成简报。

    参数：
    - url：待分析的完整 http(s) URL，原样传入用户消息中的链接，不要改写或拼接。
    - user_request：用户这句话里除 URL 之外表达的具体要求（例如想重点看哪方面），没有的话
      留空，工具会用默认措辞。
    """
    research_service = _research_service()
    channel_id = _conversation_channel_id()
    is_feishu = channel_id.startswith("feishu:")
    clean_url = url.strip() if isinstance(url, str) else ""
    if not re.match(r"^https?://", clean_url):
        return json.dumps(
            {"status": "invalid_url", "message": "url 必须是完整的 http(s) 链接，无法立项。"}, ensure_ascii=False
        )

    payload: dict[str, Any] = {
        "user_request": user_request.strip() if isinstance(user_request, str) else "",
        "url": clean_url,
        "source_mode": "web",
        "output_mode": "brief",
        "auto_save": False,
        "channel_id": channel_id,
        "delivery_target": "feishu_doc" if is_feishu else "web",
        "pipeline_mode": "single_url",
    }
    job = research_service.create_research_job(payload)
    result = {
        "status": "single_url_brief_started",
        "job_id": job["job_id"],
        "job_type": job.get("job_type"),
        "job_status": job.get("status"),
        "message": "单 URL 简报任务已创建，正在后台执行，可用 check_research_job_tool 查询进度。",
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def save_single_url_brief_tool(job_id: str = "", topic: str = "") -> str:
    """把一个已生成完成的单 URL 简报（pipeline_mode="single_url"）保存进知识库。

    这是 start_single_url_brief_tool 生成简报之后，"生成"和"保存"两个解耦动作里"保存"这一步
    的唯一合法入口——start_single_url_brief_tool 本身从不落库。这是真正的落库动作，会触发强制
    人工审批：调用本工具后，只有人工确认（approve/edit）才会真正写入知识库，拒绝（reject）
    则不会落库。

    只有当用户在简报生成完成之后，明确要求"存一下/保存到知识库/归档"这类意图时才调用；不要在
    用户只是看了简报、没有表达保存意图时主动调用。

    参数：
    - job_id：不传时使用当前会话最近一个 pipeline_mode="single_url" 的 job。
    - topic：知识库归档目录名，必须由用户明确给出或确认（不要自行提炼）；这个简报此前没有
      落库过，topic 不能留空。
    """
    research_service = _research_service()
    job_service = _job_service_module()

    resolved_job_id = job_id
    if not resolved_job_id:
        channel_id = _conversation_channel_id()
        candidates = job_service.list(job_type="research", channel_id=channel_id, limit=20)
        single_url_candidates = [c for c in candidates if (c.get("result") or {}).get("pipeline_mode") == "single_url"]
        if not single_url_candidates:
            return json.dumps(
                {"status": "no_job", "message": "当前会话没有找到可保存的单 URL 简报任务。"},
                ensure_ascii=False,
            )
        resolved_job_id = single_url_candidates[0]["job_id"]

    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if not clean_topic:
        return json.dumps(
            {"status": "topic_required", "message": "保存单 URL 简报到知识库前必须先确认归档 topic。"},
            ensure_ascii=False,
        )

    try:
        saved = research_service.approve_research(resolved_job_id, {"topic": clean_topic})
    except ValueError as exc:
        return json.dumps({"status": "rejected", "job_id": resolved_job_id, "error": str(exc)}, ensure_ascii=False)
    return json.dumps({"status": "saved", "job_id": resolved_job_id, **saved}, ensure_ascii=False)


@tool
def check_research_job_tool(job_id: str = "") -> str:
    """查询一个调研 job 的当前状态、进度和关键结果摘要。

    - 传入 job_id 时查询该 job。
    - 不传 job_id 时，按更新时间查询当前会话真正最新的 research job，不会为了找到
      waiting_approval 而跳过一个更新但已经 succeeded/failed 的任务。这样用户重复说“批准”时，
      不会静默切换到更早的遗留任务。
    - 每次准备判断用户是否在批准落库、要求返修某一节、或询问调研进度之前，应先调用本工具确认
      是否存在这样的 job，不要凭记忆判断——飞书等场景下，job 的完成提醒可能来自不经过你的后台
      推送（watchdog 直接发消息），你的推理记忆里不一定有"我问过是否保存"这件事；即使是你自己
      刚问过的问题，job 是否真的跑完也是异步的，工具调用当轮拿到的回执不等于任务已完成，必须
      靠这次查询获取最新状态，不能假设。
    - 返回的 sections 是本次报告的章节标题，可用于判断用户的修改意见指向哪一节。
    - 返回的 revisable 为 true 时才能调用 revise_report_section_tool；为 false（例如这是一个
      pipeline_mode 不是 question_loop 的旧版 job）时，章节级返修不可用，应告知用户这份报告不
      支持章节级修改，不要尝试调用 revise_report_section_tool（它会被 revise_research_sections()
      直接拒绝）。
    - 返回的 quality_detail（quality_gate 未生成过时为 null）带有逐章节的质量问题清单和
      claim_entailment 声明核验详情：
      issues 是 [{section_id, type, instruction}]（章节级 critic 发现的具体问题，如某节缺表格、
      某节综合缺乏证据支撑），weak_sections 是 [{section_id, title, reason, evidence_count}]
      （证据不足的章节），claim_entailment 包含不受证据支持、标签不匹配、非原子 claim 的数量和
      代表性 claim，并明确区分 blocking_claims 与 advisory_claims。非阻断的 label_mismatch 只会
      出现在 advisory_claims，不得据此推翻 quality_detail.approval_ready。
      用户追问"具体哪节有问题/为什么没通过质量门"时，必须同时报告 issues 和 claim_entailment，
      具体到章节和 claim，不能只回复笼统的 reason 摘要。
    - quality_detail.approval_ready=false 表示当前版本不能批准，且没有人工强制绕过入口。此时不得
      向用户提供"仍然批准/先批准当前版本"选项；应说明阻断项并建议返修或补查。
      quality_detail.approval_ready=true / approval_decision=approve_allowed 则表示当前版本可以批准；
      advisory_claim_count 不得被当成额外的隐藏硬门禁。用户已经明确说“落库/批准”时，应在同一轮
      调用 approve_research_job_tool，不要再次询问是否返修。
    """
    research_service = _research_service()
    job_service = _job_service_module()

    job: dict[str, Any] | None = None
    if job_id:
        try:
            job = research_service.get_research_job(job_id)
        except ValueError as exc:
            return json.dumps({"status": "not_found", "error": str(exc)}, ensure_ascii=False)
    else:
        channel_id = _conversation_channel_id()
        job = job_service.latest(job_type="research", channel_id=channel_id)
        if job is None:
            return json.dumps(
                {"status": "no_job", "message": "当前会话没有找到调研任务。"}, ensure_ascii=False
            )

    result = job.get("result") or {}
    blueprint = result.get("report_blueprint") or {}
    sections = [
        {"section_id": section.get("section_id"), "title": section.get("title")}
        for section in blueprint.get("sections") or []
        if section.get("title")
    ]
    if sections:
        # "editorial" (executive summary + conclusion) is not a blueprint section, but
        # revise_research_sections() accepts it as a manual-revision target -- without this
        # pseudo-entry the agent has no way to discover that "editorial" is a valid section_id,
        # so it would never pass it to revise_report_section_tool even when the user explicitly
        # asks to rewrite the summary/conclusion. See docs/modules/deep-research-step4-
        # approval-and-delivery-discussion-v0.1.md 商榷点一.
        sections.append({"section_id": "editorial", "title": "执行摘要与结论"})
    pipeline_mode = result.get("pipeline_mode")
    payload = {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "already_approved": job.get("status") == "succeeded",
        "progress": job.get("progress"),
        "waiting_approval": job.get("status") == "waiting_approval",
        "pipeline_mode": pipeline_mode,
        # Mirrors the legacy-pipeline guard research_service.py::revise_research_sections()
        # already enforces (rejects anything that isn't pipeline_mode="question_loop") -- see
        # the feishu-channel-router-legacy-pipeline-bug lesson referenced throughout that file.
        "revisable": bool(job.get("status") == "waiting_approval" and pipeline_mode == "question_loop"),
        "suggested_topic": result.get("suggested_topic") or (job.get("input") or {}).get("topic"),
        "sections": sections,
        "message": result.get("message") or "",
        "error": job.get("error"),
        "quality_detail": _quality_detail(result),
    }
    return json.dumps(payload, ensure_ascii=False)


def _quality_detail(result: dict[str, Any]) -> dict[str, Any] | None:
    """Surface the section-level quality_gate detail that combined_quality_gate() /
    _quality_gate() already computed but that previously never left research_service.py --
    see docs' "check_research_job_tool 丢失 quality_gate 章节级细节" investigation. Both the
    new question_loop gate (backend/research/quality.py::combined_quality_gate, whose
    checks.report_quality.issues is already truncated to 8 by evaluate_report_quality) and the
    legacy/targeted-followup gate (research_service.py::_quality_gate, whose top-level
    checks.weak_sections holds the untruncated per-section list) are handled here without
    duplicating either truncation/aggregation logic.
    """
    quality_gate = result.get("quality_gate") if isinstance(result.get("quality_gate"), dict) else {}
    if not quality_gate:
        return None
    checks = quality_gate.get("checks") if isinstance(quality_gate.get("checks"), dict) else {}
    report_quality = checks.get("report_quality") if isinstance(checks.get("report_quality"), dict) else {}
    issues = report_quality.get("issues") if isinstance(report_quality.get("issues"), list) else []
    weak_sections = checks.get("weak_sections") if isinstance(checks.get("weak_sections"), list) else []
    approval_ready = bool(quality_gate.get("approval_ready"))
    return {
        "status": quality_gate.get("status"),
        "approval_ready": approval_ready,
        "approval_decision": "approve_allowed" if approval_ready else "approval_blocked",
        "reason": quality_gate.get("reason"),
        "issues": issues,
        "weak_sections": weak_sections,
        "claim_entailment": _claim_entailment_detail(result),
        "matrix_coverage": result.get("matrix_coverage") or None,
        "requires_matrix_gap_acknowledgement": bool(
            (result.get("matrix_coverage") or {}).get("requires_gap_acknowledgement")
        ),
    }


def _claim_entailment_detail(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return compact, Agent-facing claim-entailment diagnostics.

    New question-loop jobs annotate every judgment with ``quality_gate_blocking``. That field is
    authoritative for the Agent-facing blocking/advisory split; the raw entailment status can be
    ``failed`` for non-blocking label mismatches. Older jobs do not have the annotation, so they
    retain the conservative legacy behavior where unsupported/label_mismatch/non_atomic rows are
    treated as blocking.
    """
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    entailment = validation.get("entailment") if isinstance(validation.get("entailment"), dict) else None
    if entailment is None:
        diagnostics = (
            result.get("pipeline_diagnostics")
            if isinstance(result.get("pipeline_diagnostics"), dict)
            else {}
        )
        entailment = (
            diagnostics.get("entailment")
            if isinstance(diagnostics.get("entailment"), dict)
            else None
        )
    if entailment is None:
        return None

    blocking_statuses = {"unsupported", "label_mismatch", "non_atomic"}
    judgments = [item for item in entailment.get("judgments") or [] if isinstance(item, dict)]
    problem_judgments = [item for item in judgments if item.get("status") in blocking_statuses]
    blocking = [
        item
        for item in problem_judgments
        if item.get("quality_gate_blocking") is True
        or "quality_gate_blocking" not in item
    ]
    advisory = [
        item
        for item in problem_judgments
        if "quality_gate_blocking" in item and item.get("quality_gate_blocking") is not True
    ]
    substantive = [
        item
        for item in blocking
        if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", str(item.get("claim") or ""))
    ]
    def compact_claim(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "claim_index": item.get("claim_index"),
            "section_id": item.get("section_id"),
            "claim": str(item.get("claim") or "")[:500],
            "status": item.get("status"),
            "reason": str(item.get("reason") or "")[:300],
            "evidence_ids": list(item.get("evidence_ids") or [])[:8],
        }

    representative_claims = [
        compact_claim(item)
        for item in substantive[:8]
    ]
    advisory_claims = [compact_claim(item) for item in advisory[:8]]
    count_fields = (
        "claim_count",
        "entailed_count",
        "inference_count",
        "unsupported_count",
        "label_mismatch_count",
        "non_atomic_count",
        "model_reviewed_count",
        "deterministic_fallback_count",
    )
    raw_status = entailment.get("status")
    detail: dict[str, Any] = {}
    detail.update({key: int(entailment.get(key) or 0) for key in count_fields})
    if judgments:
        blocking_count = len(blocking)
    elif "blocking_core_unsupported_count" in entailment:
        blocking_count = int(entailment.get("blocking_core_unsupported_count") or 0)
    else:
        blocking_count = detail["unsupported_count"] + detail["label_mismatch_count"] + detail["non_atomic_count"]
    detail["status"] = "failed" if blocking_count else "passed"
    if blocking_count:
        detail["raw_status"] = raw_status
    detail.update({
        "approval_blocking": bool(blocking_count),
        "blocking_claim_count": blocking_count,
        "non_substantive_blocking_count": len(blocking) - len(substantive),
        "blocking_claims": representative_claims,
        "omitted_blocking_claim_count": max(
            0,
            blocking_count - len(representative_claims) - (len(blocking) - len(substantive)),
        ),
        "advisory_claim_count": len(advisory),
        "advisory_claims": advisory_claims,
        "omitted_advisory_claim_count": max(0, len(advisory) - len(advisory_claims)),
    })
    return detail


@tool
def approve_research_job_tool(
    job_id: str = "",
    topic: str = "",
    keywords: str | list[str] | None = None,
    acknowledge_matrix_gaps: bool = False,
) -> str:
    """批准一个处于 waiting_approval 状态的调研 job，将其归档到层级知识库。

    这是真正的落库动作，会触发强制人工审批：调用本工具后，只有人工确认（approve/edit）
    才会生成并确认层级归档，拒绝（reject）则不会落库。不要因为这一步有审批拦截就随意调用——
    调用前应先用 check_research_job_tool 确认：
    1) 确实存在一个 waiting_approval 的调研 job；
    2) quality_detail.approval_ready 必须为 true；为 false 时不得调用本工具，也不得向用户提供
       "仍然批准/先批准当前版本"选项，因为后端不存在手动绕过质量门的入口。必须向用户说明
       quality_detail.issues 和 quality_detail.claim_entailment 中的阻断问题，并建议返修或补查；
       为 true 时可以继续审批，不能因为存在 advisory_claims 而拒绝用户明确的批准意图；
    3) 用户这句话确实是在表达"可以了/确认保存/批准"，而不是在问别的问题或做其他确认。

    调用本工具时应把 check_research_job_tool 返回的确切 job_id 原样传入，避免把一句简短的
    “批准”关联到其他遗留任务。没有传 job_id 时，只检查当前会话按更新时间真正最新的调研任务；
    不会向后查找更旧的 waiting_approval 任务。最新任务已经 succeeded 时幂等返回“已经落库”。

    参数：
    - job_id：优先传入 check_research_job_tool 返回的确切 job_id。
    - topic：兼容旧调用方；非空时作为 ResearchItem 标题覆盖，不再创建扁平 topic 目录。
    - keywords：ResearchItem 标签，可选。
    - acknowledge_matrix_gaps：比较报告为 partial 时，仅在用户查看缺口并明确同意知情归档后设为 true。
    """
    research_service = _research_service()
    job_service = _job_service_module()

    target_job: dict[str, Any] | None = None
    resolved_job_id = job_id.strip() if isinstance(job_id, str) else ""
    if resolved_job_id:
        try:
            target_job = job_service.get(resolved_job_id)
        except ValueError as exc:
            return json.dumps({"status": "not_found", "job_id": resolved_job_id, "error": str(exc)}, ensure_ascii=False)
    else:
        channel_id = _conversation_channel_id()
        target_job = job_service.latest(job_type="research", channel_id=channel_id)
        if target_job is None:
            return json.dumps(
                {"status": "no_pending_job", "message": "当前会话没有找到等待审批的调研任务。"},
                ensure_ascii=False,
            )
        resolved_job_id = str(target_job.get("job_id") or "")

    job_status = str((target_job or {}).get("status") or "")
    job_result = (target_job or {}).get("result") or {}
    if job_status == "succeeded":
        archive_result = job_result.get("archive_result") if isinstance(job_result.get("archive_result"), dict) else {}
        return json.dumps(
            {
                "status": "already_approved",
                "job_id": resolved_job_id,
                "job_status": job_status,
                "topic": archive_result.get("title") or job_result.get("saved_topic") or job_result.get("suggested_topic"),
                "research_id": archive_result.get("research_id"),
                "path": archive_result.get("path"),
                "paths": archive_result.get("document_paths") or job_result.get("paths") or [],
                "message": "该调研任务已经落库，不会切换去批准其他旧任务。",
            },
            ensure_ascii=False,
        )
    if job_status != "waiting_approval":
        return json.dumps(
            {
                "status": "job_not_waiting_approval",
                "job_id": resolved_job_id,
                "job_status": job_status,
                "message": "当前最新调研任务不处于等待审批状态；未向后查找其他旧任务。",
            },
            ensure_ascii=False,
        )

    archive_payload: dict[str, Any] = {}
    clean_topic = topic.strip() if isinstance(topic, str) else ""
    if clean_topic:
        archive_payload["title"] = clean_topic
    if keywords:
        archive_payload["keywords"] = keywords if isinstance(keywords, list) else [keywords]
    archive_payload["acknowledge_matrix_gaps"] = acknowledge_matrix_gaps

    preview = research_service.preview_research_archive(resolved_job_id, archive_payload)
    saved = research_service.confirm_research_archive(resolved_job_id, str(preview["preview_id"]))
    return json.dumps(
        {
            "job_id": resolved_job_id,
            **saved,
            "status": "approved",
            "archive_preview": {
                "preview_id": preview.get("preview_id"),
                "recommendation": preview.get("recommendation"),
                "archive": preview.get("archive"),
            },
        },
        ensure_ascii=False,
    )


@tool
def revise_report_section_tool(
    section_ids: str | list[str],
    instruction: str,
    job_id: str = "",
) -> str:
    """对一个已交付、正在等待审批(waiting_approval)的调研报告发起章节级返修。

    这是 `backend/services/research_service.py::revise_research_sections()` 的薄封装，不新增
    任何返修执行逻辑：目标 section 之外的其他章节原样透传、绝不会被模型碰到；每个目标 section
    独立重写，有质量门拦截，不达标会自动重试一次；处理完统一跑一次总编环节；job 状态处理完后
    会回到 waiting_approval，和最初草稿走同一套审批入口——落库仍然只能通过
    approve_research_job_tool，本工具本身不落库，不需要 HITL 审批。

    调用前必须先用 check_research_job_tool 确认：
    1) 目标 job 存在且 revisable=true（pipeline_mode=question_loop 且处于 waiting_approval，
       否则会被 revise_research_sections() 直接拒绝）；
    2) 用户这句话确实是在要求修改报告内容——不要求用户明确说"修改""返修"这类词，很多真实请求
       是"能不能加些表格呢""让 04 节再充实点"这种没有触发词、也没直接点名"报告"两个字的说法，
       只要合理指向"让某部分内容更详细/更简洁/换个语气/加表格/补充缺失内容"，就应判断为返修请求；
       只有明显在聊别的事情（寒暄、全新话题）时才不是。
    3) section_ids 能从 check_research_job_tool 返回的章节列表 + 用户原话中确定；如果确定不了具体
       是哪一节，不要调用本工具，改为用对话追问用户具体是哪一节。
    4) 如果返回 status=evidence_insufficient，必须把 evidence_gaps 告知用户并建议先做针对性补查；
       不要声称章节已经修改，也不要重复调用本工具（它不会重新检索证据）。
    5) 如果返回 status=revision_started 但同时带有 skipped_evidence_insufficient 字段：说明这批
       section_ids 里有部分章节证据不足被跳过，只有剩下证据充分的章节被实际返修了。必须如实告知
       用户哪些章节改了、哪些因证据不足被跳过（用 skipped_evidence_insufficient.section_ids 和
       .evidence_gaps），并建议对被跳过的章节先做针对性补查——不要笼统地说"已全部返修"。

    参数：
    - section_ids：一个或多个 section_id（必须来自 check_research_job_tool 返回的章节列表），
      不要凭空编造不存在的 section_id——传入无效 id 会被直接拒绝。
    - instruction：用中文简洁复述用户想要的修改，尽量具体可执行。
    - job_id：不传时对当前会话最近一个 waiting_approval 的调研 job 发起返修。
    """
    research_service = _research_service()
    job_service = _job_service_module()

    resolved_job_id = job_id
    if not resolved_job_id:
        channel_id = _conversation_channel_id()
        candidates = job_service.list(job_type="research", channel_id=channel_id, status="waiting_approval", limit=1)
        if not candidates:
            return json.dumps(
                {"status": "no_pending_job", "message": "当前会话没有找到等待审批的调研任务。"},
                ensure_ascii=False,
            )
        resolved_job_id = candidates[0]["job_id"]

    ids = section_ids if isinstance(section_ids, list) else [section_ids]
    ids = [str(item).strip() for item in ids if str(item).strip()]
    if not ids:
        return json.dumps({"status": "error", "message": "section_ids is required"}, ensure_ascii=False)

    try:
        revision = research_service.revise_research_sections(
            resolved_job_id, {"section_ids": ids, "instruction": instruction}
        )
    except ValueError as exc:
        # revise_research_sections() itself rejects unknown section_ids / wrong job status /
        # non-question_loop pipeline_mode -- surface that rejection as-is rather than retrying
        # or reinterpreting it (see module docstring: no validation is rebuilt here).
        return json.dumps({"status": "rejected", "job_id": resolved_job_id, "error": str(exc)}, ensure_ascii=False)
    if revision.get("status") == "evidence_insufficient":
        return json.dumps(
            {
                **revision,
                "status": "rejected",
                "reason_code": "evidence_insufficient",
                "job_id": resolved_job_id,
                "message": (
                    f"{revision.get('message') or '目标章节证据不足，未执行返修。'}"
                    " 本轮到此结束：不要调用 web_worker_tool 或再次调用 revise_report_section_tool；"
                    "通用 worker 的结果无法写回该调研 job。请直接把 evidence_gaps 告知用户，"
                    "并说明需要另行发起可写回 job 的针对性补查。"
                ),
            },
            ensure_ascii=False,
        )
    return json.dumps({"status": "revision_started", "job_id": resolved_job_id, **revision}, ensure_ascii=False)
