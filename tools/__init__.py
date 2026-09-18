"""Tools exposed to the main Deep Research agent and restricted workers."""

from __future__ import annotations

from .file import list_files, read_file, write_file
from .knowledge import research_coverage_check, save_knowledge, save_knowledge_set
from .knowledge_qa import (
    get_active_knowledge_base_topic,
    index_knowledge_base_lightrag,
    list_knowledge_bases,
    query_knowledge_base,
    set_active_knowledge_base,
)
from .knowledge_eval import evaluate_knowledge_base_qa
from .knowledge_manager import (
    bind_channel_to_topic_tool,
    bind_channel_to_scope_tool,
    check_knowledge_base_index,
    delete_knowledge_base_index,
    get_channel_push_config_tool,
    get_channel_topic_binding_tool,
    confirm_knowledge_tree_change_tool,
    confirm_research_archive_tool,
    inspect_knowledge_base,
    inspect_theme_tool,
    list_personal_knowledge_tree_tool,
    list_knowledge_base_status,
    set_channel_push_config_tool,
    preview_knowledge_tree_change_tool,
    preview_research_archive_tool,
    query_personal_knowledge_scope_tool,
    summarize_knowledge_base,
    unbind_channel_topic_tool,
)
from .memory import (
    list_memory_topics,
    load_memory_topic,
    read_dream_report,
    read_memory_index,
    refresh_memory_index,
    search_memory,
    write_memory_topic,
)
from .research_jobs import (
    approve_research_job_tool,
    check_research_job_tool,
    clarify_concept_tool,
    draft_research_plan_tool,
    get_latest_research_plan_tool,
    list_conversation_jobs,
    revise_report_section_tool,
    save_single_url_brief_tool,
    start_research_job_tool,
    start_single_url_brief_tool,
)
from .skills import list_skills, load_skill
from .web import broad_discovery, docs_search, fetch_context7_context, fetch_url, fetch_url_with_chain, search_web

# Re-export internal helpers for backward compatibility
from ._registry import (
    configure_surfaced_registry,
    ensure_demo_files,
    ensure_knowledge_base,
    set_current_thread_id,
    skill_summaries,
)

__all__ = [
    # Skills
    "list_skills",
    "load_skill",
    # Memory
    "search_memory",
    "load_memory_topic",
    "write_memory_topic",
    "read_memory_index",
    "list_memory_topics",
    "refresh_memory_index",
    "read_dream_report",
    # Knowledge Base
    "research_coverage_check",
    "save_knowledge",
    "save_knowledge_set",
    "list_knowledge_bases",
    "set_active_knowledge_base",
    "get_active_knowledge_base_topic",
    "index_knowledge_base_lightrag",
    "query_knowledge_base",
    "evaluate_knowledge_base_qa",
    "list_knowledge_base_status",
    "inspect_knowledge_base",
    "inspect_theme_tool",
    "summarize_knowledge_base",
    "check_knowledge_base_index",
    "delete_knowledge_base_index",
    "bind_channel_to_topic_tool",
    "bind_channel_to_scope_tool",
    "unbind_channel_topic_tool",
    "get_channel_topic_binding_tool",
    "get_channel_push_config_tool",
    "set_channel_push_config_tool",
    "list_personal_knowledge_tree_tool",
    "preview_knowledge_tree_change_tool",
    "confirm_knowledge_tree_change_tool",
    "query_personal_knowledge_scope_tool",
    "preview_research_archive_tool",
    "confirm_research_archive_tool",
    # Research job dispatch
    "clarify_concept_tool",
    "draft_research_plan_tool",
    "get_latest_research_plan_tool",
    "start_research_job_tool",
    "check_research_job_tool",
    "approve_research_job_tool",
    "revise_report_section_tool",
    "start_single_url_brief_tool",
    "save_single_url_brief_tool",
    "list_conversation_jobs",
    # File
    "list_files",
    "read_file",
    "write_file",
    # Web
    "broad_discovery",
    "docs_search",
    "fetch_context7_context",
    "fetch_url",
    "fetch_url_with_chain",
    "search_web",
    # Internal helpers (backward compatibility)
    "configure_surfaced_registry",
    "ensure_demo_files",
    "ensure_knowledge_base",
    "set_current_thread_id",
    "skill_summaries",
]
