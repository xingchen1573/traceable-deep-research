import { JobSummary } from './api';

export type ResearchSourceMode = 'web' | 'repo' | 'mixed';
export type ResearchOutputMode = 'single_document' | 'document_set';
export type ResearchStrategy = 'standard' | 'hv_analysis';
export type ResearchDepth = 'normal' | 'deep';

export interface ResearchPlanQuestion {
  id?: string;
  text?: string;
  question?: string;
  purpose?: string;
  answer_contract_level?: string;
  success_criteria?: string[];
  preferred_source_types?: string[];
  dependencies?: string[];
  criteria_queries?: string[];
  [key: string]: unknown;
}

export interface ResearchPlanSection {
  id?: string;
  title: string;
  intent?: string;
  axis?: string;
  questions?: ResearchPlanQuestion[];
  preferred_source_types?: string[];
  evidence_budget?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ResearchPlanningClarifyRequest {
  user_request: string;
}

export interface ResearchPlanningClarifyResponse {
  status: 'clear' | 'ambiguous' | 'unavailable';
  entity_type?: string;
  ambiguities?: string[];
  suggested_question?: string;
  recon_token?: string;
  disambiguation_terms?: string[];
  failure_reason?: string;
  message?: string;
  recommended_strategy?: ResearchStrategy;
}

export interface ConfirmedResearchPlan {
  version: 'v1' | string;
  user_request: string;
  topic?: string;
  strategy: ResearchStrategy;
  research_type?: string;
  subjects?: string[];
  comparison_axes?: string[];
  sections: ResearchPlanSection[];
}

export interface ResearchPlanningAdjustment {
  keep?: ResearchPlanSection[];
  redo?: Array<Pick<ResearchPlanSection, 'title'> & Partial<ResearchPlanSection>>;
  add?: Array<Pick<ResearchPlanSection, 'title'> & Partial<ResearchPlanSection>>;
}

export interface ResearchPlanningDraftRequest {
  user_request: string;
  topic?: string;
  recon_token: string;
  strategy?: ResearchStrategy | 'auto';
  adjustment?: ResearchPlanningAdjustment;
}

export interface ResearchPlanningDraftResponse {
  status: 'ready' | 'blocked' | 'unavailable';
  reason?: string;
  message?: string;
  failure_reason?: string;
  planned_sections?: ResearchPlanSection[];
  confirmed_plan?: ConfirmedResearchPlan;
  plan_id?: string;
  plan_token?: string;
  strategy?: ResearchStrategy;
  plan_markdown?: string;
  section_count?: number;
  question_count?: number;
}

export interface ResearchJobCreateRequest {
  user_request: string;
  source_mode: ResearchSourceMode;
  research_strategy?: ResearchStrategy;
  depth?: ResearchDepth;
  topic?: string;
  channel_id?: string;
  requested_by?: string;
  delivery_target?: 'web' | 'feishu_doc';
  output_mode?: ResearchOutputMode;
  required_titles?: string[];
  planned_sections?: ResearchPlanSection[];
  confirmed_plan?: ConfirmedResearchPlan;
  plan_id?: string;
  plan_token?: string;
  plan_policy?: 'confirmed_required' | 'system_generated' | 'legacy_unplanned';
  disambiguation_terms?: string[];
  max_rounds?: number;
  auto_save?: boolean;
  /** 'question_loop' 是主题调研的默认链路；'single_url' 是链接简报链路。
   * legacy/shadow 是内部兼容参数，不应暴露给用户选择。 */
  pipeline_mode?: 'question_loop' | 'single_url';
  /** 仅 pipeline_mode === 'single_url' 时必填。 */
  url?: string;
}

export interface CoverageResult {
  ready_to_write: boolean;
  round_index: number;
  max_rounds: number;
  source_count: number;
  document_count: number;
  evidence_count: number;
  key_fact_count: number;
  evidence_pack_count?: number;
  depth_ready: boolean;
  answer_status_counts?: {
    complete?: number;
    partial?: number;
    uncovered?: number;
  };
  coverage_status_counts?: Record<string, number>;
  safety_blocked_count?: number;
  per_title_status?: Array<{ title: string; status: string }>;
  weak_sections?: Array<{
    section_id?: string;
    title?: string;
    evidence_count?: number;
    source_count?: number;
    domain_count?: number;
    question_count?: number;
    covered_question_count?: number;
    weak_matrix_cell_count?: number;
    weak_matrix_cells?: Array<{
      matrix_cell_id?: string;
      subject?: string;
      comparison_axis?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      reason?: string;
    }>;
    reason?: string;
  }>;
  covered_titles: string[];
  partial_titles?: string[];
  missing_titles: string[];
  fatal_gaps: string[];
  matrix_gap_count?: number;
  matrix_coverage?: ComparisonMatrixCoverage;
  followup_queries: string[];
  reason: string;
}

export interface ComparisonMatrixCoverage {
  enabled: boolean;
  status: 'complete' | 'partial' | 'not_applicable' | string;
  complete: boolean;
  required_cell_count?: number;
  complete_cell_count?: number;
  status_counts?: Record<string, number>;
  axes?: Array<{
    axis: string;
    status: 'complete' | 'partial' | string;
    required_cell_count: number;
    complete_cell_count: number;
  }>;
  cells?: ComparisonMatrixCell[];
  gaps?: ComparisonMatrixCell[];
  requires_gap_acknowledgement?: boolean;
}

export interface ComparisonMatrixCell {
  cell_id: string;
  subject: string;
  axis: string;
  status: 'complete' | 'partial' | 'uncovered' | 'exhausted' | 'unavailable' | string;
  evidence_ids?: string[];
  attempt_count?: number;
  unavailable_reasons?: string[];
  criteria?: Array<{
    criterion_id?: string;
    description?: string;
    status?: string;
    evidence_ids?: string[];
  }>;
}

export interface ResearchDocument {
  title: string;
  section_id?: string;
  content: string;
  evidence_refs?: Array<{ evidence_id?: string; chunk_id?: string; claim?: string; source_url?: string }>;
  evidence_table?: Array<{
    id?: string;
    section_id?: string;
    section_title?: string;
    question_id?: string;
    question?: string;
    subject?: string;
    comparison_axis?: string;
    matrix_cell_id?: string;
    claim?: string;
    quote?: string;
    source_url?: string;
    source_title?: string;
    source_domain?: string;
    confidence?: string;
    chunk_id?: string;
    used_in_document?: boolean;
    risk_flags?: string[];
  }>;
  paragraph_validation?: {
    status?: string;
    failed_group_count?: number;
    composer?: ResearchComposerDiagnostics;
    groups?: Array<{
      title?: string;
      status?: string;
      reason?: string;
      used_ids?: string[];
      missing_ids?: string[];
      unknown_ids?: string[];
    }>;
  };
  evidence_metrics?: {
    evidence_count?: number;
    source_count?: number;
    domain_count?: number;
    question_count?: number;
    used_evidence_count?: number;
    risk_flag_count?: number;
    risk_flags?: Record<string, number>;
    question_rows?: Array<{
      question_id?: string;
      question?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      risk_flag_count?: number;
      risk_flags?: string[];
    }>;
    weak_question_count?: number;
    weak_questions?: Array<{
      question_id?: string;
      question?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      risk_flag_count?: number;
      reason?: string;
    }>;
    matrix_cell_count?: number;
    covered_matrix_cell_count?: number;
    weak_matrix_cell_count?: number;
    subject_count?: number;
    axis_count?: number;
    matrix_rows?: Array<{
      matrix_cell_id?: string;
      subject?: string;
      comparison_axis?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      used_evidence_count?: number;
      risk_flag_count?: number;
      risk_flags?: string[];
    }>;
    weak_matrix_cells?: Array<{
      matrix_cell_id?: string;
      subject?: string;
      comparison_axis?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      reason?: string;
    }>;
    composer?: ResearchComposerDiagnostics;
    paragraph_validation_status?: string;
  };
  synthesis_diagnostics?: ResearchSynthesisDiagnostics;
  claim_ledger?: Array<{ text?: string; evidence_ids?: string[]; atomic_claim_ids?: string[]; atomic?: boolean }>;
  section_brief?: ResearchBlueprintSection & Record<string, unknown>;
  visual_specs?: ResearchVisualSpec[];
  revision_diagnostics?: {
    attempted?: boolean;
    accepted?: boolean;
    rejected_reason?: string;
  };
}

export interface ResearchComposerDiagnostics {
  group_count?: number;
  modes?: Record<string, number>;
  fallback_count?: number;
  model_composed_count?: number;
  fallback_reasons?: Array<{
    title?: string;
    mode?: string;
    reason?: string;
  }>;
  elapsed_ms_total?: number;
  elapsed_ms_max?: number;
  slow_groups?: Array<{
    title?: string;
    mode?: string;
    elapsed_ms?: number;
    row_count?: number;
    validation_status?: string;
    validation_reason?: string;
  }>;
}

export interface ResearchSynthesisDiagnostics {
  title?: string;
  elapsed_ms?: number;
  evidence_count?: number;
  source_count?: number;
  group_count?: number;
  finding_count?: number;
  dossier_evidence_count?: number;
  dossier_chunk_count?: number;
  dossier_context_chars?: number;
  unused_evidence_before_write?: number;
  used_evidence_count?: number;
  evidence_utilization_ratio?: number;
  selected_evidence_ids?: string[];
  cropped_evidence_ids?: string[];
  input_finding_count?: number;
  used_finding_count?: number;
  composer?: ResearchComposerDiagnostics;
  groups?: Array<{
    title?: string;
    mode?: string;
    row_count?: number;
    elapsed_ms?: number;
    validation_status?: string;
    validation_reason?: string;
    fallback_reason?: string;
  }>;
}

export interface ResearchMaterialUtilization {
  section_count?: number;
  input_evidence_count?: number;
  used_evidence_count?: number;
  evidence_utilization_ratio?: number;
  sections?: Array<{
    title?: string;
    input_finding_count?: number;
    used_finding_count?: number;
    dossier_evidence_count?: number;
    used_evidence_count?: number;
    evidence_utilization_ratio?: number;
    dossier_chunk_count?: number;
    dossier_context_chars?: number;
    output_chars?: number;
  }>;
}

export interface ResearchEvidenceExtractionDiagnostics {
  mode?: string;
  extraction_mode?: string;
  elapsed_ms?: number;
  section_count?: number;
  selected_chunk_count?: number;
  evidence_count?: number;
  model_evidence_count?: number;
  fallback_evidence_count?: number;
  timeout_count?: number;
  error_count?: number;
  rejected_count?: number;
  sections?: Array<ResearchEvidenceExtractionSection>;
  extraction_sections?: Array<ResearchEvidenceExtractionSection>;
}

export interface ResearchEvidenceExtractionSection {
  section_id?: string;
  title?: string;
  status?: string;
  chunk_count?: number;
  selected_chunk_count?: number;
  evidence_count?: number;
  source_count?: number;
  domain_count?: number;
  model_evidence_count?: number;
  fallback_evidence_count?: number;
  rejected_count?: number;
  timeout_count?: number;
  error_count?: number;
  weak_question_count?: number;
  fallback_used?: boolean;
  elapsed_ms?: number;
  error?: string;
}

export type ResearchStepStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'waiting_approval';

export interface ResearchStep {
  id: string;
  title: string;
  status: ResearchStepStatus;
  summary?: string;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_ms?: number | null;
  items?: Array<{ label?: string; value?: unknown } | string | number | boolean | null>;
}

export interface ResearchSectionReportability {
  status?: 'ready' | 'limited' | 'empty' | string;
  core_section?: boolean;
  evidence_count?: number;
  source_count?: number;
  reason?: string;
}

export interface ResearchBlueprintSection {
  section_id?: string;
  id?: string;
  title?: string;
  reportability?: ResearchSectionReportability;
}

export interface ResearchReportBlueprint extends Record<string, unknown> {
  title?: string;
  sections?: ResearchBlueprintSection[];
}

export interface ResearchJobResult {
  status: string;
  channel_id?: string | null;
  message?: string;
  writing_architecture?: 'finding' | 'evidence_native_eplus' | string;
  writing_architecture_version?: string;
  capabilities?: {
    section_revision?: boolean;
    targeted_followup?: boolean;
    approval?: boolean;
    archive?: boolean;
  };
  evidence_native_blueprint?: ResearchReportBlueprint;
  report_blueprint?: ResearchReportBlueprint;
  evidence_context_diagnostics?: Array<Record<string, unknown>>;
  section_prevalidation?: Record<string, unknown>;
  editorial_claim_bank?: Array<Record<string, unknown>>;
  research_strategy?: ResearchStrategy;
  depth?: ResearchDepth;
  comparison_budget?: Record<string, number>;
  matrix_coverage?: ComparisonMatrixCoverage;
  research_plan?: {
    strategy?: string;
    topic?: string;
    sections?: Array<{
      id?: string;
      title?: string;
      axis?: string;
      questions?: Array<string | {
        id?: string;
        text?: string;
        question?: string;
        purpose?: string;
        success_criteria?: string[];
        preferred_source_types?: string[];
        dependencies?: string[];
      }>;
      success_criteria?: string[];
      reportability?: ResearchSectionReportability;
    }>;
    strategy_profile?: string;
    subjects?: string[];
    comparison_axes?: string[];
    matrix_cells?: Array<{
      id?: string;
      subject?: string;
      axis?: string;
      question?: string;
    }>;
    judgement_questions?: string[];
    vertical?: unknown[];
    horizontal?: unknown[];
    intersections?: string[];
  };
  query_plan?: {
    strategy?: string;
    depth?: string;
    query_limit?: number;
    queries?: Array<{
      section_id?: string;
      section_title?: string;
      question_id?: string;
      question?: string;
      query?: string;
      subject?: string;
      comparison_axis?: string;
      matrix_cell_id?: string;
    }>;
  };
  coverage?: CoverageResult;
  delivery?: {
    type: 'local_markdown';
    paths: string[];
    feishu_doc_url?: string | null;
  };
  quality_gate?: {
    status: 'passed' | 'needs_more_evidence' | string;
    approval_ready: boolean;
    reason: string;
    checks?: Record<string, unknown>;
    matrix_gap_count?: number;
    archive_requires_gap_acknowledgement?: boolean;
    warnings?: string[];
  };
  grounding?: {
    evidence_url_count: number;
    per_document: Array<{
      title: string;
      evidence_ref_count: number;
      matched_evidence_ref_count: number;
      matched_refs?: Array<{ claim: string; source_url: string }>;
    }>;
  };
  documents?: ResearchDocument[];
  sources?: Array<string | { title?: string; url?: string; [key: string]: unknown }>;
  evidence_pack?: Array<{ fact?: string; sources?: number[]; [key: string]: unknown }>;
  materials?: Array<{
    id?: string;
    title?: string;
    url?: string;
    provider?: string;
    source_type?: string;
    content_length?: number;
    content_quality?: string;
    degraded?: boolean;
    excerpt?: string;
    chunk_count?: number;
  }>;
  selected_chunks?: Array<{
    id?: string;
    material_id?: string;
    section_id?: string;
    question_id?: string;
    question?: string;
    subject?: string;
    comparison_axis?: string;
    matrix_cell_id?: string;
    text?: string;
    score?: number;
    source_url?: string;
    source_title?: string;
  }>;
  evidence_map?: Array<{
    id?: string;
    chunk_id?: string;
    section_id?: string;
    question_id?: string;
    subject?: string;
    comparison_axis?: string;
    matrix_cell_id?: string;
    claim?: string;
    atomic_claim?: string;
    evidence_text?: string;
    raw_quote?: string;
    cleaned_quote?: string;
    source_url?: string;
    source_title?: string;
    source_type?: string;
    authority_score?: number;
    evidence_score?: number;
    evidence_type?: string;
    claim_type?: 'direct_fact' | 'design_principle' | 'limitation' | string;
    scope?: 'domain_specific' | 'cross_domain' | 'general' | string;
    directness_score?: number;
    applicability_score?: number;
    publisher_relation?: 'official' | 'third_party' | 'community' | 'unknown' | string;
    stance?: 'support' | 'counter' | 'limitation' | string;
    support_strength?: string;
    grounded_exactly?: boolean;
    quality_flags?: string[];
    confidence?: string;
    used_by_documents?: string[];
  }>;
  validation?: {
    status?: string;
    reason?: string;
    evidence_url_count?: number;
    evidence_claim_count?: number;
    per_document?: Array<{
      title?: string;
      evidence_ref_count?: number;
      matched_evidence_ref_count?: number;
      matched_by_id_count?: number;
      matched_by_fallback_count?: number;
      match_ratio?: number;
      unsupported_refs?: Array<{ evidence_id?: string; chunk_id?: string; claim?: string; source_url?: string }>;
      unsupported_claims?: Array<{ text?: string; reason?: string }>;
      missing_inline_refs?: Array<{ evidence_id?: string; claim?: string; source_url?: string }>;
    }>;
    per_section?: Array<{
      section_id?: string;
      evidence_count?: number;
      source_count?: number;
      domain_count?: number;
      question_count?: number;
      used_ref_count?: number;
    }>;
    entailment?: {
      status?: string;
      claim_count?: number;
      entailed_count?: number;
      inference_count?: number;
      unsupported_count?: number;
      label_mismatch_count?: number;
      unsupported_inference_count?: number;
      judgments?: Array<Record<string, unknown>>;
    };
  };
  followup_actions?: Array<{
    issue_id?: string;
    issue_type?: string;
    evidence_id?: string;
    subject?: string;
    comparison_axis?: string;
    query?: string;
    reason?: string;
    status?: string;
    resolved_at?: string;
  }>;
  revision_actions?: Array<{
    id?: string;
    section_ids?: string[];
    instruction?: string;
    status?: string;
    requested_at?: string;
    resolved_at?: string;
    error?: string;
  }>;
  revision_history?: Array<{
    revision_id?: string;
    section_ids?: string[];
    instruction?: string;
    requested_at?: string;
    resolved_at?: string;
    per_section_outcome?: Record<string, { attempted?: boolean; accepted?: boolean; rejected_reason?: string }>;
  }>;
  initial_validation?: ResearchJobResult['validation'];
  suggested_topic?: string;
  saved_topic?: string;
  paths?: string[];
  artifacts?: string[];
  retrieval?: {
    evidence_diagnostics?: ResearchEvidenceExtractionDiagnostics;
    diagnostics?: {
      section_evidence_extraction?: ResearchEvidenceExtractionDiagnostics;
      global_synthesis?: {
        mode?: string;
        elapsed_ms?: number;
        model_input_chars?: number;
        raw_model_output_chars?: number;
      };
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  pipeline_mode?: 'legacy' | 'shadow' | 'question_loop' | 'single_url' | string;
  /** single_url_brief 结果专属字段。 */
  url?: string;
  source_title?: string;
  brief_markdown?: string;
  citation_count?: number;
  /** 存在即代表这个 job 已经成功存入过知识库（approve_research 写入）。 */
  save_result?: {
    topic?: string;
    paths?: string[];
    document_count?: number;
    titles?: string[];
    [key: string]: unknown;
  };
  question_states?: Array<{
    question_id: string;
    section_id: string;
    section_title: string;
    question: string;
    success_criteria?: string[];
    question_kind?: 'definition' | 'implementation' | 'evaluation' | 'comparison' | 'production_practice' | 'roadmap' | string;
    budget?: {
      max_rounds?: number;
      max_sources?: number;
      max_materials?: number;
      initial_fetch_limit?: number;
      followup_fetch_limit?: number;
      max_fetch_urls?: number;
      timeout_seconds?: number;
      max_model_calls?: number;
    };
    coverage_requirements?: {
      min_claim_clusters?: number;
      min_domains?: number;
      requires_boundary?: boolean;
      required_source_groups?: string[][];
    };
    status: 'pending' | 'researching' | 'sufficient' | 'exhausted' | string;
    stop_reason?: string;
    rounds_used?: number;
    elapsed_seconds?: number;
    attempted_queries?: string[];
    suggested_queries?: string[];
    preferred_source_types?: string[];
    criteria?: Array<{
      criterion_id: string;
      description: string;
      level: 'hard' | 'soft' | 'optional' | string;
      status?: 'uncovered' | 'weak' | 'covered' | string;
      preferred_source_types?: string[];
      attempts?: Array<{ round_index?: number; query?: string; attempt_type?: string }>;
      evidence_ids?: string[];
      coverage_delta?: number;
    }>;
    source_requirements?: Array<{
      source_type: string;
      min_count?: number;
      level?: 'required' | 'preferred' | string;
      fallback_types?: string[];
    }>;
    gap_actions?: Array<{
      action?: string;
      gap_type?: string;
      source_type?: string;
      query?: string;
      blocking?: boolean;
      fallback_used?: boolean;
    }>;
    access_issues?: Array<{
      capability?: string;
      provider?: string;
      status?: string;
      reason?: string;
      url?: string;
    }>;
    attempted_fetch_urls?: string[];
    round_budgets?: Array<Record<string, unknown>>;
    gap_execution_log?: Array<Record<string, unknown>>;
    generated_gap_action_count?: number;
    stagnant_rounds?: number;
    accepted_evidence?: Array<Record<string, unknown>>;
    rejected_evidence?: Array<Record<string, unknown>>;
    coverage?: {
      sufficient?: boolean;
      answer_status?: 'complete' | 'partial' | 'uncovered' | string;
      coverage_status?: 'sufficient' | 'sufficient_with_gaps' | 'insufficient' | 'blocked_by_safety' | string;
      answered_points?: string[];
      missing_question_points?: string[];
      enrichment_gaps?: Array<Record<string, unknown>>;
      evidence_count?: number;
      claim_cluster_count?: number;
      source_count?: number;
      source_mix_satisfied?: boolean;
      source_type_counts?: Record<string, number>;
      weak_aspects?: string[];
      semantic_coverage_counts?: Record<string, number>;
      gaps?: Array<Record<string, unknown>>;
      reason?: string;
    };
  }>;
  findings?: Array<{
    finding_id: string;
    claim_id?: string;
    section_id?: string;
    question_id?: string;
    finding: string;
    evidence_ids: string[];
    counter_evidence_ids?: string[];
    limitations?: string[];
    confidence?: string;
    finding_type?: 'fact' | 'inference' | 'recommendation' | 'limitation' | string;
    support_scope?: 'domain_specific' | 'cross_domain' | 'general' | string;
    direct_evidence_ids?: string[];
    inference_evidence_ids?: string[];
    atomic?: boolean;
    decomposition?: string;
  }>;
  rejected_evidence?: Array<{
    id?: string;
    claim?: string;
    source_url?: string;
    rejection_reason?: string;
    quality_flags?: string[];
  }>;
  report_document?: ResearchDocument & {
    delivery_status?: 'complete_report' | 'partial_report' | string;
    completeness?: ResearchReportCompleteness;
    research_mode?: ResearchMode;
    claim_ledger?: Array<{
      text?: string;
      evidence_ids?: string[];
      atomic_claim_ids?: string[];
      atomic?: boolean;
      section_id?: string;
      claim_type?: string;
      support_scope?: string;
      direct_evidence_ids?: string[];
      inference_evidence_ids?: string[];
    }>;
    visual_specs?: ResearchVisualSpec[];
  };
  report_quality?: {
    status?: string;
    average_score?: number;
    scores?: Record<string, number>;
    blocking_issues?: string[];
    noise_hits?: string[];
    sufficient_question_count?: number;
    question_count?: number;
    rejected_evidence_count?: number;
    report_status?: 'complete_report' | 'partial_report' | string;
    report_completeness?: ResearchReportCompleteness;
  };
  report_status?: 'complete_report' | 'partial_report' | string;
  report_completeness?: ResearchReportCompleteness;
  research_mode?: ResearchMode;
  evidence_landscape?: ResearchEvidenceLandscape;
  pipeline_diagnostics?: {
    mode?: string;
    elapsed_ms?: number;
    question_count?: number;
    sufficient_count?: number;
    exhausted_count?: number;
    answer_status_counts?: {
      complete?: number;
      partial?: number;
      uncovered?: number;
    };
    round_count?: number;
    model_call_count?: number;
    rejected_evidence_count?: number;
    writer?: {
      elapsed_ms?: number;
      section_count?: number;
      parallelism?: number;
      model_mode?: string;
      material_utilization?: ResearchMaterialUtilization;
      [key: string]: unknown;
    };
    report_status?: 'complete_report' | 'partial_report' | string;
    report_completeness?: ResearchReportCompleteness;
    research_mode?: ResearchMode;
    evidence_landscape?: ResearchEvidenceLandscape;
    reportability_rescue?: {
      enabled?: boolean;
      executed?: boolean;
      section_ids?: string[];
      question_count?: number;
      added_rounds?: number;
    };
    targeted_followup_generated_count?: number;
    targeted_followup_executed_count?: number;
    targeted_followup_executed_rate?: number;
    domain_specific_evidence_ratio?: number;
    unsupported_inference_count?: number;
    retrieval_execution?: 'global_source_pool' | string;
    source_ranker_mode?: 'legacy' | 'scoped' | string;
    material_consumer_mode?: 'batch' | 'question' | string;
    source_selection_shadow?: {
      legacy_count?: number;
      scoped_count?: number;
      overlap_count?: number;
      overlap_ratio?: number;
      legacy_domains?: string[];
      scoped_domains?: string[];
      scoped_relationship_counts?: Record<string, number>;
    };
    last_global_fetch_url_count?: number;
    source_pool?: {
      candidate_count?: number;
      discovered_count?: number;
      deduplicated_count?: number;
      fetched_count?: number;
      fetch_count?: number;
      multi_question_candidate_count?: number;
      multi_criterion_candidate_count?: number;
      selection_rounds?: number;
    };
    material_store?: {
      document_count?: number;
      failed_url_count?: number;
      provider_attempt_count?: number;
      content_chars?: number;
    };
    last_shared_wave_elapsed_ms?: number;
    entailment?: Record<string, unknown>;
  };
  shadow_comparison?: Record<string, unknown>;
}

export interface ResearchReportCompleteness {
  status?: 'complete_report' | 'partial_report' | string;
  required_section_count?: number;
  ready_section_count?: number;
  limited_section_ids?: string[];
  core_empty_section_ids?: string[];
  supplemental_empty_section_ids?: string[];
  reason?: string;
}

export type ResearchMode = 'domain_practice_review' | 'hybrid_practice_and_reference' | 'evidence_informed_reference_design' | string;

export interface ResearchEvidenceLandscape {
  mode?: ResearchMode;
  evidence_count?: number;
  source_count?: number;
  domain_specific_count?: number;
  direct_fact_count?: number;
  case_study_count?: number;
  domain_specific_ratio?: number;
}

export interface ResearchVisualSpec {
  visual_id?: string;
  section_id?: string;
  type?: 'table' | 'comparison_matrix' | 'timeline' | 'roadmap' | 'risk_matrix' | 'data_cards' | string;
  title?: string;
  purpose?: string;
  columns?: string[];
  rows?: Array<{ label?: string; value?: string; evidence_ids?: string[] }>;
  render_mode?: 'markdown' | string;
}

export interface ResearchJob extends JobSummary<ResearchJobResult> {
  current_step?: string | null;
  steps?: ResearchStep[];
}

export interface ResearchArtifactsResponse {
  channel_id?: string | null;
  sources: Array<string | { title: string; url: string }>;
  evidence_pack: Array<{ fact: string; sources: number[] }>;
  materials?: ResearchJobResult['materials'];
  selected_chunks?: ResearchJobResult['selected_chunks'];
  evidence_map?: ResearchJobResult['evidence_map'];
  validation?: ResearchJobResult['validation'];
  writing_architecture?: ResearchJobResult['writing_architecture'];
  writing_architecture_version?: string;
  capabilities?: ResearchJobResult['capabilities'];
  evidence_native_blueprint?: ResearchJobResult['evidence_native_blueprint'];
  evidence_context_diagnostics?: ResearchJobResult['evidence_context_diagnostics'];
  section_prevalidation?: ResearchJobResult['section_prevalidation'];
  editorial_claim_bank?: ResearchJobResult['editorial_claim_bank'];
  research_plan?: ResearchJobResult['research_plan'];
  query_plan?: ResearchJobResult['query_plan'];
  coverage: CoverageResult;
  comparison_budget?: ResearchJobResult['comparison_budget'];
  matrix_coverage?: ResearchJobResult['matrix_coverage'];
  documents: ResearchDocument[];
  delivery?: ResearchJobResult['delivery'];
  suggested_topic: string;
}

export interface ResearchApproveRequest {
  topic: string;
  documents?: ResearchDocument[];
  sources?: string[];
  keywords?: string[];
  acknowledge_matrix_gaps?: boolean;
}

export interface ResearchApproveResponse {
  topic: string;
  document_count: number;
  paths: string[];
}

export interface ResearchFollowupRequest {
  issue_id?: string;
  issue_type?: 'unsupported' | 'missing_ref' | 'weak_cell' | 'unused' | 'risk' | string;
  evidence_id?: string;
  subject?: string;
  comparison_axis?: string;
  query?: string;
  reason?: string;
}

export interface ResearchFollowupResponse {
  job_id: string;
  followup_job_id?: string;
  status: string;
  message: string;
  followup_queries: string[];
}

export interface ResearchSectionRevisionRequest {
  section_ids: string[];
  instruction: string;
}

export interface ResearchSectionRevisionResponse {
  job_id: string;
  revision_id: string;
  status: string;
  section_ids: string[];
}
