import { JobSummary } from './api';
import { QueryBackend, QueryMode } from './knowledge-base';

export interface EvaluationSummary {
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  average_score: number;
  failed_checks: Record<string, number>;
  categories: Record<string, number>;
  failure_types: Record<string, number>;
  backend_usage?: Record<string, number>;
  fallback_count?: number;
  fallback_reasons?: Record<string, number>;
  retrieval_quality?: {
    status: 'completed' | 'partial' | 'unavailable' | 'not_applicable';
    evaluated_cases: number;
    unavailable_cases: number;
    semantic_unavailable_cases: number;
    needs_review_cases: number;
    advisory_cases?: number;
    metrics: RetrievalMetrics;
  };
  semantic_quality?: {
    status: 'completed' | 'partial' | 'unavailable' | 'not_applicable';
    judged_cases: number;
    unavailable_cases: number;
    needs_review_cases: number;
    scores: SemanticScores;
  };
}

export interface RetrievalMetrics {
  evidence_recall?: number | null;
  point_coverage?: number | null;
  hit_at_3?: number | boolean | null;
  first_key_evidence_rank?: number | null;
  reciprocal_rank?: number | null;
  context_relevance?: number | null;
  noise_ratio?: number | null;
  source_concentration?: number | null;
  duplicate_ratio?: number | null;
  boundary_safety?: number | null;
}

export interface RetrievalChunkEvaluation {
  rank: number;
  chunk_id?: string;
  path?: string;
  content?: string;
  matched_evidence_ids?: string[];
  evidence_match_score?: number;
  relevance_score?: number;
  relevance_label?: 'supporting' | 'related' | 'noise' | 'misleading';
  judge_reason?: string;
}

export interface RetrievalEvaluation {
  status: 'completed' | 'partial' | 'unavailable' | 'not_applicable';
  quality_status: 'passed' | 'needs_review' | 'not_evaluated';
  semantic_judge_status?: 'completed' | 'unavailable' | 'pending' | 'not_started';
  metrics?: RetrievalMetrics;
  matched_evidence_ids?: string[];
  matched_point_ids?: string[];
  missing_point_ids?: string[];
  chunks?: RetrievalChunkEvaluation[];
  reason?: string;
  below_threshold?: string[];
  advisory?: string[];
  recommended_action?: string;
}

export interface SemanticScores {
  correctness: number;
  completeness: number;
  faithfulness: number;
  relevance: number;
  boundary_awareness: number;
  citation_support: number;
}

export interface SemanticEvaluation {
  status: 'completed' | 'unavailable' | 'not_applicable';
  quality_status: 'passed' | 'needs_review' | 'not_evaluated';
  scores?: SemanticScores;
  matched_point_ids?: string[];
  missing_point_ids?: string[];
  unsupported_claims?: string[];
  citation_issues?: string[];
  reason?: string;
  below_threshold?: string[];
  recommended_action?: string;
}

export interface EvaluationResult {
  topic: string;
  summary: EvaluationSummary;
  report_path: string;
  results?: EvaluationCase[];
}

export interface EvaluationCase {
  id: string;
  category: string;
  passed: boolean;
  score: number;
  failed_checks: string[];
  likely_failure_type: string;
  recommended_action: string;
  backend: QueryBackend;
  requested_backend?: QueryBackend;
  actual_backend?: QueryBackend | 'mixed';
  fallback_occurred?: boolean;
  fallback_reason?: string;
  mode: QueryMode;
  citation_count: number;
  question?: string;
  answer?: string;
  retrieval_evaluation?: RetrievalEvaluation;
  semantic_evaluation?: SemanticEvaluation;
}

export interface EvaluationJob extends JobSummary<EvaluationResult> {}

export interface EvaluationJobRequest {
  topic: string;
  backend?: QueryBackend;
  mode?: QueryMode;
}

export interface EvaluationReportMeta {
  report_id: string;
  topic: string;
  path: string;
  created_at: string;
  run_id?: string;
  backend?: QueryBackend | null;
  mode?: QueryMode | null;
  eval_set_id?: string | null;
  eval_set_version?: number | null;
  knowledge_content_hash?: string | null;
  knowledge_content_stale?: boolean;
  case_set_modified?: boolean;
  scope?: { type: 'knowledge_base' | 'theme' | 'research'; id: string; title?: string; path?: string } | null;
  summary: Partial<EvaluationSummary> & { pass_rate: number };
}

export interface EvaluationReportListResponse {
  reports: EvaluationReportMeta[];
}

export interface EvaluationReportResponse {
  report_id: string;
  topic: string;
  content: string;
  created_at?: string;
  run_id?: string;
  job_id?: string;
  summary?: EvaluationSummary;
  cases?: EvaluationCase[];
  recommended_actions?: string[];
  backend?: QueryBackend;
  mode?: QueryMode;
  eval_set_id?: string | null;
  eval_set_version?: number | null;
  case_set_hash?: string | null;
  eval_set_case_hash?: string | null;
  case_set_modified?: boolean;
  knowledge_content_hash?: string | null;
  knowledge_content_stale?: boolean;
  scope?: { type: 'knowledge_base' | 'theme' | 'research'; id: string; title?: string; path?: string } | null;
}

export interface EvaluationReportComparisonMetric {
  key: string;
  label: string;
  current: number | null;
  baseline: number | null;
  delta: number | null;
}

export interface EvaluationReportComparison {
  topic: string;
  current_report_id: string;
  baseline_report_id: string;
  current_created_at?: string;
  baseline_created_at?: string;
  configuration_changed: boolean;
  current_configuration: Record<string, unknown>;
  baseline_configuration: Record<string, unknown>;
  metrics: EvaluationReportComparisonMetric[];
}

export type EvalType = 'knowledge_qa' | 'research_probe' | 'single_url_brief';

export interface EvalCase extends Record<string, unknown> {
  id: string;
  case_version?: number;
  kind?: string;
  title?: string;
  url?: string;
  user_request?: string;
  strategy?: 'standard' | 'hv_analysis';
  depth?: 'normal' | 'deep';
  core_intents?: string[];
  optional_topics?: string[];
  scope_notes?: string;
  question?: string;
  category?: string;
  answerability?: 'answerable' | 'unanswerable';
  reference_answer?: string;
  evidence_refs?: Array<Record<string, unknown>>;
  eval_set_id?: string;
  eval_set_version?: number;
}

export interface EvalCaseListResponse {
  eval_type: EvalType;
  topic?: string | null;
  scopes: string[];
  scope: 'topic' | 'default';
  has_topic_cases: boolean;
  eval_set_id?: string | null;
  eval_set_version?: number | null;
  versions?: KnowledgeQaEvalSetVersion[];
  knowledge_content_hash?: string | null;
  eval_set_content_hash?: string | null;
  is_stale?: boolean;
  case_set_modified?: boolean;
  stale_reason?: string;
  cases: EvalCase[];
}

export interface KnowledgeQaEvalSetVersion {
  eval_set_id: string;
  version: number;
  case_set_hash: string;
  knowledge_content_hash: string;
  approved_at: string;
  case_count: number;
  active: boolean;
}

export interface KnowledgeQaCaseGenerationResult {
  status: 'draft_ready';
  approval_status: 'pending' | 'approved';
  topic: string;
  topic_title: string;
  eval_set_id: string;
  eval_set_version: number;
  target_case_count: number;
  case_count: number;
  answerable_count: number;
  unanswerable_count: number;
  document_count: number;
  evidence_count: number;
  previous_case_count?: number;
  active_case_count?: number;
  cases: EvalCase[];
  rejected_candidates: Array<Record<string, unknown>>;
  generated_at: string;
  approved_at?: string;
}

export interface KnowledgeQaCaseGenerationJob extends JobSummary<KnowledgeQaCaseGenerationResult> {}

export interface ResearchProbeEvaluationJobRequest {
  case_ids?: string[] | null;
  strategy?: string;
  depth?: string;
  case_interval_seconds?: number;
}

export interface ResearchContentReviewDimension {
  label: '良好' | '一般' | '需改进' | '未评估';
  reason: string;
  section_ids?: string[];
}

export interface ResearchContentReview {
  status: 'completed' | 'completed_with_advice' | 'content_review_unavailable' | 'not_run';
  blocking: false;
  dimensions?: Record<string, ResearchContentReviewDimension>;
  suggestions?: Array<{
    section_id?: string | null;
    severity: 'low' | 'medium' | 'high';
    reason: string;
  }>;
  confidence?: number | null;
  reason?: string;
  judge?: { status: string; model?: string; provider?: string };
}

export interface ResearchProbeEvaluationRow {
  case_id: string;
  kind: string;
  title: string;
  failed: boolean;
  outcome: 'completed' | 'pipeline_gate_failed' | 'execution_failed' | 'cancelled';
  execution_status: 'succeeded' | 'failed' | 'cancelled';
  production_gate_status: 'passed' | 'failed' | 'not_evaluated';
  job_id?: string | null;
  elapsed_seconds?: number | null;
  quality_gate_status?: string | null;
  approval_ready?: boolean;
  validation_status?: string | null;
  sources?: number;
  materials?: number;
  selected_chunks?: number;
  evidence_map?: number;
  documents?: number;
  hybrid_scored_chunks?: number;
  weak_question_count?: number;
  risk_single_source_count?: number;
  unsupported_claims?: number;
  blocking_unsupported_claims?: number;
  missing_inline_refs?: number;
  quality_reason?: string;
  validation_reason?: string;
  content_review?: ResearchContentReview;
}

export interface ResearchProbeEvaluationSummary {
  total_cases: number;
  execution_succeeded_cases: number;
  execution_failed_cases: number;
  cancelled_cases: number;
  production_gate_passed_cases: number;
  production_gate_failed_cases: number;
  production_gate_not_evaluated_cases: number;
  production_gate_pass_rate: number;
  elapsed_seconds: number;
  avg_case_seconds: number;
  content_review_completed_cases: number;
  content_review_advisory_cases: number;
  content_review_unavailable_cases: number;
  weak_question_total?: number;
  hybrid_scored_total?: number;
}

export interface ResearchProbeReportMeta {
  report_id: string;
  evaluation_type: 'research_probe';
  path: string;
  created_at: string;
  started_at?: string;
  schema_version?: number;
  configuration?: Record<string, unknown>;
  summary: Partial<ResearchProbeEvaluationSummary>;
}

export interface ResearchProbeReportListResponse {
  reports: ResearchProbeReportMeta[];
}

export interface ResearchProbeReportResponse extends ResearchProbeReportMeta {
  content: string;
  finished_at?: string;
  rows: ResearchProbeEvaluationRow[];
  results?: Array<Record<string, unknown>>;
  child_job_ids?: string[];
}

export interface SingleUrlBriefEvaluationJobRequest {
  case_ids?: string[] | null;
}
