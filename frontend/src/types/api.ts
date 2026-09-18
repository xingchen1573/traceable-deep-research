export type ApiStatus = 'ok' | 'accepted' | 'error';

export interface ApiEnvelope<T = unknown> {
  status: ApiStatus;
  request_id: string;
  data?: T;
  meta?: Record<string, unknown>;
  error?: {
    code: string;
    message: string;
    retryable: boolean;
    details?: Record<string, unknown>;
  };
}

export type JobStatus =
  | 'queued'
  | 'running'
  | 'waiting_approval'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

export interface JobSummary<TResult = Record<string, unknown>> {
  job_id: string;
  job_type: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  progress?: {
    current: number;
    total: number;
    message: string;
  };
  current_step?: string | null;
  steps?: Array<{
    id: string;
    title: string;
    status: string;
    summary?: string;
    started_at?: string | null;
    finished_at?: string | null;
    items?: unknown[];
  }>;
  input?: Record<string, unknown>;
  result?: TResult | null;
  error?: {
    code: string;
    message: string;
    retryable: boolean;
    details?: Record<string, unknown>;
  } | null;
  artifacts?: string[];
  archived?: boolean;
  owner_instance_id?: string | null;
  status_reason?: string | null;
  started_at?: string | null;
  last_heartbeat_at?: string | null;
  last_progress_at?: string | null;
  finished_at?: string | null;
  health_status?: 'healthy' | 'stalled' | 'orphaned' | null;
  environment?: string;
  attempt?: number;
  retry_of_job_id?: string | null;
}

export interface JobCreatedResponse {
  job_id: string;
  job_type: string;
  status: JobStatus;
}

export interface JobListParams {
  jobType?: string;
  researchType?: 'deep_research' | 'single_url';
  status?: JobStatus;
  statuses?: JobStatus[];
  archived?: boolean;
  limit?: number;
  cursor?: string;
  offset?: number;
  query?: string;
  channelId?: string;
}

export interface JobListResponse {
  jobs: JobSummary[];
  next_cursor?: string | null;
  total: number;
}

export interface JobStats {
  total: number;
  attention: number;
  active: number;
  failed: number;
  by_status: Partial<Record<JobStatus, number>>;
  by_type: Record<string, number>;
}

export interface HealthResponse {
  service: string;
  version: string;
  time: string;
}

export interface CapabilitiesResponse {
  chat: boolean;
  research_jobs: boolean;
  knowledge_base_management: boolean;
  knowledge_qa: boolean;
  lightrag_index: boolean;
  knowledge_evaluation: boolean;
  observability: boolean;
  model_check: boolean;
}

export interface RuntimeConfigResponse {
  model_name: string;
  base_url_configured: boolean;
  embedding_model: string;
  embedding_dim: number;
  lightrag_enabled: boolean;
  knowledge_base_root: string;
  log_enabled: boolean;
}

export interface ObservabilityEvent {
  timestamp: string;
  session_id: string;
  run_id?: string;
  event_type: string;
  level: string;
  component: string;
  message: string;
  thread_id?: string;
  job_id?: string;
  request_id?: string;
  report_id?: string;
  data: Record<string, unknown>;
}

export interface ObservabilityEventsResponse {
  events: ObservabilityEvent[];
  next_cursor?: string | null;
  has_more: boolean;
  limit: number;
  scanned: number;
  days: number;
}

export type RunStatus = 'completed' | 'error' | 'running' | 'waiting_user' | 'interrupted' | 'unknown';

export interface RunSummary {
  id: string;
  title: string;
  status: RunStatus;
  session_id?: string | null;
  thread_id?: string | null;
  job_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  duration_ms?: number | null;
  event_count: number;
  error_count: number;
  components: string[];
}

export interface RunListResponse {
  runs: RunSummary[];
  days: number;
  total_runs: number;
  truncated: boolean;
}

export interface RunDetailResponse extends RunSummary {
  run_id: string;
  events: ObservabilityEvent[];
  errors: ObservabilityEvent[];
  tool_calls: ObservabilityEvent[];
  artifacts: string[];
}

export interface JobTraceResponse extends RunSummary {
  job_id: string;
  run_ids: string[];
  events: ObservabilityEvent[];
  errors: ObservabilityEvent[];
  job_type?: string | null;
  job_status?: string | null;
}

export interface ModelConfig {
  model: string;
  base_url_configured: boolean;
  api_key_configured: boolean;
}

export interface ResearchRoleModelConfig {
  model: string;
  base_url_configured: boolean;
  base_url: string | null;
  base_url_source: ResearchKeySource | null;
  key_count: number;
  keys: ResearchRoleKey[];
  inherits_keys_from: 'research_evidence_model' | null;
}

export type ResearchRole = 'evidence' | 'writer';

export type ResearchKeySource =
  | 'dedicated'
  | 'inherited_evidence'
  | 'inherited_chat'
  | 'legacy_shared';

export interface ResearchRoleKey {
  id: string;
  masked: string;
  source: ResearchKeySource;
  deletable: boolean;
  shared: boolean;
}

export interface ModelConfigResponse {
  chat_model: ModelConfig;
  recall_model: ModelConfig;
  embedding: ModelConfig & { dim: number };
  dream_model: ModelConfig;
  research_evidence_model: ResearchRoleModelConfig;
  research_writer_model: ResearchRoleModelConfig;
}

export interface ModelCheckKeyResult {
  key_index?: number;
  key_id: string;
  masked: string;
  status: string;
  latency_ms: number;
  message: string;
}

export interface ModelCheckResult {
  target: string;
  status: string;
  latency_ms: number;
  message: string;
  keys?: ModelCheckKeyResult[];
}

export interface ModelCheckResponse {
  results: ModelCheckResult[];
}

export interface ModelConfigUpdateRequest {
  chat_model: {
    model: string;
    base_url?: string;
    api_key?: string;
  };
  recall_model: {
    follow_chat_model: boolean;
    model?: string;
    base_url?: string;
    api_key?: string;
  };
  embedding: {
    model: string;
    base_url?: string;
    api_key?: string;
    dim: number;
  };
  dream_model?: {
    model: string;
    base_url?: string;
    api_key?: string;
  };
  research_evidence_model?: {
    model: string;
    base_url?: string;
    api_keys?: string[];
  };
  research_writer_model?: {
    model: string;
    base_url?: string;
    api_keys?: string[];
  };
}

export interface ResearchRoleKeyMutationResponse {
  status: 'added' | 'deleted';
  key?: ResearchRoleKey;
  deleted_key_id?: string;
  config: ResearchRoleModelConfig;
}

export interface ResearchRoleKeyCheckRequest {
  api_key?: string;
  model?: string;
  base_url?: string;
  timeout_seconds?: number;
}

export interface ModelConfigUpdateResponse {
  status: 'saved';
  restart_required: boolean;
  updated_fields: string[];
  config_path: string;
  message: string;
}

export interface ProvidersResponse {
  providers: Array<{
    provider: string;
    enabled: boolean;
  }>;
}
