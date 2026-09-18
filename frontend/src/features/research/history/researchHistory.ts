import { JobStatus, ResearchJob } from '@/types';

export type ResearchHistoryType = 'deep_research' | 'single_url';

export interface ResearchHistoryDisplay {
  jobId: string;
  title: string;
  type: ResearchHistoryType;
  typeLabel: string;
  status: JobStatus;
  updatedAt: string;
  createdAt: string;
  archived: boolean;
  qualityScore?: number;
  docCount?: number;
  sourceCount?: number;
}

/** Both topic research and link briefs share job_type="research"; only
 * `result.pipeline_mode` distinguishes them (see docs/modules/research-studio-history-v0.1.md §13). */
export function deriveHistoryDisplay(job: ResearchJob): ResearchHistoryDisplay {
  const result = job.result;
  const isSingleUrl = result?.pipeline_mode === 'single_url';
  const type: ResearchHistoryType = isSingleUrl ? 'single_url' : 'deep_research';
  const inputUrl = typeof job.input?.url === 'string' ? (job.input.url as string) : undefined;
  const inputRequest = typeof job.input?.user_request === 'string' ? (job.input.user_request as string) : undefined;
  const title = isSingleUrl
    ? result?.source_title || result?.url || inputUrl || 'URL 简报'
    : result?.suggested_topic || result?.save_result?.topic || inputRequest || '调研任务';

  return {
    jobId: job.job_id,
    title,
    type,
    typeLabel: isSingleUrl ? '链接简报' : '主题调研',
    status: job.status,
    updatedAt: job.updated_at,
    createdAt: job.created_at,
    archived: Boolean(job.archived),
    qualityScore: result?.report_quality?.average_score,
    docCount: result?.documents?.length ?? result?.coverage?.document_count,
    sourceCount: result?.coverage?.source_count ?? result?.sources?.length,
  };
}

export function matchesSearch(job: ResearchJob, term: string): boolean {
  const needle = term.trim().toLowerCase();
  if (!needle) return true;
  const display = deriveHistoryDisplay(job);
  const haystack = [display.title, job.result?.url, job.job_id]
    .filter((value): value is string => Boolean(value))
    .map((value) => value.toLowerCase());
  return haystack.some((value) => value.includes(needle));
}

export interface ResearchHistoryFilters {
  search?: string;
  type?: ResearchHistoryType | 'all';
  status?: JobStatus | 'all';
}

export function getResearchStudioSearchParams(
  current: URLSearchParams,
  nextTab: 'create' | 'history' | 'detail',
): URLSearchParams {
  const next = new URLSearchParams(current);
  if (nextTab !== 'detail') next.delete('job_id');
  return next;
}

/** Filters only — relies on the caller's list already being sorted (backend/mock both sort
 * waiting_approval-first then updated_at desc); Array.filter preserves that relative order. */
export function filterAndSortHistory(jobs: ResearchJob[], filters: ResearchHistoryFilters): ResearchJob[] {
  const { search = '', type = 'all', status = 'all' } = filters;
  return jobs.filter((job) => {
    if (search && !matchesSearch(job, search)) return false;
    if (type !== 'all' && deriveHistoryDisplay(job).type !== type) return false;
    if (status !== 'all' && job.status !== status) return false;
    return true;
  });
}

export const HISTORY_TYPE_OPTIONS: Array<{ value: ResearchHistoryType | 'all'; label: string }> = [
  { value: 'all', label: '全部' },
  { value: 'deep_research', label: '主题调研' },
  { value: 'single_url', label: '链接简报' },
];

export const HISTORY_STATUS_OPTIONS: Array<{ value: JobStatus | 'all'; label: string }> = [
  { value: 'all', label: '全部' },
  { value: 'running', label: '运行中' },
  { value: 'waiting_approval', label: '待审批' },
  { value: 'succeeded', label: '已完成' },
  { value: 'failed', label: '失败' },
  { value: 'cancelled', label: '已取消' },
  { value: 'interrupted', label: '已中断' },
];
