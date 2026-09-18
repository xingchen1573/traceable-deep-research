import { ApiAdapter } from '../adapters';
import { JobListParams, JobListResponse, JobStats, JobSummary } from '@/types';

export function jobEndpoints(adapter: ApiAdapter) {
  return {
    list: (params: JobListParams = {}) =>
      adapter.get<JobListResponse>('/api/v1/jobs', {
        job_type: params.jobType,
        research_type: params.researchType,
        status: params.status,
        statuses: params.statuses?.join(','),
        limit: params.limit ?? 50,
        cursor: params.cursor,
        offset: params.offset,
        query: params.query,
        channel_id: params.channelId,
        archived: params.archived,
      }),

    stats: (archived: boolean | undefined = false) =>
      adapter.get<JobStats>('/api/v1/jobs/stats', { archived }),

    get: (jobId: string) => adapter.get<JobSummary>(`/api/v1/jobs/${encodeURIComponent(jobId)}`),

    cancel: (jobId: string, reason = '用户取消') => adapter.post<{ job_id: string; status: string }>(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { reason }),

    archive: (jobId: string, archived: boolean) =>
      adapter.post<{ job_id: string; archived: boolean }>(`/api/v1/jobs/${encodeURIComponent(jobId)}/archive`, { archived }),
  };
}
