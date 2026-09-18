import { ApiAdapter } from '../adapters';
import { JobTraceResponse, ObservabilityEventsResponse, RunDetailResponse, RunListResponse } from '@/types';

export function observabilityEndpoints(adapter: ApiAdapter) {
  return {
    listEvents: (params?: {
      session_id?: string;
      thread_id?: string;
      run_id?: string;
      job_id?: string;
      request_id?: string;
      event_type?: string;
      level?: string;
      limit?: number;
      cursor?: string;
      days?: number;
    }) =>
      adapter.get<ObservabilityEventsResponse>('/api/v1/observability/events', {
        session_id: params?.session_id,
        thread_id: params?.thread_id,
        run_id: params?.run_id,
        job_id: params?.job_id,
        request_id: params?.request_id,
        event_type: params?.event_type,
        level: params?.level,
        limit: params?.limit,
        cursor: params?.cursor,
        days: params?.days,
      }),

    listRuns: (params?: { thread_id?: string; job_id?: string; days?: number; limit?: number }) =>
      adapter.get<RunListResponse>('/api/v1/observability/runs', {
        thread_id: params?.thread_id,
        job_id: params?.job_id,
        days: params?.days,
        limit: params?.limit,
      }),

    getRun: (runId: string, days?: number) =>
      adapter.get<RunDetailResponse>(`/api/v1/observability/runs/${encodeURIComponent(runId)}`, { days }),

    getJobTrace: (jobId: string, days?: number) =>
      adapter.get<JobTraceResponse>(`/api/v1/observability/jobs/${encodeURIComponent(jobId)}/trace`, { days }),

    streamUrl: (params?: { sessionId?: string; runId?: string; jobId?: string; replay?: number }) => {
      const base = '/api/v1/observability/stream';
      const search = new URLSearchParams();
      if (params?.sessionId) search.set('session_id', params.sessionId);
      if (params?.runId) search.set('run_id', params.runId);
      if (params?.jobId) search.set('job_id', params.jobId);
      search.set('replay', String(params?.replay ?? 20));
      return `${base}?${search.toString()}`;
    },
  };
}
