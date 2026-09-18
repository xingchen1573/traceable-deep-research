import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useObservabilityEvents(params?: {
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
}) {
  return useQuery({
    queryKey: ['observability', 'events', params],
    queryFn: () => api.observability.listEvents(params),
  });
}

export function useObservabilityStream(params?: {
  sessionId?: string;
  runId?: string;
  jobId?: string;
  replay?: number;
}) {
  return { url: api.observability.streamUrl(params) };
}

export function useObservabilityRuns(params?: { thread_id?: string; job_id?: string; days?: number; limit?: number }) {
  return useQuery({
    queryKey: ['observability', 'runs', params],
    queryFn: () => api.observability.listRuns(params),
  });
}

export function useObservabilityRun(runId: string, days?: number) {
  return useQuery({
    queryKey: ['observability', 'run', runId, days],
    queryFn: () => api.observability.getRun(runId, days),
    enabled: !!runId,
    refetchInterval: 2_000,
  });
}

export function useJobTrace(jobId: string, days?: number) {
  return useQuery({
    queryKey: ['observability', 'job-trace', jobId, days],
    queryFn: () => api.observability.getJobTrace(jobId, days),
    enabled: !!jobId,
    // Job traces may receive approval or archival events after the worker has
    // reached a terminal state, so keep the open detail view synchronized.
    refetchInterval: 2_000,
  });
}
