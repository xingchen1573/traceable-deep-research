import { useInfiniteQuery, useQuery, useMutation, useQueryClient, UseQueryResult } from '@tanstack/react-query';
import { api } from '@/api/client';
import { JobListParams, JobListResponse } from '@/types';
import { getJobPollingInterval } from './jobPolling';

const JOB_KEY = 'jobs';

export function useJobs(filters?: JobListParams): UseQueryResult<JobListResponse>;
export function useJobs(jobType?: string, status?: string, archived?: boolean, limit?: number): UseQueryResult<JobListResponse>;
export function useJobs(
  filtersOrJobType?: JobListParams | string,
  status?: string,
  archived?: boolean,
  limit?: number
) {
  const filters: JobListParams = typeof filtersOrJobType === 'object'
    ? filtersOrJobType
    : {
        jobType: filtersOrJobType,
        status: status as JobListParams['status'],
        archived,
        limit,
      };
  return useQuery({
    queryKey: [JOB_KEY, 'list', filters],
    queryFn: () => api.jobs.list(filters),
    refetchInterval: (query) => query.state.data?.jobs.some((job) => ['queued', 'running'].includes(job.status)) ? 3000 : 30000,
  });
}

export function useJobStats(archived: boolean | undefined = false) {
  return useQuery({
    queryKey: [JOB_KEY, 'stats', archived],
    queryFn: () => api.jobs.stats(archived),
    refetchInterval: (query) => query.state.data?.active ? 3000 : 10000,
  });
}

export function useInfiniteJobs(filters: JobListParams) {
  return useInfiniteQuery({
    queryKey: [JOB_KEY, 'infinite', filters],
    queryFn: ({ pageParam }) => api.jobs.list({ ...filters, cursor: pageParam }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor || undefined,
    refetchInterval: (query) => query.state.data?.pages.some((page) => page.jobs.some((job) => ['queued', 'running'].includes(job.status))) ? 3000 : false,
  });
}

export function useJob(jobId: string) {
  return useQuery({
    queryKey: [JOB_KEY, jobId],
    queryFn: () => api.jobs.get(jobId),
    enabled: !!jobId,
  });
}

export function useJobPolling(jobId: string) {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: [JOB_KEY, jobId],
    queryFn: async () => {
      const job = await api.jobs.get(jobId);
      const terminal = ['succeeded', 'failed', 'cancelled', 'interrupted'];
      if (terminal.includes(job.status)) {
        queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] });
        queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'list'] });
        queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'stats'] });
      }
      return job;
    },
    enabled: !!jobId,
    retry: 1,
    refetchInterval: (query) => {
      // Historical chat messages can reference jobs that have since been deleted.
      // Do not keep retrying those missing records forever: every retry used to toggle
      // an inline loading placeholder and continuously change the chat's scroll height.
      return getJobPollingInterval(query.state.data?.status);
    },
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, reason }: { jobId: string; reason?: string }) =>
      api.jobs.cancel(jobId, reason),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, jobId] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'list'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'stats'] });
    },
  });
}

export function useArchiveJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, archived }: { jobId: string; archived: boolean }) =>
      api.jobs.archive(jobId, archived),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'list'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'infinite'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'stats'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, jobId] });
    },
  });
}

export function useBulkArchiveJobs() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobIds, archived }: { jobIds: string[]; archived: boolean }) =>
      Promise.all(jobIds.map((jobId) => api.jobs.archive(jobId, archived))),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'list'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'infinite'] });
      queryClient.invalidateQueries({ queryKey: [JOB_KEY, 'stats'] });
    },
  });
}
