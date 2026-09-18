import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useJobStore } from '@/stores';
import { ResearchArchivePreviewRequest, ResearchFollowupRequest, ResearchJobCreateRequest, ResearchPlanningClarifyRequest, ResearchPlanningDraftRequest, ResearchSectionRevisionRequest } from '@/types';

const RESEARCH_KEY = 'research';

export function useClarifyResearchPlan() {
  return useMutation({
    mutationFn: (request: ResearchPlanningClarifyRequest) => api.research.clarifyPlan(request),
  });
}

export function useDraftResearchPlan() {
  return useMutation({
    mutationFn: (request: ResearchPlanningDraftRequest) => api.research.draftPlan(request),
  });
}

export function useResearchJob(jobId: string) {
  return useQuery({
    queryKey: [RESEARCH_KEY, jobId],
    queryFn: () => api.research.getJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      const hasRunningFollowup = query.state.data?.result?.followup_actions?.some((action) => action.status === 'running');
      const hasRunningRevision = query.state.data?.result?.revision_actions?.some((action) => action.status === 'running');
      return status === 'queued' || status === 'running' || hasRunningFollowup || hasRunningRevision ? 1500 : false;
    },
  });
}

export function useResearchArtifacts(jobId: string) {
  return useQuery({
    queryKey: [RESEARCH_KEY, jobId, 'artifacts'],
    queryFn: () => api.research.getArtifacts(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.documents?.length ? 'ready' : undefined;
      return status === 'ready' ? false : 2500;
    },
  });
}

export function useCreateResearchJob() {
  const addJob = useJobStore((s) => s.addJob);
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (request: ResearchJobCreateRequest) => api.research.createJob(request),
    onSuccess: (data) => {
      addJob(data.job_id);
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY] });
    },
  });
}

export function useApproveResearch() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      request,
    }: {
      jobId: string;
      request: { topic: string; documents?: { title: string; content: string }[]; sources?: string[]; keywords?: string[] };
    }) => api.research.approve(jobId, request),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-bases'] });
    },
  });
}

export function useResearchArchivePreview() {
  return useMutation({
    mutationFn: ({ jobId, request }: { jobId: string; request: ResearchArchivePreviewRequest }) =>
      api.research.archivePreview(jobId, request),
  });
}

export function useConfirmResearchArchive() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, previewId }: { jobId: string; previewId: string }) =>
      api.research.archiveConfirm(jobId, previewId),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId] });
      queryClient.invalidateQueries({ queryKey: ['knowledge-tree'] });
    },
  });
}

export function useResearchFollowup() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, request }: { jobId: string; request: ResearchFollowupRequest }) =>
      api.research.followup(jobId, request),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId] });
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId, 'artifacts'] });
    },
  });
}

export function useResearchSectionRevision() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, request }: { jobId: string; request: ResearchSectionRevisionRequest }) =>
      api.research.reviseSections(jobId, request),
    onSuccess: (_, { jobId }) => {
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId] });
      queryClient.invalidateQueries({ queryKey: [RESEARCH_KEY, jobId, 'artifacts'] });
    },
  });
}
