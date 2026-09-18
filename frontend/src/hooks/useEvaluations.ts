import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useJobStore } from '@/stores';
import { EvalCase, EvalType, EvaluationResult, JobSummary, ResearchProbeEvaluationJobRequest, SingleUrlBriefEvaluationJobRequest } from '@/types';

const EVAL_KEY = 'evaluations';

export function useEvaluationJob<TResult = EvaluationResult>(jobId: string) {
  return useQuery<JobSummary<TResult>>({
    queryKey: [EVAL_KEY, 'jobs', jobId],
    queryFn: async () => api.evaluations.getJob(jobId) as unknown as JobSummary<TResult>,
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'running' ? 1500 : false;
    },
  });
}

export function useCreateKnowledgeQaCaseGenerationJob() {
  const addJob = useJobStore((s) => s.addJob);
  return useMutation({
    mutationFn: (topic: string) => api.evaluations.createKnowledgeQaCaseGenerationJob(topic),
    onSuccess: (data) => addJob(data.job_id),
  });
}

export function useApproveKnowledgeQaCaseGenerationJob(topic?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.evaluations.approveKnowledgeQaCaseGenerationJob(jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'cases', 'knowledge_qa', topic] }),
  });
}

export function useActivateKnowledgeQaCaseVersion(topic?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (version: number) => api.evaluations.activateKnowledgeQaCaseVersion(topic || '', version),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'cases', 'knowledge_qa', topic] }),
  });
}

export function useCreateEvaluationJob() {
  const queryClient = useQueryClient();
  const addJob = useJobStore((s) => s.addJob);
  return useMutation({
    mutationFn: (request: { topic: string; backend?: string; mode?: string }) =>
      api.evaluations.createJob(request as { topic: string; backend?: 'auto' | 'lightrag' | 'local'; mode?: 'mix' | 'local' | 'global' | 'hybrid' | 'naive' }),
    onSuccess: (data) => {
      addJob(data.job_id);
      queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'reports'] });
    },
  });
}

export function useCreateKnowledgeScopeEvaluationJob() {
  const queryClient = useQueryClient();
  const addJob = useJobStore((state) => state.addJob);
  return useMutation({
    mutationFn: (request: {
      scope_type: 'knowledge_base' | 'theme' | 'research';
      scope_id: string;
      mode?: 'mix' | 'local' | 'global' | 'hybrid' | 'naive';
      include_history?: boolean;
    }) => api.evaluations.createScopeJob(request),
    onSuccess: (data) => {
      addJob(data.job_id);
      queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'reports'] });
    },
  });
}

export function useKnowledgeScopeEvaluationCases(
  scopeType?: 'knowledge_base' | 'theme' | 'research',
  scopeId?: string,
) {
  return useQuery({
    queryKey: [EVAL_KEY, 'scope-cases', scopeType, scopeId],
    queryFn: () => api.evaluations.listScopeCases(scopeType!, scopeId!),
    enabled: Boolean(scopeType && scopeId),
  });
}

export function useUpsertKnowledgeScopeEvaluationCase(
  scopeType?: 'knowledge_base' | 'theme' | 'research',
  scopeId?: string,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (evalCase: EvalCase) => api.evaluations.upsertScopeCase(scopeType!, scopeId!, evalCase),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'scope-cases', scopeType, scopeId] }),
  });
}

export function useDeleteKnowledgeScopeEvaluationCase(
  scopeType?: 'knowledge_base' | 'theme' | 'research',
  scopeId?: string,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (caseId: string) => api.evaluations.deleteScopeCase(scopeType!, scopeId!, caseId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'scope-cases', scopeType, scopeId] }),
  });
}

export function useCreateResearchProbeEvaluationJob() {
  const queryClient = useQueryClient();
  const addJob = useJobStore((s) => s.addJob);
  return useMutation({
    mutationFn: (request: ResearchProbeEvaluationJobRequest) => api.evaluations.createResearchProbeJob(request),
    onSuccess: (data) => {
      addJob(data.job_id);
      queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'research-probe-reports'] });
    },
  });
}

export function useResearchProbeReports(limit = 50, enabled = true) {
  return useQuery({
    queryKey: [EVAL_KEY, 'research-probe-reports', limit],
    queryFn: () => api.evaluations.listResearchProbeReports(limit),
    enabled,
    refetchInterval: 5000,
  });
}

export function useResearchProbeReport(reportId: string) {
  return useQuery({
    queryKey: [EVAL_KEY, 'research-probe-reports', reportId],
    queryFn: () => api.evaluations.getResearchProbeReport(reportId),
    enabled: !!reportId,
  });
}

export function useCreateSingleUrlBriefEvaluationJob() {
  const addJob = useJobStore((s) => s.addJob);
  return useMutation({
    mutationFn: (request: SingleUrlBriefEvaluationJobRequest) => api.evaluations.createSingleUrlBriefJob(request),
    onSuccess: (data) => addJob(data.job_id),
  });
}

export function useEvaluationReports(topic?: string, limit = 50) {
  return useQuery({
    queryKey: [EVAL_KEY, 'reports', topic, limit],
    queryFn: () => api.evaluations.listReports(topic, limit),
  });
}

export function useEvaluationReport(reportId: string) {
  return useQuery({
    queryKey: [EVAL_KEY, 'reports', reportId],
    queryFn: () => api.evaluations.getReport(reportId),
    enabled: !!reportId,
  });
}

export function useEvaluationReportComparison(reportId: string, baselineReportId: string) {
  return useQuery({
    queryKey: [EVAL_KEY, 'reports', reportId, 'compare', baselineReportId],
    queryFn: () => api.evaluations.compareReports(reportId, baselineReportId),
    enabled: !!reportId && !!baselineReportId && reportId !== baselineReportId,
  });
}

export function useEvaluationCases(evalType: EvalType, topic?: string) {
  return useQuery({
    queryKey: [EVAL_KEY, 'cases', evalType, topic],
    queryFn: () => api.evaluations.listCases(evalType, topic),
  });
}

export function useUpsertEvaluationCase(evalType: EvalType, topic?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (evalCase: EvalCase) => api.evaluations.upsertCase(evalType, evalCase, topic),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'cases', evalType, topic] }),
  });
}

export function useDeleteEvaluationCase(evalType: EvalType, topic?: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (caseId: string) => api.evaluations.deleteCase(evalType, caseId, topic),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [EVAL_KEY, 'cases', evalType, topic] }),
  });
}
