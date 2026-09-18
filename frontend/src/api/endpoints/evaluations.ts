import { ApiAdapter } from '../adapters';
import {
  EvalCase,
  EvalCaseListResponse,
  EvalType,
  EvaluationJob,
  EvaluationJobRequest,
  EvaluationReportListResponse,
  EvaluationReportResponse,
  JobCreatedResponse,
  ResearchProbeEvaluationJobRequest,
  ResearchProbeReportListResponse,
  ResearchProbeReportResponse,
  SingleUrlBriefEvaluationJobRequest,
} from '@/types';

export function evaluationEndpoints(adapter: ApiAdapter) {
  return {
    createJob: (request: EvaluationJobRequest) =>
      adapter.post<JobCreatedResponse>('/api/v1/evaluations/knowledge-qa/jobs', request),

    createScopeJob: (request: {
      scope_type: 'knowledge_base' | 'theme' | 'research';
      scope_id: string;
      mode?: 'mix' | 'local' | 'global' | 'hybrid' | 'naive';
      include_history?: boolean;
    }) => adapter.post<JobCreatedResponse>('/api/v1/evaluations/knowledge-scopes/jobs', request),

    listScopeCases: (scopeType: 'knowledge_base' | 'theme' | 'research', scopeId: string) =>
      adapter.get<{ scope_key: string; has_scope_cases: boolean; cases: EvalCase[] }>(
        `/api/v1/evaluations/knowledge-scopes/${scopeType}/${encodeURIComponent(scopeId)}/cases`,
      ).then((response) => ({
        eval_type: 'knowledge_qa' as const,
        topic: response.scope_key,
        scopes: [],
        scope: response.has_scope_cases ? 'topic' as const : 'default' as const,
        has_topic_cases: response.has_scope_cases,
        cases: response.cases,
      })),

    upsertScopeCase: (
      scopeType: 'knowledge_base' | 'theme' | 'research',
      scopeId: string,
      evalCase: EvalCase,
    ) => adapter.post<EvalCase>(
      `/api/v1/evaluations/knowledge-scopes/${scopeType}/${encodeURIComponent(scopeId)}/cases`,
      { case: evalCase },
    ),

    deleteScopeCase: (
      scopeType: 'knowledge_base' | 'theme' | 'research',
      scopeId: string,
      caseId: string,
    ) => adapter.delete<{ deleted: string }>(
      `/api/v1/evaluations/knowledge-scopes/${scopeType}/${encodeURIComponent(scopeId)}/cases/${encodeURIComponent(caseId)}`,
    ),

    createKnowledgeQaCaseGenerationJob: (topic: string) =>
      adapter.post<JobCreatedResponse>('/api/v1/evaluations/knowledge-qa/case-generation/jobs', { topic }),

    approveKnowledgeQaCaseGenerationJob: (jobId: string) =>
      adapter.post<import('@/types').KnowledgeQaCaseGenerationResult>(
        `/api/v1/evaluations/knowledge-qa/case-generation/jobs/${encodeURIComponent(jobId)}/approve`,
        {}
      ),

    activateKnowledgeQaCaseVersion: (topic: string, version: number) =>
      adapter.post<{ topic: string; version: number; active: boolean; is_stale: boolean; stale_reason?: string }>(
        '/api/v1/evaluations/knowledge-qa/case-versions/activate',
        { topic, version }
      ),

    createResearchProbeJob: (request: ResearchProbeEvaluationJobRequest) =>
      adapter.post<JobCreatedResponse>('/api/v1/evaluations/research-probe/jobs', request),

    listResearchProbeReports: (limit = 50) =>
      adapter.get<ResearchProbeReportListResponse>('/api/v1/evaluations/research-probe/reports', { limit }),

    getResearchProbeReport: (reportId: string) =>
      adapter.get<ResearchProbeReportResponse>(
        `/api/v1/evaluations/research-probe/reports/${encodeURIComponent(reportId)}`
      ),

    createSingleUrlBriefJob: (request: SingleUrlBriefEvaluationJobRequest) =>
      adapter.post<JobCreatedResponse>('/api/v1/evaluations/single-url-brief/jobs', request),

    getJob: (jobId: string) => adapter.get<EvaluationJob>(`/api/v1/evaluations/jobs/${encodeURIComponent(jobId)}`),

    listReports: (topic?: string, limit = 50) =>
      adapter.get<EvaluationReportListResponse>('/api/v1/evaluations/reports', { topic, limit }),

    getReport: (reportId: string) =>
      adapter.get<EvaluationReportResponse>(
        `/api/v1/evaluations/reports/${encodeURIComponent(reportId)}`
      ),

    compareReports: (reportId: string, baselineReportId: string) =>
      adapter.get<import('@/types').EvaluationReportComparison>(
        `/api/v1/evaluations/reports/${encodeURIComponent(reportId)}/compare`,
        { baseline_report_id: baselineReportId }
      ),

    listCases: (evalType: EvalType, topic?: string) =>
      adapter.get<EvalCaseListResponse>('/api/v1/evaluations/cases', { eval_type: evalType, topic }),

    upsertCase: (evalType: EvalType, evalCase: EvalCase, topic?: string) =>
      adapter.post<EvalCase>('/api/v1/evaluations/cases', { eval_type: evalType, topic, case: evalCase }),

    deleteCase: (evalType: EvalType, caseId: string, topic?: string) =>
      adapter.delete<{ deleted: string }>(
        `/api/v1/evaluations/cases/${encodeURIComponent(caseId)}?eval_type=${encodeURIComponent(evalType)}${
          topic ? `&topic=${encodeURIComponent(topic)}` : ''
        }`
      ),
  };
}
