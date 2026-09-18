import { ApiAdapter } from '../adapters';
import {
  JobCreatedResponse,
  ResearchApproveRequest,
  ResearchApproveResponse,
  ResearchArtifactsResponse,
  ResearchFollowupRequest,
  ResearchFollowupResponse,
  ResearchJob,
  ResearchJobCreateRequest,
  ResearchSectionRevisionRequest,
  ResearchSectionRevisionResponse,
  ResearchArchivePreviewRequest,
  ResearchPlanningClarifyRequest,
  ResearchPlanningClarifyResponse,
  ResearchPlanningDraftRequest,
  ResearchPlanningDraftResponse,
  KnowledgePreview,
} from '@/types';

export function researchEndpoints(adapter: ApiAdapter) {
  return {
    clarifyPlan: (request: ResearchPlanningClarifyRequest) =>
      adapter.post<ResearchPlanningClarifyResponse>('/api/v1/research/planning/clarify', request),

    draftPlan: (request: ResearchPlanningDraftRequest) =>
      adapter.post<ResearchPlanningDraftResponse>('/api/v1/research/planning/draft', request),

    createJob: (request: ResearchJobCreateRequest) =>
      adapter.post<JobCreatedResponse>('/api/v1/research/jobs', request),

    getJob: (jobId: string) => adapter.get<ResearchJob>(`/api/v1/research/jobs/${encodeURIComponent(jobId)}`),

    getArtifacts: (jobId: string) => adapter.get<ResearchArtifactsResponse>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/artifacts`
      ),

    approve: (jobId: string, request: ResearchApproveRequest) =>
      adapter.post<ResearchApproveResponse>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/approve`,
        {
          ...request,
          sources: request.sources?.map((source) => String(source)),
        }
      ),

    archivePreview: (jobId: string, request: ResearchArchivePreviewRequest) =>
      adapter.post<KnowledgePreview>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/archive-preview`,
        request,
      ),

    archiveConfirm: (jobId: string, previewId: string) =>
      adapter.post<Record<string, unknown>>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/archive-confirm`,
        { preview_id: previewId },
      ),

    followup: (jobId: string, request: ResearchFollowupRequest) =>
      adapter.post<ResearchFollowupResponse>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/followup`,
        request
      ),

    reviseSections: (jobId: string, request: ResearchSectionRevisionRequest) =>
      adapter.post<ResearchSectionRevisionResponse>(
        `/api/v1/research/jobs/${encodeURIComponent(jobId)}/revise-sections`,
        request
      ),
  };
}
