import type { ApiAdapter } from '../adapters';
import type {
  KnowledgeOverviewResponse,
  KnowledgePreview,
  KnowledgeScopeQueryRequest,
  KnowledgeScopeQueryResponse,
  KnowledgeTrashResponse,
  KnowledgeTreeAction,
  KnowledgeTreeResponse,
  ResearchDocumentResponse,
} from '@/types';
import type { GraphNodeDetailResponse, GraphStatsResponse, IndexStatusResponse, VectorStatsResponse } from '@/types';

export function knowledgeTreeEndpoints(adapter: ApiAdapter) {
  return {
    list: (includeLegacy = true) =>
      adapter.get<KnowledgeTreeResponse>('/api/v1/knowledge-tree', { include_legacy: includeLegacy }),
    trash: () => adapter.get<KnowledgeTrashResponse>('/api/v1/knowledge-tree/trash'),
    overview: (nodeType: 'knowledge_base' | 'theme', nodeId: string) =>
      adapter.get<KnowledgeOverviewResponse>(
        `/api/v1/knowledge-tree/nodes/${nodeType}/${encodeURIComponent(nodeId)}/overview`,
      ),
    document: (researchId: string, filename: string) =>
      adapter.get<ResearchDocumentResponse>(
        `/api/v1/knowledge-tree/research/${encodeURIComponent(researchId)}/documents/${encodeURIComponent(filename)}`,
      ),
    createPreview: (action: KnowledgeTreeAction, payload: Record<string, unknown>) =>
      adapter.post<KnowledgePreview>('/api/v1/knowledge-tree/previews', { action, payload }),
    getPreview: (previewId: string) =>
      adapter.get<KnowledgePreview>(`/api/v1/knowledge-tree/previews/${encodeURIComponent(previewId)}`),
    confirmPreview: (previewId: string) =>
      adapter.post<KnowledgePreview>(`/api/v1/knowledge-tree/previews/${encodeURIComponent(previewId)}/confirm`),
    queryScope: (scopeType: 'knowledge_base' | 'theme' | 'research', scopeId: string, request: KnowledgeScopeQueryRequest) =>
      adapter.post<KnowledgeScopeQueryResponse>(
        `/api/v1/knowledge-tree/scopes/${scopeType}/${encodeURIComponent(scopeId)}/query`,
        request,
      ),
    scopeIndex: (scopeType: 'knowledge_base' | 'theme' | 'research', scopeId: string) =>
      adapter.get<IndexStatusResponse>(
        `/api/v1/knowledge-tree/scopes/${scopeType}/${encodeURIComponent(scopeId)}/index`,
      ),
    scopeGraph: (scopeType: 'knowledge_base' | 'theme' | 'research', scopeId: string, maxNodes?: number) =>
      adapter.get<GraphStatsResponse>(
        `/api/v1/knowledge-tree/scopes/${scopeType}/${encodeURIComponent(scopeId)}/graph`,
        maxNodes ? { max_nodes: maxNodes } : undefined,
      ),
    scopeGraphNode: (
      scopeType: 'knowledge_base' | 'theme' | 'research',
      scopeId: string,
      entityId: string,
    ) => adapter.get<GraphNodeDetailResponse>(
      `/api/v1/knowledge-tree/scopes/${scopeType}/${encodeURIComponent(scopeId)}/graph/nodes/${encodeURIComponent(entityId)}`,
    ),
    scopeVectors: (scopeType: 'knowledge_base' | 'theme' | 'research', scopeId: string) =>
      adapter.get<VectorStatsResponse>(
        `/api/v1/knowledge-tree/scopes/${scopeType}/${encodeURIComponent(scopeId)}/vectors`,
      ),
    retryIndex: (researchId: string, force = true) =>
      adapter.post<{ job_id: string; job_type: string; status: string }>(
        `/api/v1/knowledge-tree/research/${encodeURIComponent(researchId)}/index/jobs`,
        { force },
      ),
    deleteResearchIndex: (researchId: string) =>
      adapter.delete<{ status: string; research_id: string; deleted_path: string }>(
        `/api/v1/knowledge-tree/research/${encodeURIComponent(researchId)}/index`,
      ),
  };
}
