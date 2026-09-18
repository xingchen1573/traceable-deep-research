import { ApiAdapter } from '../adapters';
import {
  ActiveKnowledgeBaseResponse,
  DeleteIndexResponse,
  DocumentListResponse,
  DocumentResponse,
  GraphStatsResponse,
  GraphNodeDetailResponse,
  IndexStatusResponse,
  JobCreatedResponse,
  KnowledgeBaseDetail,
  KnowledgeBaseListResponse,
  KnowledgeBaseSummaryResponse,
  KnowledgeQueryHistoryMessage,
  KnowledgeQaHistoryListResponse,
  QueryBackend,
  QueryMode,
  QueryResponse,
  KnowledgeQaHistoryEntry,
  VectorStatsResponse,
} from '@/types';

function normalizeKnowledgeBaseList(data: KnowledgeBaseListResponse): KnowledgeBaseListResponse {
  return {
    ...data,
    topics: data.topics.map((topic) => ({
      ...topic,
      title: topic.title || topic.metadata?.title || topic.topic,
      entry_document: topic.entry_document || '',
      updated_at: topic.updated_at || '',
      metadata_status: topic.metadata_status || 'missing',
      source_mode: topic.source_mode || topic.metadata?.source_mode,
      version: topic.version || topic.metadata?.version,
    })),
  };
}

export function knowledgeBaseEndpoints(adapter: ApiAdapter) {
  return {
    list: (include_index = true, include_stats = true) =>
      adapter
        .get<KnowledgeBaseListResponse>('/api/v1/knowledge-bases', {
          include_index,
          include_stats,
        })
        .then(normalizeKnowledgeBaseList),

    getActive: (thread_id?: string) =>
      adapter.get<ActiveKnowledgeBaseResponse>('/api/v1/knowledge-bases/active', { thread_id }),

    setActive: (topic: string, thread_id?: string) =>
      adapter.post<ActiveKnowledgeBaseResponse>('/api/v1/knowledge-bases/active', {
        topic,
        thread_id,
      }),

    get: (topic: string) => adapter.get<KnowledgeBaseDetail>(`/api/v1/knowledge-bases/${encodeURIComponent(topic)}`),

    getSummary: (topic: string) => adapter.get<KnowledgeBaseSummaryResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/summary`
      ),

    listDocuments: (topic: string) => adapter.get<DocumentListResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/documents`
      ),

    getDocument: (topic: string, documentId: string) => adapter.get<DocumentResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/documents/${encodeURIComponent(documentId)}`
      ),

    query: (topic: string, question: string, backend: QueryBackend = 'auto', mode: QueryMode = 'mix') =>
      adapter.post<QueryResponse>(`/api/v1/knowledge-bases/${encodeURIComponent(topic)}/query`, {
        question,
        backend,
        mode,
        include_citations: true,
      }),

    queryStream: (
      topic: string,
      question: string,
      backend: QueryBackend = 'auto',
      mode: QueryMode = 'mix',
      onDelta: (text: string) => void,
      history: KnowledgeQueryHistoryMessage[] = [],
    ) =>
      adapter.stream<QueryResponse>(`/api/v1/knowledge-bases/${encodeURIComponent(topic)}/query/stream`, {
        question,
        history,
        backend,
        mode,
        include_citations: true,
      }, onDelta),

    listQaHistory: (limit = 20, offset = 0) =>
      adapter.get<KnowledgeQaHistoryListResponse>('/api/v1/knowledge-bases/history/qa', { limit, offset }),

    saveQaHistory: (entry: unknown) =>
      adapter.post<KnowledgeQaHistoryEntry>('/api/v1/knowledge-bases/history/qa', entry),

    deleteQaHistory: (id: string) =>
      adapter.delete<{ id: string; deleted: boolean }>(`/api/v1/knowledge-bases/history/qa/${encodeURIComponent(id)}`),

    getIndex: (topic: string) => adapter.get<IndexStatusResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/index`
      ),

    getGraph: (topic: string, maxNodes?: number) =>
      adapter.get<GraphStatsResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/graph`,
        maxNodes ? { max_nodes: maxNodes } : undefined
      ),

    getGraphNode: (topic: string, entityId: string) =>
      adapter.get<GraphNodeDetailResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/graph/nodes/${encodeURIComponent(entityId)}`
      ),

    getVectors: (topic: string) => adapter.get<VectorStatsResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/vectors`
      ),

    createIndexJob: (topic: string, force = false) => adapter.post<JobCreatedResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/index/jobs`,
        { force, mode: 'lightrag' }
      ),

    deleteIndex: (topic: string) => adapter.delete<DeleteIndexResponse>(
        `/api/v1/knowledge-bases/${encodeURIComponent(topic)}/index`
      ),
  };
}
