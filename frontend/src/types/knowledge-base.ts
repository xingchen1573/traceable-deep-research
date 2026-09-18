export interface KnowledgeBase {
  topic: string;
  title: string;
  document_count: number;
  entry_document: string;
  updated_at: string;
  index_status: IndexStatus;
  metadata_status?: MetadataStatus;
  source_mode?: SourceMode;
  version?: string;
  metadata?: KnowledgeBaseMetadata;
  graph_stats?: GraphStats;
  vector_stats?: VectorStats;
}

export type IndexStatus = 'ready' | 'stale' | 'missing' | 'invalid' | 'incomplete' | 'error';
export type MetadataStatus = 'explicit' | 'inferred' | 'missing';
export type SourceMode = 'web' | 'repo' | 'mixed';

export interface GraphStats {
  nodes: number;
  edges: number;
}

export interface VectorStats {
  chunks: number;
  entities: number;
  relationships: number;
}

export interface KnowledgeBaseListResponse {
  active_topic: string | null;
  topic_count: number;
  topics: KnowledgeBase[];
}

export interface ActiveKnowledgeBaseResponse {
  active_topic: string | null;
  thread_id?: string;
}

export interface KnowledgeBaseDetail {
  topic: string;
  metadata: KnowledgeBaseMetadata;
  metadata_status: MetadataStatus;
  document_count: number;
  documents: DocumentMeta[];
  index_status: IndexStatus;
  index_reason?: string;
  index_diagnosis: Record<string, unknown>;
  embedding_model?: string;
  embedding_dim?: number;
  graph_stats?: GraphStats;
  vector_stats?: VectorStats;
}

export interface KnowledgeBaseMetadata {
  topic?: string;
  title?: string;
  version?: string;
  source_mode?: SourceMode;
  description?: string;
  [key: string]: unknown;
}

export interface DocumentMeta {
  document_id: string;
  title: string;
  filename: string;
  path: string;
  bytes: number;
  sha256: string;
  frontmatter: Record<string, unknown>;
}

export interface DocumentResponse {
  topic: string;
  document_id: string;
  title: string;
  content: string;
  frontmatter: Record<string, unknown>;
  path: string;
}

export interface DocumentListResponse {
  topic: string;
  documents: DocumentMeta[];
}

export interface KnowledgeBaseSummaryResponse {
  topic: string;
  summary: string;
  status: Record<string, unknown>;
}

export type QueryBackend = 'auto' | 'lightrag' | 'local';
export type QueryMode = 'mix' | 'local' | 'global' | 'hybrid' | 'naive';

export interface KnowledgeQueryHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Citation {
  title: string;
  path: string;
  snippet: string;
  research_id?: string;
  research_title?: string;
}

export interface QueryResponse {
  status: string;
  topic: string;
  backend: QueryBackend;
  requested_backend?: QueryBackend;
  actual_backend?: QueryBackend | 'mixed';
  fallback_occurred?: boolean;
  fallback_reason?: string;
  mode: QueryMode | 'markdown';
  answer: string;
  citations: Citation[];
  confidence: 'high' | 'medium' | 'low';
  gaps: string[];
  metadata?: {
    embedding_model?: string;
    embedding_dim?: number;
    indexed_at?: string;
    [key: string]: unknown;
  };
}

export interface KnowledgeQaHistoryEntry {
  id: string;
  topic: string;
  topic_title: string;
  backend: string;
  mode: string;
  title: string;
  messages: Array<Record<string, unknown>>;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeQaHistoryListResponse {
  items: KnowledgeQaHistoryEntry[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface IndexStatusResponse {
  topic: string;
  index_status: IndexStatus;
  index_reason: string;
  diagnosis: Record<string, unknown>;
  embedding_model?: string;
  embedding_dim?: number;
  indexed_at?: string;
  workspace?: string;
}

export interface GraphNodeLite {
  id: string;
  type: string;
  degree: number;
}

export interface GraphEdgeLite {
  source: string;
  target: string;
  weight: number;
}

export interface GraphTopEntity {
  entity: string;
  type?: string;
  count: number;
}

export interface GraphStatsResponse {
  topic: string;
  nodes: number;
  edges: number;
  top_entities: GraphTopEntity[];
  workspace?: string;
  graph_file?: string;
  graph_nodes: GraphNodeLite[];
  graph_edges: GraphEdgeLite[];
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
  node_limit: number;
}

export interface GraphNodeEdgeDetail {
  entity_id: string;
  entity_type: string;
  weight: number;
  keywords: string;
  description: string;
  neighbor_preview: string;
}

export interface GraphNodeDetailResponse {
  topic: string;
  entity_id: string;
  entity_type: string;
  degree: number;
  description: string;
  source_id: string;
  file_path: string;
  edges: GraphNodeEdgeDetail[];
}

export interface VectorStatsResponse {
  topic: string;
  chunks: number;
  entities: number;
  relationships: number;
  embedding_model?: string;
  embedding_dim?: number;
  chunk_size?: number;
  chunk_overlap_size?: number;
}

export interface DeleteIndexResponse {
  status: string;
  topic: string;
  deleted_path: string;
}
