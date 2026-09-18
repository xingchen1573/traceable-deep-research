import type { QueryMode } from './knowledge-base';

export type KnowledgeNodeType = 'knowledge_base' | 'theme' | 'research' | 'legacy';
export type ResearchTimeStatus = 'active' | 'possibly_stale' | 'superseded';

export interface ResearchIndexStatus {
  status: 'pending' | 'running' | 'ready' | 'failed' | string;
  job_id?: string | null;
  error?: string | null;
  updated_at?: string;
}

export interface ResearchDocumentMeta {
  title?: string;
  filename: string;
  bytes?: number;
  sha256?: string;
  html_filename?: string;
  html_bytes?: number;
  html_sha256?: string;
}

export interface ResearchSource {
  title?: string;
  url?: string;
  [key: string]: unknown;
}

export interface KnowledgeResearchItem {
  type: 'research';
  id: string;
  research_id: string;
  knowledge_base_id: string;
  theme_id?: string | null;
  title: string;
  summary?: string;
  path: string;
  researched_at?: string;
  status: ResearchTimeStatus;
  tags?: string[];
  documents: ResearchDocumentMeta[];
  document_count: number;
  html_filename?: string;
  html_bytes?: number;
  html_sha256?: string;
  sources?: ResearchSource[];
  source_count?: number;
  evidence_count?: number;
  index_status?: ResearchIndexStatus;
  updated_at?: string;
}

export interface KnowledgeTheme {
  type: 'theme';
  id: string;
  theme_id: string;
  knowledge_base_id: string;
  title: string;
  description?: string;
  path: string;
  overview_path?: string | null;
  research_items: KnowledgeResearchItem[];
  research_count: number;
  sort_order?: number;
  updated_at?: string;
}

export interface HierarchicalKnowledgeBase {
  type: 'knowledge_base';
  id: string;
  knowledge_base_id: string;
  title: string;
  description?: string;
  path: string;
  overview_path?: string | null;
  themes: KnowledgeTheme[];
  research_items: KnowledgeResearchItem[];
  research_count: number;
  tags?: string[];
  sort_order?: number;
  updated_at?: string;
}

export interface LegacyKnowledgeTopic {
  legacy_topic: string;
  path: string;
  document_count: number;
  updated_at?: string;
}

export interface KnowledgeTreeResponse {
  schema_version: number;
  knowledge_bases: HierarchicalKnowledgeBase[];
  legacy: LegacyKnowledgeTopic[];
  trash_count: number;
}

export interface TrashItem {
  type: KnowledgeNodeType;
  id: string;
  title: string;
  original_path?: string;
  trash_path: string;
  trashed_at?: string;
}

export interface KnowledgeTrashResponse {
  items: TrashItem[];
  count: number;
}

export type KnowledgeTreeAction =
  | 'create_knowledge_base'
  | 'create_theme'
  | 'archive_research'
  | 'update_research_status'
  | 'rename_node'
  | 'move_research'
  | 'trash_node'
  | 'trash_legacy'
  | 'restore_node';

export interface KnowledgeChange {
  op: string;
  node_type?: string;
  node_id?: string;
  path?: string;
  from?: string;
  to?: string;
  title?: string;
  [key: string]: unknown;
}

export interface KnowledgePreview {
  preview_id: string;
  action: KnowledgeTreeAction;
  state: 'pending' | 'committed' | 'stale' | 'failed';
  created_at?: string;
  updated_at?: string;
  changes: KnowledgeChange[];
  result?: Record<string, unknown>;
  archive?: {
    title?: string;
    summary?: string;
    knowledge_base?: { id?: string; title?: string; created?: boolean };
    theme?: { id?: string; title?: string; created?: boolean } | null;
    documents?: ResearchDocumentMeta[];
    tags?: string[];
    source_count?: number;
    evidence_count?: number;
  };
  recommendation?: {
    reason?: string;
    confidence?: string;
    [key: string]: unknown;
  };
}

export interface KnowledgeOverviewResponse {
  node_type: 'knowledge_base' | 'theme';
  node_id: string;
  title: string;
  filename: string;
  content: string;
}

export interface ResearchDocumentResponse {
  research_id: string;
  title: string;
  filename: string;
  content: string;
  has_html: boolean;
  html_filename?: string | null;
  citations: Array<{
    number: number;
    title?: string;
    url?: string;
    evidence: Array<{ evidence_id?: string; quote?: string; claim?: string }>;
  }>;
}

export interface KnowledgeScopeQueryResponse {
  scope: { type: 'knowledge_base' | 'theme' | 'research'; id: string; title?: string };
  status?: string;
  answer: string;
  citations: Array<{
    title?: string;
    path?: string;
    snippet?: string;
    content?: string;
    text?: string;
    research_id?: string;
    research_title?: string;
  }>;
  warnings?: string[];
  queried_research_count?: number;
  gaps?: string[];
  mode?: QueryMode;
  [key: string]: unknown;
}

export interface KnowledgeScopeQueryRequest {
  question: string;
  mode?: QueryMode;
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  include_history?: boolean;
  max_candidates?: number;
}

export interface ResearchArchivePreviewRequest {
  knowledge_base_id?: string;
  theme_id?: string;
  new_knowledge_base?: { title: string; description?: string };
  new_theme?: { title: string; description?: string };
  title?: string;
  tags?: string[];
  acknowledge_matrix_gaps?: boolean;
}
