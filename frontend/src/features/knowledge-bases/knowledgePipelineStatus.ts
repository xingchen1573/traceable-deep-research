import type { KnowledgeBaseDetail } from '@/types';

export interface KnowledgePipelineStatus {
  documents: boolean;
  metadata: boolean;
  index: boolean;
  vectors: boolean;
  qa: boolean;
  evaluation: boolean;
}

export function deriveKnowledgePipelineStatus(kb: KnowledgeBaseDetail): KnowledgePipelineStatus {
  const indexReady = kb.index_status === 'ready';
  const vectorGraphReady = indexReady && Boolean(kb.vector_stats?.chunks);

  return {
    documents: kb.documents.length > 0,
    metadata: kb.metadata_status !== 'missing',
    index: indexReady,
    vectors: vectorGraphReady,
    qa: indexReady && vectorGraphReady,
    evaluation: false,
  };
}
