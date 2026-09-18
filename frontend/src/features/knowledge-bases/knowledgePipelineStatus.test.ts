import { describe, expect, it } from 'vitest';
import type { KnowledgeBaseDetail } from '@/types';
import { deriveKnowledgePipelineStatus } from './knowledgePipelineStatus';

function detail(overrides: Partial<KnowledgeBaseDetail> = {}): KnowledgeBaseDetail {
  return {
    topic: 'topic',
    metadata: { title: 'Topic' },
    metadata_status: 'explicit',
    document_count: 1,
    documents: [
      {
        document_id: 'doc',
        title: 'Doc',
        filename: 'doc.md',
        path: 'doc.md',
        bytes: 10,
        sha256: 'sha',
        frontmatter: {},
      },
    ],
    index_status: 'missing',
    index_reason: 'missing_metadata',
    ...overrides,
  } as KnowledgeBaseDetail;
}

describe('deriveKnowledgePipelineStatus', () => {
  it('does not advance to vectors when a failed index left partial chunks behind', () => {
    const status = deriveKnowledgePipelineStatus(
      detail({ index_status: 'incomplete', vector_stats: { chunks: 12, entities: 0, relationships: 0 } }),
    );

    expect(status.index).toBe(false);
    expect(status.vectors).toBe(false);
    expect(status.qa).toBe(false);
  });

  it('marks downstream stages only for a complete usable index', () => {
    const status = deriveKnowledgePipelineStatus(
      detail({ index_status: 'ready', vector_stats: { chunks: 12, entities: 5, relationships: 3 } }),
    );

    expect(status.index).toBe(true);
    expect(status.vectors).toBe(true);
    expect(status.qa).toBe(true);
  });
});
