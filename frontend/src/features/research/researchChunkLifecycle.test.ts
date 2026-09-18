import { describe, expect, it } from 'vitest';
import { buildChunkLifecycleMap } from './researchChunkLifecycle';

describe('buildChunkLifecycleMap', () => {
  it('distinguishes candidate, evidence, writer input, and cited chunks', () => {
    const chunks = [
      { id: 'chunk_candidate' },
      { id: 'chunk_evidence' },
      { id: 'chunk_writer' },
      { id: 'chunk_cited' },
    ];
    const evidenceMap = [
      { id: 'ev_evidence', chunk_id: 'chunk_evidence' },
      { id: 'ev_writer', chunk_id: 'chunk_writer' },
      { id: 'ev_cited', chunk_id: 'chunk_cited' },
    ];
    const documents = [{
      title: '证据引用比较',
      content: '',
      synthesis_diagnostics: { selected_evidence_ids: ['ev_writer', 'ev_cited'] },
      claim_ledger: [{ text: '已引用事实', evidence_ids: ['ev_cited'] }],
    }];

    const result = buildChunkLifecycleMap({ chunks, evidenceMap, documents });

    expect(result.get('chunk_candidate')?.stage).toBe('candidate');
    expect(result.get('chunk_evidence')?.stage).toBe('evidence');
    expect(result.get('chunk_writer')?.stage).toBe('writer');
    expect(result.get('chunk_cited')?.stage).toBe('cited');
  });

  it('treats used_by_documents as a conservative citation fallback for historical jobs', () => {
    const result = buildChunkLifecycleMap({
      chunks: [{ id: 'chunk_legacy' }],
      evidenceMap: [{ id: 'ev_legacy', chunk_id: 'chunk_legacy', used_by_documents: ['报告'] }],
      documents: [],
    });

    expect(result.get('chunk_legacy')).toMatchObject({
      stage: 'cited',
      writerEvidenceIds: ['ev_legacy'],
      citedEvidenceIds: ['ev_legacy'],
    });
  });

  it('uses context diagnostics when section documents are not available yet', () => {
    const result = buildChunkLifecycleMap({
      chunks: [{ id: 'chunk_live' }],
      evidenceMap: [{ id: 'ev_live', chunk_id: 'chunk_live' }],
      documents: [],
      contextDiagnostics: [{ selected_evidence_ids: ['ev_live'] }],
    });

    expect(result.get('chunk_live')?.stage).toBe('writer');
  });

  it('links a historical evidence row only when its source and quote match one chunk', () => {
    const result = buildChunkLifecycleMap({
      chunks: [
        { id: 'chunk_match', source_url: 'https://example.com/a', text: '页面前文。这是可核验的原文句子。页面后文。' },
        { id: 'chunk_other', source_url: 'https://example.com/b', text: '这是可核验的原文句子。' },
      ],
      evidenceMap: [{
        id: 'ev_historical',
        source_url: 'https://example.com/a',
        evidence_text: '这是可核验的原文句子。',
        used_by_documents: ['报告'],
      }],
      documents: [],
    });

    expect(result.get('chunk_match')?.stage).toBe('cited');
    expect(result.get('chunk_other')?.stage).toBe('candidate');
  });
});
