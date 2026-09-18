import type { ResearchJobResult } from '@/types';

type SelectedChunk = NonNullable<ResearchJobResult['selected_chunks']>[number];
type EvidenceRow = NonNullable<ResearchJobResult['evidence_map']>[number];
type ResearchDocument = NonNullable<ResearchJobResult['documents']>[number];

export const CHUNK_LIFECYCLE_STAGES = [
  { id: 'candidate', label: '仅候选' },
  { id: 'evidence', label: '已成证据' },
  { id: 'writer', label: '已送入写作' },
  { id: 'cited', label: '正文已引用' },
] as const;

export type ChunkLifecycleStage = (typeof CHUNK_LIFECYCLE_STAGES)[number]['id'];

export type ChunkLifecycle = {
  stage: ChunkLifecycleStage;
  stageIndex: number;
  evidenceIds: string[];
  writerEvidenceIds: string[];
  citedEvidenceIds: string[];
};

function stringIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(String).filter(Boolean);
}

function comparableText(value: unknown): string {
  return String(value || '')
    .normalize('NFKC')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

export function buildChunkLifecycleMap({
  chunks,
  evidenceMap,
  documents,
  contextDiagnostics = [],
}: {
  chunks: SelectedChunk[];
  evidenceMap: EvidenceRow[];
  documents: ResearchDocument[];
  contextDiagnostics?: Array<Record<string, unknown>>;
}): Map<string, ChunkLifecycle> {
  const writerEvidenceIds = new Set<string>();
  const citedEvidenceIds = new Set<string>();

  contextDiagnostics.forEach((row) => {
    stringIds(row.selected_evidence_ids).forEach((id) => writerEvidenceIds.add(id));
  });

  documents.forEach((document) => {
    stringIds(document.synthesis_diagnostics?.selected_evidence_ids).forEach((id) => writerEvidenceIds.add(id));
    (document.claim_ledger || []).forEach((claim) => {
      stringIds(claim.evidence_ids).forEach((id) => citedEvidenceIds.add(id));
    });
    (document.evidence_table || []).forEach((row) => {
      if (row.used_in_document && row.id) citedEvidenceIds.add(row.id);
    });
  });

  evidenceMap.forEach((row) => {
    if (row.id && (row.used_by_documents || []).length) citedEvidenceIds.add(row.id);
  });
  citedEvidenceIds.forEach((id) => writerEvidenceIds.add(id));

  const evidenceIdsByChunk = new Map<string, string[]>();
  evidenceMap.forEach((row) => {
    const evidenceId = String(row.id || '');
    if (!evidenceId) return;
    const directChunkId = String(row.chunk_id || '');
    if (directChunkId) {
      evidenceIdsByChunk.set(directChunkId, [...(evidenceIdsByChunk.get(directChunkId) || []), evidenceId]);
      return;
    }

    // Historical jobs and local mock fixtures may predate chunk_id. Link them only when the
    // source URL agrees and the frozen quote has one unambiguous text match in the chunk set.
    const quote = comparableText(row.cleaned_quote || row.raw_quote || row.evidence_text || row.claim);
    const sourceUrl = String(row.source_url || '');
    if (!quote || !sourceUrl) return;
    const matches = chunks.filter((chunk) => {
      const text = comparableText(chunk.text);
      return String(chunk.source_url || '') === sourceUrl
        && Boolean(text)
        && (text.includes(quote) || quote.includes(text));
    });
    if (matches.length !== 1 || !matches[0].id) return;
    const matchedChunkId = String(matches[0].id);
    evidenceIdsByChunk.set(matchedChunkId, [...(evidenceIdsByChunk.get(matchedChunkId) || []), evidenceId]);
  });

  return new Map(
    chunks.map((chunk, index) => {
      const chunkId = String(chunk.id || `chunk_${index}`);
      const evidenceIds = evidenceIdsByChunk.get(chunkId) || [];
      const chunkWriterEvidenceIds = evidenceIds.filter((id) => writerEvidenceIds.has(id));
      const chunkCitedEvidenceIds = evidenceIds.filter((id) => citedEvidenceIds.has(id));
      const stageIndex = chunkCitedEvidenceIds.length
        ? 3
        : chunkWriterEvidenceIds.length
          ? 2
          : evidenceIds.length
            ? 1
            : 0;

      return [
        chunkId,
        {
          stage: CHUNK_LIFECYCLE_STAGES[stageIndex].id,
          stageIndex,
          evidenceIds,
          writerEvidenceIds: chunkWriterEvidenceIds,
          citedEvidenceIds: chunkCitedEvidenceIds,
        },
      ];
    }),
  );
}
