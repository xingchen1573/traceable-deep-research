import { describe, expect, it } from 'vitest';
import { buildQualitySummary } from './researchQualitySummary';

const baseChecks = {
  grounding: true,
  citation_validation: true,
  report_quality: { status: 'passed' },
  core_claim_support: true,
  semantic_review_complete: true,
  report_completeness: true,
  fallback_sections_absent: true,
};

function build(overrides: Record<string, unknown> = {}) {
  return buildQualitySummary({
    qualityGate: {
      status: 'passed',
      approval_ready: true,
      reason: '质量门通过',
      checks: baseChecks,
    },
    validation: { status: 'passed', per_document: [] },
    coverage: {
      ready_to_write: true,
      round_index: 2,
      max_rounds: 3,
      source_count: 9,
      document_count: 50,
      evidence_count: 59,
      key_fact_count: 0,
      depth_ready: false,
      covered_titles: [],
      missing_titles: [],
      fatal_gaps: [],
      followup_queries: [],
      reason: '',
      answer_status_counts: { complete: 2, partial: 0, uncovered: 1 },
    },
    matrixCoverage: {
      enabled: true,
      status: 'partial',
      complete: false,
      required_cell_count: 2,
      complete_cell_count: 0,
      gaps: [
        { cell_id: 'a', subject: 'A', axis: '能力', status: 'uncovered' },
        { cell_id: 'b', subject: 'B', axis: '能力', status: 'uncovered' },
      ],
      requires_gap_acknowledgement: true,
    },
    matrix: { rows: [], weakCellCount: 0 },
    documents: [{ title: '报告', content: '内容' }],
    sources: [],
    materials: [],
    evidenceMap: [],
    evidencePack: [],
    ...overrides,
  } as Parameters<typeof buildQualitySummary>[0]);
}

describe('buildQualitySummary', () => {
  it('separates passed hard gates from non-blocking matrix and coverage gaps', () => {
    const summary = build();

    expect(summary.tone).toBe('pass');
    expect(summary.gatePassedCount).toBe(7);
    expect(summary.gateAvailableCount).toBe(7);
    expect(summary.matrixSource).toBe('backend');
    expect([summary.matrixCovered, summary.matrixTotal]).toEqual([0, 2]);
    expect(summary.reviewItemCount).toBe(2);
    expect(summary.questionGapCount).toBe(1);
  });

  it('uses a report-quality object as the hard-gate verdict', () => {
    const summary = build({
      qualityGate: {
        status: 'needs_more_evidence',
        approval_ready: false,
        reason: '报告存在硬性缺陷',
        checks: { ...baseChecks, report_quality: { status: 'failed' } },
      },
    });

    expect(summary.gatePassedCount).toBe(6);
    expect(summary.gateChecks.find((check) => check.id === 'report_quality')?.status).toBe(false);
    expect(summary.approvalReady).toBe(false);
  });
});
