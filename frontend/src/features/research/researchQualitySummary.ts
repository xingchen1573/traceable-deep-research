import type { ResearchJobResult } from '@/types';

export type QualityGateCheck = {
  id: string;
  label: string;
  status: boolean | null;
};

export type QualitySummary = {
  status: string;
  tone: 'pass' | 'warn' | 'fail' | 'pending';
  title: string;
  reason: string;
  approvalReady: boolean;
  validationStatus: string;
  sourceCount: number;
  materialCount: number;
  evidenceCount: number;
  documentCount: number;
  matrixCovered: number;
  matrixTotal: number;
  matrixSource: 'backend' | 'client';
  weakMatrixCount: number;
  unsupportedClaimCount: number;
  missingInlineRefCount: number;
  reviewItemCount: number;
  questionGapCount: number;
  warnings: string[];
  gateChecks: QualityGateCheck[];
  gatePassedCount: number;
  gateAvailableCount: number;
};

type MatrixEvidenceSummary = {
  rows: Array<{ evidenceCount: number; sourceCount: number }>;
  weakCellCount: number;
};

function checkStatus(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

export function buildQualityGateChecks(
  qualityGate?: ResearchJobResult['quality_gate'],
): QualityGateCheck[] {
  const checks = qualityGate?.checks || {};
  const reportQuality = checks.report_quality;
  const reportQualityStatus = typeof reportQuality === 'boolean'
    ? reportQuality
    : reportQuality && typeof reportQuality === 'object' && 'status' in reportQuality
      ? String((reportQuality as { status?: unknown }).status || '') === 'passed'
      : null;

  return [
    { id: 'grounding', label: '证据可追溯', status: checkStatus(checks.grounding) },
    { id: 'citation_validation', label: '引用结构正确', status: checkStatus(checks.citation_validation) },
    { id: 'report_quality', label: '报告无硬性缺陷', status: reportQualityStatus },
    { id: 'core_claim_support', label: '核心结论有证据支持', status: checkStatus(checks.core_claim_support) },
    { id: 'semantic_review_complete', label: '语义复核已完成', status: checkStatus(checks.semantic_review_complete) },
    { id: 'report_completeness', label: '核心章节可交付', status: checkStatus(checks.report_completeness) },
    { id: 'fallback_sections_absent', label: '没有模板降级章节', status: checkStatus(checks.fallback_sections_absent) },
  ];
}

export function buildQualitySummary({
  qualityGate,
  validation,
  coverage,
  matrixCoverage,
  matrix,
  documents,
  sources,
  materials,
  evidenceMap,
  evidencePack,
  qualityReason,
}: {
  qualityGate?: ResearchJobResult['quality_gate'];
  validation?: ResearchJobResult['validation'];
  coverage?: ResearchJobResult['coverage'];
  matrixCoverage?: ResearchJobResult['matrix_coverage'];
  matrix: MatrixEvidenceSummary;
  documents: NonNullable<ResearchJobResult['documents']>;
  sources: NonNullable<ResearchJobResult['sources']>;
  materials: NonNullable<ResearchJobResult['materials']>;
  evidenceMap: NonNullable<ResearchJobResult['evidence_map']>;
  evidencePack: NonNullable<ResearchJobResult['evidence_pack']>;
  qualityReason?: string;
}): QualitySummary {
  const unsupportedClaimCount = (validation?.per_document || []).reduce(
    (sum, item) => sum + (item.unsupported_claims?.length || 0),
    0,
  );
  const missingInlineRefCount = (validation?.per_document || []).reduce(
    (sum, item) => sum + (item.missing_inline_refs?.length || 0),
    0,
  );
  const status = qualityGate?.status || validation?.status || (documents.length ? 'pending' : 'waiting');
  const approvalReady = Boolean(
    qualityGate?.approval_ready ?? (documents.length > 0 && validation?.status !== 'failed'),
  );
  const tone: QualitySummary['tone'] =
    status === 'failed' || validation?.status === 'failed'
      ? 'fail'
      : approvalReady && status === 'passed'
        ? 'pass'
        : documents.length
          ? 'warn'
          : 'pending';
  const title =
    tone === 'pass'
      ? '质量校验已通过'
      : tone === 'fail'
        ? '存在阻断问题'
        : tone === 'warn'
          ? '需要复核证据'
          : '等待调研产物';

  const backendMatrixTotal = Number(matrixCoverage?.required_cell_count);
  const backendMatrixCovered = Number(matrixCoverage?.complete_cell_count);
  const hasBackendMatrixCounts = matrixCoverage?.enabled === true
    && Number.isFinite(backendMatrixTotal)
    && Number.isFinite(backendMatrixCovered);
  const matrixTotal = hasBackendMatrixCounts ? backendMatrixTotal : matrix.rows.length;
  const matrixCovered = hasBackendMatrixCounts
    ? backendMatrixCovered
    : matrix.rows.filter((row) => row.evidenceCount > 0 && row.sourceCount > 0).length;
  const weakMatrixCount = hasBackendMatrixCounts
    ? Math.max(matrixCoverage?.gaps?.length || 0, matrixTotal - matrixCovered)
    : matrix.weakCellCount;
  const answerCounts = coverage?.answer_status_counts;
  const questionGapCount = Number(answerCounts?.partial || 0) + Number(answerCounts?.uncovered || 0);
  const gateChecks = buildQualityGateChecks(qualityGate);
  const availableChecks = gateChecks.filter((check) => check.status !== null);

  return {
    status,
    tone,
    title,
    reason: qualityReason || qualityGate?.reason || validation?.reason || coverage?.reason || '调研完成后会展示质量判断。',
    approvalReady,
    validationStatus: validation?.status || 'pending',
    sourceCount: coverage?.source_count ?? sources.length,
    materialCount: materials.length || coverage?.document_count || 0,
    evidenceCount: evidenceMap.length || coverage?.evidence_pack_count || evidencePack.length,
    documentCount: documents.length,
    matrixCovered,
    matrixTotal,
    matrixSource: hasBackendMatrixCounts ? 'backend' : 'client',
    weakMatrixCount,
    unsupportedClaimCount,
    missingInlineRefCount,
    reviewItemCount: unsupportedClaimCount + missingInlineRefCount + weakMatrixCount,
    questionGapCount,
    warnings: (qualityGate?.warnings || []).map(String).filter(Boolean),
    gateChecks,
    gatePassedCount: availableChecks.filter((check) => check.status === true).length,
    gateAvailableCount: availableChecks.length,
  };
}
