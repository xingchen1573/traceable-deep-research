export const RESEARCH_STEP_FALLBACK_TARGET = 'research-step-detail';

export const RESEARCH_STEP_TARGETS = {
  plan: 'research-plan-panel',
  query_plan: 'research-plan-panel',
  search: 'research-source-panel',
  fetch: 'research-material-panel',
  material_pool: 'research-material-panel',
  chunk_select: 'research-chunks-panel',
  evidence: 'research-evidence-ledger',
  coverage: 'research-coverage-panel',
  draft: 'research-document-preview',
  validation: 'research-validation-panel',
  approval: 'research-approval-panel',
} as const;

export type KnownResearchStepId = keyof typeof RESEARCH_STEP_TARGETS;

export function getResearchStepTargetId(stepId: string): string {
  return RESEARCH_STEP_TARGETS[stepId as KnownResearchStepId] || RESEARCH_STEP_FALLBACK_TARGET;
}
