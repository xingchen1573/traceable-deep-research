import { describe, expect, it } from 'vitest';
import {
  getResearchStepTargetId,
  RESEARCH_STEP_FALLBACK_TARGET,
  RESEARCH_STEP_TARGETS,
} from './researchStepNavigation';

describe('research step navigation', () => {
  it('maps every production research step to a stable result panel', () => {
    expect(RESEARCH_STEP_TARGETS).toEqual({
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
    });
  });

  it('falls back to the step detail for a new or unknown backend step', () => {
    expect(getResearchStepTargetId('future_step')).toBe(RESEARCH_STEP_FALLBACK_TARGET);
  });
});
