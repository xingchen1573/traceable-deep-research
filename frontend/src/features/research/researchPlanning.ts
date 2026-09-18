import { ConfirmedResearchPlan, ResearchDepth, ResearchJobCreateRequest, ResearchPlanSection, ResearchStrategy } from '@/types';

const PLANNING_DRAFT_KEY = 'harness.research.current-plan.v1';

export interface PersistedResearchPlanningDraft {
  request: string;
  resolvedRequest: string;
  topic: string;
  depth: ResearchDepth;
  reconToken: string;
  disambiguationTerms: string[];
  plannedSections: ResearchPlanSection[];
  confirmedPlan: ConfirmedResearchPlan;
  planId: string;
  planToken: string;
  strategy: ResearchStrategy;
}

export function loadResearchPlanningDraft(): PersistedResearchPlanningDraft | undefined {
  if (typeof window === 'undefined') return undefined;
  try {
    const value = JSON.parse(window.localStorage.getItem(PLANNING_DRAFT_KEY) || 'null');
    return value && Array.isArray(value.plannedSections) && value.confirmedPlan ? value : undefined;
  } catch {
    return undefined;
  }
}

export function saveResearchPlanningDraft(value: PersistedResearchPlanningDraft) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(PLANNING_DRAFT_KEY, JSON.stringify(value));
}

export function clearResearchPlanningDraft() {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(PLANNING_DRAFT_KEY);
}

export function buildClarifiedResearchRequest(baseRequest: string, clarificationAnswer: string) {
  return `${baseRequest.trim()}\n\n用户补充说明：${clarificationAnswer.trim()}`;
}

export function buildConfirmedResearchJobRequest(input: {
  request: string;
  resolvedRequest?: string;
  topic?: string;
  depth: ResearchDepth;
  strategy: ResearchStrategy;
  plannedSections: ResearchPlanSection[];
  confirmedPlan: ConfirmedResearchPlan;
  planId: string;
  planToken: string;
  disambiguationTerms?: string[];
}): ResearchJobCreateRequest {
  return {
    user_request: input.resolvedRequest?.trim() || input.request.trim(),
    source_mode: 'web',
    pipeline_mode: 'question_loop',
    research_strategy: input.strategy,
    depth: input.depth,
    output_mode: 'document_set',
    topic: input.topic?.trim() || undefined,
    planned_sections: input.plannedSections,
    confirmed_plan: input.confirmedPlan,
    plan_id: input.planId,
    plan_token: input.planToken,
    plan_policy: 'confirmed_required',
    disambiguation_terms: input.disambiguationTerms || [],
    auto_save: false,
  };
}
