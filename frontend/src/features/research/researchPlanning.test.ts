import { describe, expect, it, vi } from 'vitest';
import {
  buildClarifiedResearchRequest,
  buildConfirmedResearchJobRequest,
  clearResearchPlanningDraft,
  loadResearchPlanningDraft,
  saveResearchPlanningDraft,
} from './researchPlanning';

describe('research planning handoff', () => {
  it('keeps the original request and explicit clarification together', () => {
    expect(buildClarifiedResearchRequest('调研 Harness', '指 AI Agent 的 Harness Engineering')).toBe(
      '调研 Harness\n\n用户补充说明：指 AI Agent 的 Harness Engineering'
    );
  });

  it('hands the confirmed plan to question_loop without legacy required titles', () => {
    const plannedSections = [
      {
        id: 'sec_01',
        title: '01_核心机制',
        questions: [{ id: 'sec_01_q1', text: '核心机制如何工作？' }],
      },
    ];
    const request = buildConfirmedResearchJobRequest({
      request: '原始请求',
      resolvedRequest: '已澄清请求',
      topic: ' harness ',
      depth: 'deep',
      strategy: 'hv_analysis',
      plannedSections,
      confirmedPlan: {
        version: 'v1',
        user_request: '已澄清请求',
        topic: 'harness',
        strategy: 'hv_analysis',
        research_type: 'comparison',
        subjects: ['LangGraph', 'AutoGen'],
        comparison_axes: ['状态管理'],
        sections: plannedSections,
      },
      planId: 'plan_123',
      planToken: 'p1.signed',
      disambiguationTerms: ['AI agent'],
    });

    expect(request).toMatchObject({
      user_request: '已澄清请求',
      source_mode: 'web',
      pipeline_mode: 'question_loop',
      research_strategy: 'hv_analysis',
      depth: 'deep',
      output_mode: 'document_set',
      topic: 'harness',
      planned_sections: plannedSections,
      plan_id: 'plan_123',
      plan_token: 'p1.signed',
      plan_policy: 'confirmed_required',
      disambiguation_terms: ['AI agent'],
      auto_save: false,
    });
    expect(request.required_titles).toBeUndefined();
  });

  it('persists and restores the latest structured plan draft', () => {
    const storage = new Map<string, string>();
    vi.stubGlobal('window', {
      localStorage: {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => storage.set(key, value),
        removeItem: (key: string) => storage.delete(key),
      },
    });
    const plannedSections = [{ title: '01_核心机制', questions: [{ text: '如何工作？' }] }];
    const value = {
      request: '调研 Harness',
      resolvedRequest: '调研 AI Agent Harness',
      topic: 'harness',
      depth: 'normal' as const,
      reconToken: 'v2.signed',
      disambiguationTerms: ['AI agent'],
      plannedSections,
      confirmedPlan: {
        version: 'v1',
        user_request: '调研 AI Agent Harness',
        strategy: 'standard' as const,
        sections: plannedSections,
      },
      planId: 'plan_restore',
      planToken: 'p1.restore',
      strategy: 'standard' as const,
    };

    saveResearchPlanningDraft(value);
    expect(loadResearchPlanningDraft()).toEqual(value);
    clearResearchPlanningDraft();
    expect(loadResearchPlanningDraft()).toBeUndefined();
    vi.unstubAllGlobals();
  });
});
