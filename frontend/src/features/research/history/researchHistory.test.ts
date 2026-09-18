import { describe, expect, it } from 'vitest';
import { ResearchJob } from '@/types';
import {
  deriveHistoryDisplay,
  filterAndSortHistory,
  getResearchStudioSearchParams,
  matchesSearch,
} from './researchHistory';

function makeJob(overrides: Partial<ResearchJob> = {}): ResearchJob {
  return {
    job_id: 'job_demo_history_abcdef',
    job_type: 'research',
    status: 'succeeded',
    created_at: '2026-01-01 00:00:00',
    updated_at: '2026-01-01 00:00:00',
    input: {},
    result: null,
    error: null,
    artifacts: [],
    ...overrides,
  };
}

describe('deriveHistoryDisplay', () => {
  it('classifies single_url jobs and prefers source_title over url', () => {
    const job = makeJob({
      result: { status: 'succeeded', pipeline_mode: 'single_url', source_title: 'Some Article', url: 'https://example.com/a' },
    });
    const display = deriveHistoryDisplay(job);
    expect(display.type).toBe('single_url');
    expect(display.typeLabel).toBe('链接简报');
    expect(display.title).toBe('Some Article');
  });

  it('falls back to url then input.url then a stable placeholder for single_url jobs', () => {
    const urlOnly = deriveHistoryDisplay(
      makeJob({ result: { status: 'succeeded', pipeline_mode: 'single_url', url: 'https://example.com/b' } })
    );
    expect(urlOnly.title).toBe('https://example.com/b');

    const inputUrlOnly = deriveHistoryDisplay(
      makeJob({ result: { status: 'succeeded', pipeline_mode: 'single_url' }, input: { url: 'https://example.com/c' } })
    );
    expect(inputUrlOnly.title).toBe('https://example.com/c');

    const noneAtAll = deriveHistoryDisplay(makeJob({ result: { status: 'succeeded', pipeline_mode: 'single_url' } }));
    expect(noneAtAll.title).toBe('URL 简报');
  });

  it('classifies deep research jobs (any non-single_url pipeline_mode, including missing)', () => {
    const withoutPipelineMode = deriveHistoryDisplay(makeJob({ result: { status: 'succeeded', suggested_topic: '主题 A' } }));
    expect(withoutPipelineMode.type).toBe('deep_research');
    expect(withoutPipelineMode.title).toBe('主题 A');

    const legacyMode = deriveHistoryDisplay(makeJob({ result: { status: 'succeeded', pipeline_mode: 'legacy', save_result: { topic: '主题 B' } } }));
    expect(legacyMode.type).toBe('deep_research');
    expect(legacyMode.title).toBe('主题 B');
  });

  it('falls back to input.user_request then a stable placeholder for deep research jobs', () => {
    const fromRequest = deriveHistoryDisplay(makeJob({ result: null, input: { user_request: '调研 X' } }));
    expect(fromRequest.title).toBe('调研 X');

    const nothing = deriveHistoryDisplay(makeJob({ result: null, input: {} }));
    expect(nothing.title).toBe('调研任务');
  });

  it('does not fabricate missing quality/doc/source counts', () => {
    const display = deriveHistoryDisplay(makeJob({ result: { status: 'succeeded' } }));
    expect(display.qualityScore).toBeUndefined();
    expect(display.docCount).toBeUndefined();
    expect(display.sourceCount).toBeUndefined();
  });

  it('reads quality/doc/source counts from the best-available result fields', () => {
    const display = deriveHistoryDisplay(
      makeJob({
        result: {
          status: 'succeeded',
          report_quality: { average_score: 4.5 },
          documents: [{ title: 'a', content: '' }, { title: 'b', content: '' }],
          coverage: { source_count: 6 } as never,
        },
      })
    );
    expect(display.qualityScore).toBe(4.5);
    expect(display.docCount).toBe(2);
    expect(display.sourceCount).toBe(6);
  });
});

describe('getResearchStudioSearchParams', () => {
  it('removes job_id when returning to history and preserves unrelated parameters', () => {
    const current = new URLSearchParams('job_id=job_123&source=notification');

    expect(getResearchStudioSearchParams(current, 'history').toString()).toBe('source=notification');
    expect(current.get('job_id')).toBe('job_123');
  });
});

describe('matchesSearch', () => {
  const job = makeJob({
    job_id: 'job_demo_history_needle',
    result: { status: 'succeeded', pipeline_mode: 'single_url', source_title: 'LangGraph 状态管理', url: 'https://example.com/langgraph' },
  });

  it('matches on title, url, and job_id (case-insensitively)', () => {
    expect(matchesSearch(job, 'LangGraph')).toBe(true);
    expect(matchesSearch(job, 'langgraph')).toBe(true);
    expect(matchesSearch(job, 'example.com')).toBe(true);
    expect(matchesSearch(job, 'needle')).toBe(true);
    expect(matchesSearch(job, '')).toBe(true);
  });

  it('does not match unrelated terms', () => {
    expect(matchesSearch(job, 'nonexistent-term')).toBe(false);
  });
});

describe('filterAndSortHistory', () => {
  const waitingSingleUrl = makeJob({
    job_id: 'job_waiting',
    status: 'waiting_approval',
    result: { status: 'waiting_approval', pipeline_mode: 'single_url', source_title: 'Brief A' },
  });
  const succeededDeepResearch = makeJob({
    job_id: 'job_succeeded',
    status: 'succeeded',
    result: { status: 'succeeded', suggested_topic: 'Deep Research A' },
  });
  const failedDeepResearch = makeJob({
    job_id: 'job_failed',
    status: 'failed',
    result: { status: 'failed', suggested_topic: 'Deep Research B' },
  });
  const jobs = [waitingSingleUrl, succeededDeepResearch, failedDeepResearch];

  it('filters by type', () => {
    const result = filterAndSortHistory(jobs, { type: 'single_url' });
    expect(result.map((j) => j.job_id)).toEqual(['job_waiting']);
  });

  it('filters by status', () => {
    const result = filterAndSortHistory(jobs, { status: 'failed' });
    expect(result.map((j) => j.job_id)).toEqual(['job_failed']);
  });

  it('combines type and status filters', () => {
    const result = filterAndSortHistory(jobs, { type: 'deep_research', status: 'succeeded' });
    expect(result.map((j) => j.job_id)).toEqual(['job_succeeded']);
  });

  it('combines search with type/status filters', () => {
    const result = filterAndSortHistory(jobs, { search: 'Deep Research', type: 'deep_research' });
    expect(result.map((j) => j.job_id)).toEqual(['job_succeeded', 'job_failed']);
  });

  it('preserves the input order (relies on the caller pre-sorting waiting_approval first)', () => {
    const result = filterAndSortHistory(jobs, {});
    expect(result.map((j) => j.job_id)).toEqual(['job_waiting', 'job_succeeded', 'job_failed']);
  });
});
