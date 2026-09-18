import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChannelPushConfig, JobCreatedResponse, ResearchJob } from '@/types';

async function createTestMockAdapter() {
  await import('./adapters');
  const { createMockAdapter } = await import('./mockAdapter');
  return createMockAdapter();
}

async function resolveMockRequest<T>(request: Promise<T>): Promise<T> {
  await vi.advanceTimersByTimeAsync(300);
  return request;
}

describe('mock adapter current frontend contracts', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2030-01-02T00:00:00.000Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns current quality-gate checks and section reportability for research jobs', async () => {
    const adapter = await createTestMockAdapter();
    const created = await resolveMockRequest(adapter.post<JobCreatedResponse>('/api/v1/research/jobs', {
      user_request: '比较三个深度调研工具',
      source_mode: 'web',
    }));
    const job = await resolveMockRequest(
      adapter.get<ResearchJob>(`/api/v1/research/jobs/${created.job_id}`),
    );

    expect(job.result?.quality_gate?.checks).toMatchObject({
      grounding: true,
      citation_validation: true,
      report_quality: { status: 'passed' },
      core_claim_support: true,
      semantic_review_complete: true,
      report_completeness: true,
      fallback_sections_absent: true,
    });
    expect(job.result?.evidence_native_blueprint?.sections).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          section_id: 'horizontal_compare',
          reportability: expect.objectContaining({ core_section: true, status: 'ready' }),
        }),
      ]),
    );
    expect(job.result?.documents?.every((document) => document.section_brief?.reportability)).toBe(true);
  });

  it('computes an enabled channel next_push_at from the current date and selected time', async () => {
    const adapter = await createTestMockAdapter();
    const config = await resolveMockRequest(adapter.put<ChannelPushConfig>(
      '/api/v1/channels/feishu%3Aoc_agent_learning_demo/push-config',
      { enabled: true, push_time: '10:30', timezone: 'Asia/Shanghai' },
    ));

    expect(config.next_push_at).toBe('2030-01-03 10:30');
  });
});
