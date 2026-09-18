import { describe, expect, it } from 'vitest';
import { JobSummary } from '@/types';
import { isTerminalJob, jobTarget, jobTitle, jobTypeLabel } from './jobDisplay';

function makeJob(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    job_id: 'job_1',
    job_type: 'research',
    status: 'running',
    created_at: '2026-08-28 10:00:00',
    updated_at: '2026-08-28 10:01:00',
    ...overrides,
  };
}

describe('job display helpers', () => {
  it('prefers a user-facing request for the title', () => {
    expect(jobTitle(makeJob({ input: { user_request: '调研 Agent Harness' } }))).toBe('调研 Agent Harness');
  });

  it('falls back to a readable job type label', () => {
    expect(jobTitle(makeJob({ job_type: 'knowledge_index' }))).toBe('知识库索引');
    expect(jobTypeLabel('custom_batch_job')).toBe('custom batch job');
  });

  it('routes research jobs to research and other jobs to observability', () => {
    expect(jobTarget(makeJob())).toBe('/research?job_id=job_1');
    expect(jobTarget(makeJob({ job_type: 'knowledge_index' }))).toBe('/observability?job_id=job_1');
  });

  it('only treats final statuses as terminal', () => {
    expect(isTerminalJob(makeJob({ status: 'succeeded' }))).toBe(true);
    expect(isTerminalJob(makeJob({ status: 'waiting_approval' }))).toBe(false);
  });
});
