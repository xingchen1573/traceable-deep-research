import { describe, expect, it } from 'vitest';
import { getJobPollingInterval } from './jobPolling';

describe('getJobPollingInterval', () => {
  it('does not poll again when a historical job cannot be loaded', () => {
    expect(getJobPollingInterval()).toBe(false);
  });

  it('keeps active and approval-waiting jobs fresh', () => {
    expect(getJobPollingInterval('running')).toBe(2000);
    expect(getJobPollingInterval('waiting_approval')).toBe(2000);
  });

  it('stops polling terminal jobs', () => {
    expect(getJobPollingInterval('succeeded')).toBe(false);
    expect(getJobPollingInterval('failed')).toBe(false);
  });
});
