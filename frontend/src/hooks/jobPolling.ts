import type { JobStatus } from '@/types';

const TERMINAL_JOB_STATUSES: JobStatus[] = ['succeeded', 'failed', 'cancelled', 'interrupted'];

export function getJobPollingInterval(status?: JobStatus): number | false {
  if (!status || TERMINAL_JOB_STATUSES.includes(status)) return false;
  return 2000;
}
