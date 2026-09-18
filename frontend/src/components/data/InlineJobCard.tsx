import { useJobPolling } from '@/hooks';
import { JobCard } from './JobCard';

// job_service.py::JobService.create() always mints ids shaped job_<YYYYMMDD>_<HHMMSS>_<6 hex
// chars> (backend/services/job_service.py). Matching on that fixed shape lets the frontend
// detect a job_id anywhere in a plain-text assistant reply without requiring the model to
// follow a special marker syntax -- see docs/modules/agent-dispatcher-role-v0.1.md 3.6.
const JOB_ID_PATTERN = /job_\d{8}_\d{6}_[0-9a-f]{6}/;

export function extractJobId(text: string): string | null {
  const match = text.match(JOB_ID_PATTERN);
  return match ? match[0] : null;
}

/** Renders a live-polling JobCard for a job_id found in a chat message. Reuses useJobPolling
 * (the hook that actually implements refetch-until-terminal, despite the design doc's 3.6
 * referring to it as "useJob") so the card auto-updates as the job progresses without the page
 * needing its own polling logic. */
export function InlineJobCard({ jobId }: { jobId: string }) {
  const { data } = useJobPolling(jobId);

  if (!data) {
    return null;
  }

  return (
    <div className="mt-2 max-w-[78%]">
      <JobCard job={data} />
    </div>
  );
}
