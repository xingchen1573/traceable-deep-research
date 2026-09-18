import { create } from 'zustand';
import { useUIStore } from './uiStore';

interface JobState {
  trackedJobs: string[];
  addJob: (jobId: string) => void;
  removeJob: (jobId: string) => void;
}

export const useJobStore = create<JobState>((set) => ({
  trackedJobs: [],
  addJob: (jobId) =>
    set((s) => {
      if (s.trackedJobs.includes(jobId)) return s;
      useUIStore.getState().setJobDrawerOpen(true);
      return { trackedJobs: [jobId, ...s.trackedJobs] };
    }),
  removeJob: (jobId) =>
    set((s) => ({
      trackedJobs: s.trackedJobs.filter((id) => id !== jobId),
    })),
}));
