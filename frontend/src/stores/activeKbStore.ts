import { create } from 'zustand';
import { api } from '@/api/client';

interface ActiveKBState {
  activeTopic: string | null;
  isLoading: boolean;
  setActiveTopic: (topic: string | null) => Promise<void>;
  refresh: () => Promise<void>;
}

export const useActiveKBStore = create<ActiveKBState>((set) => ({
  activeTopic: null,
  isLoading: false,
  setActiveTopic: async (topic) => {
    set({ isLoading: true });
    try {
      await api.knowledgeBases.setActive(topic || '');
      set({ activeTopic: topic, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },
  refresh: async () => {
    set({ isLoading: true });
    try {
      const res = await api.knowledgeBases.getActive();
      set({ activeTopic: res.active_topic, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },
}));
