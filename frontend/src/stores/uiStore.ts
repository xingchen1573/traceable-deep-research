import { create } from 'zustand';

interface UIState {
  jobDrawerOpen: boolean;
  mobileNavVisible: boolean;
  setJobDrawerOpen: (open: boolean) => void;
  setMobileNavVisible: (visible: boolean) => void;
}

export const useUIStore = create<UIState>((set) => ({
  jobDrawerOpen: false,
  mobileNavVisible: false,
  setJobDrawerOpen: (open) => set({ jobDrawerOpen: open }),
  setMobileNavVisible: (visible) => set({ mobileNavVisible: visible }),
}));
