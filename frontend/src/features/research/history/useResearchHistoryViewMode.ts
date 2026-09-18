import { useState } from 'react';

const VIEW_MODE_KEY = 'harness-agent:research-studio-history-view';

export type ResearchHistoryViewMode = 'list' | 'card';

const DEFAULT_VIEW_MODE: ResearchHistoryViewMode = 'list';

function loadViewMode(): ResearchHistoryViewMode {
  if (typeof window === 'undefined') return DEFAULT_VIEW_MODE;
  try {
    const value = window.localStorage.getItem(VIEW_MODE_KEY);
    return value === 'list' || value === 'card' ? value : DEFAULT_VIEW_MODE;
  } catch {
    return DEFAULT_VIEW_MODE;
  }
}

function saveViewMode(mode: ResearchHistoryViewMode) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(VIEW_MODE_KEY, mode);
  } catch {
    // Storage can be unavailable in private/restricted browser contexts.
  }
}

export function useResearchHistoryViewMode(): [ResearchHistoryViewMode, (mode: ResearchHistoryViewMode) => void] {
  const [viewMode, setViewModeState] = useState<ResearchHistoryViewMode>(() => loadViewMode());
  const setViewMode = (mode: ResearchHistoryViewMode) => {
    setViewModeState(mode);
    saveViewMode(mode);
  };
  return [viewMode, setViewMode];
}

export { VIEW_MODE_KEY, loadViewMode, saveViewMode };
