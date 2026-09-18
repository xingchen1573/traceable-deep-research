import { beforeEach, describe, expect, it } from 'vitest';

class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string) {
    return this.store.has(key) ? this.store.get(key)! : null;
  }
  setItem(key: string, value: string) {
    this.store.set(key, value);
  }
  removeItem(key: string) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}

// The hook guards on `typeof window === 'undefined'` for SSR safety (mirrors
// KnowledgeQAPage's loadQAHistory/saveQAHistory pattern), and vitest's default `node`
// environment has no `window` global -- polyfill just enough of it so the guard passes
// and `window.localStorage` resolves, without pulling in a jsdom dependency.
const memoryStorage = new MemoryStorage();
(globalThis as unknown as { window: { localStorage: MemoryStorage } }).window = { localStorage: memoryStorage };

const { VIEW_MODE_KEY, loadViewMode, saveViewMode } = await import('./useResearchHistoryViewMode');

beforeEach(() => {
  memoryStorage.clear();
});

describe('research history view mode persistence', () => {
  it('defaults to list when nothing is stored', () => {
    expect(loadViewMode()).toBe('list');
  });

  it('round-trips a saved view mode', () => {
    saveViewMode('card');
    expect(loadViewMode()).toBe('card');
    expect(memoryStorage.getItem(VIEW_MODE_KEY)).toBe('card');
  });

  it('falls back to the default on corrupt/unexpected stored values', () => {
    memoryStorage.setItem(VIEW_MODE_KEY, '{"not":"a valid mode"}');
    expect(loadViewMode()).toBe('list');

    memoryStorage.setItem(VIEW_MODE_KEY, 'grid');
    expect(loadViewMode()).toBe('list');
  });
});
