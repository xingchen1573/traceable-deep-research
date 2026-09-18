import { describe, expect, it } from 'vitest';
import { shouldContainWheel } from './scrollBoundary';

describe('shouldContainWheel', () => {
  it('contains downward wheel input at the bottom boundary', () => {
    expect(shouldContainWheel({
      delta: 120,
      scrollPosition: 500,
      scrollSize: 1000,
      clientSize: 500,
    })).toBe(true);
  });

  it('contains upward wheel input at the top boundary', () => {
    expect(shouldContainWheel({
      delta: -120,
      scrollPosition: 0,
      scrollSize: 1000,
      clientSize: 500,
    })).toBe(true);
  });

  it('allows the inner container to scroll while it has room', () => {
    expect(shouldContainWheel({
      delta: 120,
      scrollPosition: 250,
      scrollSize: 1000,
      clientSize: 500,
    })).toBe(false);
  });

  it('does not trap wheel input on a non-scrollable element', () => {
    expect(shouldContainWheel({
      delta: 120,
      scrollPosition: 0,
      scrollSize: 500,
      clientSize: 500,
    })).toBe(false);
  });

  it('uses the same boundary rules for horizontal scrolling', () => {
    expect(shouldContainWheel({
      delta: 80,
      scrollPosition: 300,
      scrollSize: 600,
      clientSize: 300,
    })).toBe(true);
  });
});
