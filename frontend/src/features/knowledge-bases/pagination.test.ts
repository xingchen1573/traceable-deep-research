import { describe, expect, it } from 'vitest';
import { getPaginationItems } from './pagination';

describe('getPaginationItems', () => {
  it('shows every page for a short result set', () => {
    expect(getPaginationItems(1, 4)).toEqual([1, 2, 3, 4]);
  });

  it('keeps the current page and both ends visible', () => {
    expect(getPaginationItems(5, 10)).toEqual([1, 'ellipsis-left', 4, 5, 6, 'ellipsis-right', 10]);
  });

  it('expands a single-page gap instead of showing an ellipsis', () => {
    expect(getPaginationItems(3, 8)).toEqual([1, 2, 3, 4, 'ellipsis-right', 8]);
  });
});
