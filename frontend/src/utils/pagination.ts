export type PaginationItem = number | 'ellipsis-left' | 'ellipsis-right';

export function getPaginationItems(currentPage: number, totalPages: number): PaginationItem[] {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, index) => index + 1);
  }

  const pages = [...new Set([1, currentPage - 1, currentPage, currentPage + 1, totalPages])]
    .filter((page) => page >= 1 && page <= totalPages)
    .sort((left, right) => left - right);

  const items: PaginationItem[] = [];
  pages.forEach((page, index) => {
    const previous = pages[index - 1];
    if (previous && page - previous === 2) {
      items.push(previous + 1);
    } else if (previous && page - previous > 2) {
      items.push(index === 1 ? 'ellipsis-left' : 'ellipsis-right');
    }
    items.push(page);
  });

  return items;
}
