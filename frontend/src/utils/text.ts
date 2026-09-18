export function truncateMiddle(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  const side = Math.floor((maxLength - 3) / 2);
  return `${text.slice(0, side)}...${text.slice(-side)}`;
}

export function truncateEnd(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .slice(0, 80);
}
