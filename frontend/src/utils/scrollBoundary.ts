interface ScrollMetrics {
  delta: number;
  scrollPosition: number;
  scrollSize: number;
  clientSize: number;
}

const SCROLLABLE_OVERFLOW = new Set(['auto', 'scroll']);

export function shouldContainWheel({
  delta,
  scrollPosition,
  scrollSize,
  clientSize,
}: ScrollMetrics): boolean {
  if (!delta || scrollSize <= clientSize + 1) return false;

  const atStart = scrollPosition <= 1;
  const atEnd = scrollPosition + clientSize >= scrollSize - 1;
  return (delta < 0 && atStart) || (delta > 0 && atEnd);
}

type ScrollAxis = 'x' | 'y';

function findNearestScrollContainer(target: EventTarget | null, axis: ScrollAxis): HTMLElement | null {
  let element = target instanceof Element ? target : null;

  while (element && element !== document.body && element !== document.documentElement) {
    if (element instanceof HTMLElement) {
      const style = window.getComputedStyle(element);
      const overflow = axis === 'y' ? style.overflowY : style.overflowX;
      const scrollSize = axis === 'y' ? element.scrollHeight : element.scrollWidth;
      const clientSize = axis === 'y' ? element.clientHeight : element.clientWidth;
      if (SCROLLABLE_OVERFLOW.has(overflow) && scrollSize > clientSize + 1) {
        return element;
      }
    }
    element = element.parentElement;
  }

  return null;
}

export function installWheelScrollBoundaryGuard(): () => void {
  const handleWheel = (event: WheelEvent) => {
    if (event.defaultPrevented || event.ctrlKey || (!event.deltaX && !event.deltaY)) return;

    const axis: ScrollAxis = Math.abs(event.deltaX) > Math.abs(event.deltaY) ? 'x' : 'y';
    const delta = axis === 'y' ? event.deltaY : event.deltaX;
    const container = findNearestScrollContainer(event.target, axis);
    if (!container) return;

    if (shouldContainWheel({
      delta,
      scrollPosition: axis === 'y' ? container.scrollTop : container.scrollLeft,
      scrollSize: axis === 'y' ? container.scrollHeight : container.scrollWidth,
      clientSize: axis === 'y' ? container.clientHeight : container.clientWidth,
    })) {
      event.preventDefault();
    }
  };

  document.addEventListener('wheel', handleWheel, { capture: true, passive: false });
  return () => document.removeEventListener('wheel', handleWheel, { capture: true });
}
