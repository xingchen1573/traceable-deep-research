export function shouldUseMock(): boolean {
  const envMock = import.meta.env.VITE_USE_MOCK === 'true';
  const runtimeMock =
    typeof window !== 'undefined' && (window as unknown as Record<string, boolean>).__USE_MOCK__ === true;
  return envMock || runtimeMock;
}

export function getBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL ?? '';
}
