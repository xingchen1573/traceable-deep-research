import { ApiAdapter } from '../adapters';
import {
  CapabilitiesResponse,
  HealthResponse,
  RuntimeConfigResponse,
} from '@/types';

export function systemEndpoints(adapter: ApiAdapter) {
  return {
    health: () => adapter.get<HealthResponse>('/api/v1/health'),
    capabilities: () => adapter.get<CapabilitiesResponse>('/api/v1/capabilities'),
    runtimeConfig: () => adapter.get<RuntimeConfigResponse>('/api/v1/config/runtime'),
  };
}
