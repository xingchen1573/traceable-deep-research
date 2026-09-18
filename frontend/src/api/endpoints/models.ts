import { ApiAdapter } from '../adapters';
import {
  ModelCheckKeyResult,
  ModelCheckResponse,
  ModelConfigResponse,
  ModelConfigUpdateRequest,
  ModelConfigUpdateResponse,
  ProvidersResponse,
  ResearchRole,
  ResearchRoleKeyCheckRequest,
  ResearchRoleKeyMutationResponse,
} from '@/types';

export function modelEndpoints(adapter: ApiAdapter) {
  return {
    getConfig: () => adapter.get<ModelConfigResponse>('/api/v1/models/config'),

    updateConfig: (payload: ModelConfigUpdateRequest) =>
      adapter.put<ModelConfigUpdateResponse>('/api/v1/models/config', payload),

    check: (targets: string[] = ['chat_model', 'embedding'], timeout_seconds = 30) => adapter.post<ModelCheckResponse>('/api/v1/models/check', { targets, timeout_seconds }),

    addResearchRoleKey: (role: ResearchRole, api_key: string) =>
      adapter.post<ResearchRoleKeyMutationResponse>(`/api/v1/models/research-roles/${role}/keys`, { api_key }),

    deleteResearchRoleKey: (role: ResearchRole, keyId: string) =>
      adapter.delete<ResearchRoleKeyMutationResponse>(`/api/v1/models/research-roles/${role}/keys/${keyId}`),

    checkResearchRoleCandidate: (role: ResearchRole, payload: ResearchRoleKeyCheckRequest) =>
      adapter.post<ModelCheckKeyResult>(`/api/v1/models/research-roles/${role}/keys/check`, payload),

    checkResearchRoleKey: (role: ResearchRole, keyId: string, payload: ResearchRoleKeyCheckRequest = {}) =>
      adapter.post<ModelCheckKeyResult>(`/api/v1/models/research-roles/${role}/keys/${keyId}/check`, payload),

    listProviders: () => adapter.get<ProvidersResponse>('/api/v1/providers'),
  };
}
