import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { ModelConfigUpdateRequest, ResearchRole, ResearchRoleKeyCheckRequest } from '@/types';

export function useModelConfig() {
  return useQuery({
    queryKey: ['models', 'config'],
    queryFn: () => api.models.getConfig(),
  });
}

export function useModelCheck() {
  return useMutation({
    mutationFn: (targets: string[]) => api.models.check(targets),
  });
}

export function useUpdateModelConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: ModelConfigUpdateRequest) => api.models.updateConfig(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models', 'config'] });
      queryClient.invalidateQueries({ queryKey: ['models', 'providers'] });
    },
  });
}

function useInvalidateModelConfig() {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: ['models', 'config'] });
    queryClient.invalidateQueries({ queryKey: ['models', 'providers'] });
  };
}

export function useAddResearchRoleKey() {
  const invalidate = useInvalidateModelConfig();
  return useMutation({
    mutationFn: ({ role, apiKey }: { role: ResearchRole; apiKey: string }) =>
      api.models.addResearchRoleKey(role, apiKey),
    onSuccess: invalidate,
  });
}

export function useDeleteResearchRoleKey() {
  const invalidate = useInvalidateModelConfig();
  return useMutation({
    mutationFn: ({ role, keyId }: { role: ResearchRole; keyId: string }) =>
      api.models.deleteResearchRoleKey(role, keyId),
    onSuccess: invalidate,
  });
}

export function useCheckResearchRoleCandidate() {
  return useMutation({
    mutationFn: ({ role, payload }: { role: ResearchRole; payload: ResearchRoleKeyCheckRequest }) =>
      api.models.checkResearchRoleCandidate(role, payload),
  });
}

export function useCheckResearchRoleKey() {
  return useMutation({
    mutationFn: ({ role, keyId, payload }: { role: ResearchRole; keyId: string; payload?: ResearchRoleKeyCheckRequest }) =>
      api.models.checkResearchRoleKey(role, keyId, payload),
  });
}

export function useProviders() {
  return useQuery({
    queryKey: ['models', 'providers'],
    queryFn: () => api.models.listProviders(),
  });
}
