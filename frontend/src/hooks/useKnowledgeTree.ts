import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { KnowledgeScopeQueryRequest, KnowledgeTreeAction } from '@/types';
import { useJobStore } from '@/stores';

export const KNOWLEDGE_TREE_KEY = 'knowledge-tree';

export function useKnowledgeTree() {
  return useQuery({ queryKey: [KNOWLEDGE_TREE_KEY], queryFn: () => api.knowledgeTree.list(true) });
}

export function useKnowledgeTrash(enabled = true) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'trash'],
    queryFn: () => api.knowledgeTree.trash(),
    enabled,
  });
}

export function useKnowledgeOverview(nodeType?: 'knowledge_base' | 'theme', nodeId?: string) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'overview', nodeType, nodeId],
    queryFn: () => api.knowledgeTree.overview(nodeType!, nodeId!),
    enabled: Boolean(nodeType && nodeId),
  });
}

export function useResearchDocument(researchId?: string, filename?: string) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'document', researchId, filename],
    queryFn: () => api.knowledgeTree.document(researchId!, filename!),
    enabled: Boolean(researchId && filename),
  });
}

export function useKnowledgePreview() {
  return useMutation({
    mutationFn: ({ action, payload }: { action: KnowledgeTreeAction; payload: Record<string, unknown> }) =>
      api.knowledgeTree.createPreview(action, payload),
  });
}

export function useConfirmKnowledgePreview() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (previewId: string) => api.knowledgeTree.confirmPreview(previewId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [KNOWLEDGE_TREE_KEY] }),
  });
}

export function useKnowledgeScopeQuery() {
  return useMutation({
    mutationFn: ({
      scopeType,
      scopeId,
      request,
    }: {
      scopeType: 'knowledge_base' | 'theme' | 'research';
      scopeId: string;
      request: KnowledgeScopeQueryRequest;
    }) => api.knowledgeTree.queryScope(scopeType, scopeId, request),
  });
}

export function useKnowledgeScopeIndex(scopeType?: 'knowledge_base' | 'theme' | 'research', scopeId?: string) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'scope', scopeType, scopeId, 'index'],
    queryFn: () => api.knowledgeTree.scopeIndex(scopeType!, scopeId!),
    enabled: Boolean(scopeType && scopeId),
  });
}

export function useKnowledgeScopeGraph(scopeType?: 'knowledge_base' | 'theme' | 'research', scopeId?: string) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'scope', scopeType, scopeId, 'graph'],
    queryFn: () => api.knowledgeTree.scopeGraph(scopeType!, scopeId!),
    enabled: Boolean(scopeType && scopeId),
  });
}

export function useKnowledgeScopeGraphNode(
  scopeType?: 'knowledge_base' | 'theme' | 'research',
  scopeId?: string,
  entityId?: string | null,
) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'scope', scopeType, scopeId, 'graph', 'node', entityId],
    queryFn: () => api.knowledgeTree.scopeGraphNode(scopeType!, scopeId!, entityId!),
    enabled: Boolean(scopeType && scopeId && entityId),
    staleTime: Infinity,
    gcTime: Infinity,
  });
}

export function useKnowledgeScopeVectors(scopeType?: 'knowledge_base' | 'theme' | 'research', scopeId?: string) {
  return useQuery({
    queryKey: [KNOWLEDGE_TREE_KEY, 'scope', scopeType, scopeId, 'vectors'],
    queryFn: () => api.knowledgeTree.scopeVectors(scopeType!, scopeId!),
    enabled: Boolean(scopeType && scopeId),
  });
}

export function useRetryResearchIndex() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (researchId: string) => api.knowledgeTree.retryIndex(researchId, true),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [KNOWLEDGE_TREE_KEY] }),
  });
}

export function useRebuildKnowledgeScopeIndexes() {
  const queryClient = useQueryClient();
  const addJob = useJobStore((state) => state.addJob);
  return useMutation({
    mutationFn: (researchIds: string[]) => Promise.all(
      researchIds.map((researchId) => api.knowledgeTree.retryIndex(researchId, true)),
    ),
    onSuccess: (jobs) => {
      jobs.forEach((job) => addJob(job.job_id));
      queryClient.invalidateQueries({ queryKey: [KNOWLEDGE_TREE_KEY] });
    },
  });
}

export function useDeleteResearchIndex() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (researchId: string) => api.knowledgeTree.deleteResearchIndex(researchId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: [KNOWLEDGE_TREE_KEY] }),
  });
}
