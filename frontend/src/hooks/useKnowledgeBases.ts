import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useJobStore } from '@/stores';

const KB_KEY = 'knowledge-bases';

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: () => api.system.health() });
}

export function useCapabilities() {
  return useQuery({ queryKey: ['capabilities'], queryFn: () => api.system.capabilities() });
}

export function useRuntimeConfig() {
  return useQuery({ queryKey: ['runtime-config'], queryFn: () => api.system.runtimeConfig() });
}

export function useKnowledgeBases() {
  return useQuery({
    queryKey: [KB_KEY, 'list'],
    queryFn: () => api.knowledgeBases.list(),
  });
}

export function useKnowledgeBase(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic],
    queryFn: () => api.knowledgeBases.get(topic),
    enabled: !!topic,
  });
}

export function useKnowledgeBaseSummary(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'summary'],
    queryFn: () => api.knowledgeBases.getSummary(topic),
    enabled: !!topic,
  });
}

export function useActiveKnowledgeBase() {
  return useQuery({
    queryKey: [KB_KEY, 'active'],
    queryFn: () => api.knowledgeBases.getActive(),
  });
}

export function useSetActiveKnowledgeBase() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ topic, threadId }: { topic: string; threadId?: string }) =>
      api.knowledgeBases.setActive(topic, threadId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [KB_KEY, 'active'] });
      queryClient.invalidateQueries({ queryKey: [KB_KEY, 'list'] });
    },
  });
}

export function useDocuments(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'documents'],
    queryFn: () => api.knowledgeBases.listDocuments(topic),
    enabled: !!topic,
  });
}

export function useDocument(topic: string, documentId: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'documents', documentId],
    queryFn: () => api.knowledgeBases.getDocument(topic, documentId),
    enabled: !!topic && !!documentId,
  });
}

export function useQueryKnowledge() {
  return useMutation({
    mutationFn: (params: {
      topic: string;
      question: string;
      backend?: string;
      mode?: string;
    }) => api.knowledgeBases.query(params.topic, params.question, params.backend as any, params.mode as any),
  });
}

export function useIndexStatus(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'index'],
    queryFn: () => api.knowledgeBases.getIndex(topic),
    enabled: !!topic,
  });
}

export function useGraphStats(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'graph'],
    queryFn: () => api.knowledgeBases.getGraph(topic),
    enabled: !!topic,
  });
}

export function useGraphNode(topic: string, entityId: string | null) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'graph', 'node', entityId],
    queryFn: () => api.knowledgeBases.getGraphNode(topic, entityId as string),
    enabled: !!topic && !!entityId,
    // Graph details only change when the index changes. Keep visited nodes in
    // memory for the session; index jobs invalidate the knowledge-base cache.
    staleTime: Infinity,
    gcTime: Infinity,
  });
}

export function useVectorStats(topic: string) {
  return useQuery({
    queryKey: [KB_KEY, topic, 'vectors'],
    queryFn: () => api.knowledgeBases.getVectors(topic),
    enabled: !!topic,
  });
}

export function useCreateIndexJob() {
  const queryClient = useQueryClient();
  const addJob = useJobStore((s) => s.addJob);
  return useMutation({
    mutationFn: ({ topic, force = false }: { topic: string; force?: boolean }) => api.knowledgeBases.createIndexJob(topic, force),
    onSuccess: (data) => {
      addJob(data.job_id);
      queryClient.invalidateQueries({ queryKey: [KB_KEY, data.job_id] });
    },
  });
}

export function useDeleteIndex() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (topic: string) => api.knowledgeBases.deleteIndex(topic),
    onSuccess: (_, topic) => {
      queryClient.invalidateQueries({ queryKey: [KB_KEY, topic, 'index'] });
      queryClient.removeQueries({ queryKey: [KB_KEY, topic, 'graph'] });
      queryClient.invalidateQueries({ queryKey: [KB_KEY, 'list'] });
    },
  });
}
