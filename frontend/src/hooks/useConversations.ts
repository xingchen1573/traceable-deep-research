import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

const CONVERSATIONS_KEY = 'conversations';

export function useConversations(limit = 50, offset = 0) {
  return useQuery({
    queryKey: [CONVERSATIONS_KEY, 'list', limit, offset],
    queryFn: () => api.conversations.list(limit, offset),
  });
}

export function useConversation(threadId: string) {
  return useQuery({
    queryKey: [CONVERSATIONS_KEY, threadId],
    queryFn: () => api.conversations.get(threadId),
    enabled: !!threadId,
  });
}

export function useConversationMessages(threadId: string, limit = 200) {
  return useQuery({
    queryKey: [CONVERSATIONS_KEY, threadId, 'messages', limit],
    queryFn: () => api.conversations.getMessages(threadId, limit),
    enabled: !!threadId,
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (threadId: string) => api.conversations.delete(threadId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [CONVERSATIONS_KEY] });
    },
  });
}
