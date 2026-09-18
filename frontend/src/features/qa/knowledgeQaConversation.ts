import type { KnowledgeQueryHistoryMessage } from '@/types';

interface ConversationMessageLike {
  role: 'user' | 'assistant';
  question?: string;
  answer?: { answer?: string };
}

const INTERNAL_CONTEXT_PROMPT = '请基于同一知识库和以下最近对话上下文回答用户的新问题。';

export function sanitizeKnowledgeQaAnswer(answer: string): string {
  if (!answer.includes(INTERNAL_CONTEXT_PROMPT)) return answer;
  return answer.replace(
    /问题：请基于同一知识库和以下最近对话上下文回答用户的新问题。[^\n]*\n\n[\s\S]*?\n\n新问题：([^\n]+)(?=\n\n基于召回片段，可以先确认：)/,
    '问题：$1',
  );
}

export function knowledgeQaFailureAnswer(error: unknown): string {
  const message = error instanceof Error ? error.message.trim() : '';
  return message.includes('模型不可用') ? message : '回答生成失败，请重试。';
}

export function buildKnowledgeQueryHistory(
  messages: ConversationMessageLike[],
): KnowledgeQueryHistoryMessage[] {
  const history = messages.flatMap<KnowledgeQueryHistoryMessage>((message) => {
    if (message.role === 'user' && message.question?.trim()) {
      return [{ role: 'user', content: message.question.trim().slice(0, 500) }];
    }
    if (message.role === 'assistant' && message.answer?.answer?.trim()) {
      return [{
        role: 'assistant',
        content: sanitizeKnowledgeQaAnswer(message.answer.answer).trim().slice(0, 800),
      }];
    }
    return [];
  });
  return history.slice(-6);
}
