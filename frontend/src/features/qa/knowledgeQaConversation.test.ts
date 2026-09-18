import { describe, expect, it } from 'vitest';
import {
  buildKnowledgeQueryHistory,
  knowledgeQaFailureAnswer,
  sanitizeKnowledgeQaAnswer,
} from './knowledgeQaConversation';

const leakedAnswer = `当前知识库：harness

问题：请基于同一知识库和以下最近对话上下文回答用户的新问题。新问题如果包含“它、这个、上面、继续”等指代，请结合上下文理解；最终答案只回答新问题。

上一轮 1
用户：Harness Engineering 是什么？
助手：它是一种系统工程范式。

新问题：主要讲了啥

基于召回片段，可以先确认：
1. Harness Engineering 强调运行环境。 [1]`;

describe('knowledge QA conversation helpers', () => {
  it('removes the internal contextual prompt from a fallback answer', () => {
    const answer = sanitizeKnowledgeQaAnswer(leakedAnswer);

    expect(answer).toContain('问题：主要讲了啥');
    expect(answer).not.toContain('请基于同一知识库和以下最近对话上下文');
    expect(answer).not.toContain('上一轮 1');
  });

  it('builds structured history and sanitizes prior assistant answers', () => {
    const history = buildKnowledgeQueryHistory([
      { role: 'user', question: 'Harness Engineering 是什么？' },
      { role: 'assistant', answer: { answer: leakedAnswer } },
    ]);

    expect(history).toEqual([
      { role: 'user', content: 'Harness Engineering 是什么？' },
      { role: 'assistant', content: sanitizeKnowledgeQaAnswer(leakedAnswer) },
    ]);
  });

  it('shows model unavailability instead of a partial or fabricated answer', () => {
    expect(knowledgeQaFailureAnswer(new Error('模型不可用，无法生成知识库回答。')))
      .toBe('模型不可用，无法生成知识库回答。');
    expect(knowledgeQaFailureAnswer(new Error('connection reset')))
      .toBe('回答生成失败，请重试。');
  });
});
