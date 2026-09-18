import { describe, expect, it } from 'vitest';
import { buildKnowledgeScopeOptions } from './KnowledgeScopeSelector';
import type { KnowledgeTreeResponse } from '@/types';

describe('buildKnowledgeScopeOptions', () => {
  it('keeps knowledge base, theme, and research items in tree order', () => {
    const tree: KnowledgeTreeResponse = {
      schema_version: 1,
      trash_count: 0,
      legacy: [],
      knowledge_bases: [
        {
          type: 'knowledge_base',
          id: 'kb_1',
          knowledge_base_id: 'kb_1',
          title: 'Agent 学习',
          path: 'Agent 学习',
          research_count: 2,
          research_items: [],
          themes: [
            {
              type: 'theme',
              id: 'theme_1',
              theme_id: 'theme_1',
              knowledge_base_id: 'kb_1',
              title: 'Harness',
              path: 'Agent 学习/Harness',
              research_count: 2,
              research_items: [
                {
                  type: 'research',
                  id: 'research_1',
                  research_id: 'research_1',
                  knowledge_base_id: 'kb_1',
                  theme_id: 'theme_1',
                  title: 'DeepSeek Harness',
                  path: 'Agent 学习/Harness/DeepSeek Harness',
                  status: 'active',
                  documents: [],
                  document_count: 0,
                  index_status: { status: 'ready' },
                },
                {
                  type: 'research',
                  id: 'research_old',
                  research_id: 'research_old',
                  knowledge_base_id: 'kb_1',
                  theme_id: 'theme_1',
                  title: '旧调研',
                  path: 'Agent 学习/Harness/旧调研',
                  status: 'superseded',
                  documents: [],
                  document_count: 0,
                },
              ],
            },
          ],
        },
      ],
    };

    const options = buildKnowledgeScopeOptions(tree);

    expect(options.map((option) => [option.type, option.title, option.depth])).toEqual([
      ['knowledge_base', 'Agent 学习', 0],
      ['theme', 'Harness', 1],
      ['research', 'DeepSeek Harness', 2],
    ]);
    expect(options[0].researchIds).toEqual(['research_1']);
    expect(options[1].breadcrumb).toBe('Agent 学习 / Harness');
    expect(options[2].indexStatus).toBe('ready');
  });
});
