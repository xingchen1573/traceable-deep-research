import { useEffect, useMemo, useRef, useState } from 'react';
import { BookOpen, ChevronDown, FileText, FolderTree } from 'lucide-react';
import type { KnowledgeResearchItem, KnowledgeTreeResponse } from '@/types';
import { cn } from '@/utils/cn';

export type KnowledgeScopeType = 'knowledge_base' | 'theme' | 'research';

export interface KnowledgeScopeOption {
  key: string;
  type: KnowledgeScopeType;
  id: string;
  title: string;
  breadcrumb: string;
  depth: 0 | 1 | 2;
  researchIds: string[];
  indexStatus?: string;
}

interface KnowledgeScopeSelectorProps {
  value: string | null;
  options: KnowledgeScopeOption[];
  onChange: (option: KnowledgeScopeOption) => void;
  placeholder?: string;
}

const TYPE_LABELS: Record<KnowledgeScopeType, string> = {
  knowledge_base: '知识库',
  theme: '主题',
  research: '调研',
};

const TYPE_ICONS = {
  knowledge_base: BookOpen,
  theme: FolderTree,
  research: FileText,
};

export function knowledgeScopeKey(type: KnowledgeScopeType, id: string) {
  return `${type}:${id}`;
}

export function buildKnowledgeScopeOptions(tree?: KnowledgeTreeResponse): KnowledgeScopeOption[] {
  if (!tree) return [];
  const options: KnowledgeScopeOption[] = [];

  for (const knowledgeBase of tree.knowledge_bases) {
    const directResearch = activeResearch(knowledgeBase.research_items);
    const themedResearch = knowledgeBase.themes.flatMap((theme) => activeResearch(theme.research_items));
    options.push({
      key: knowledgeScopeKey('knowledge_base', knowledgeBase.id),
      type: 'knowledge_base',
      id: knowledgeBase.id,
      title: knowledgeBase.title,
      breadcrumb: knowledgeBase.title,
      depth: 0,
      researchIds: [...directResearch, ...themedResearch].map((item) => item.id),
    });

    for (const research of directResearch) {
      options.push(researchOption(research, knowledgeBase.title, 1));
    }

    for (const theme of knowledgeBase.themes) {
      const researchItems = activeResearch(theme.research_items);
      options.push({
        key: knowledgeScopeKey('theme', theme.id),
        type: 'theme',
        id: theme.id,
        title: theme.title,
        breadcrumb: `${knowledgeBase.title} / ${theme.title}`,
        depth: 1,
        researchIds: researchItems.map((item) => item.id),
      });
      for (const research of researchItems) {
        options.push(researchOption(research, `${knowledgeBase.title} / ${theme.title}`, 2));
      }
    }
  }

  return options;
}

function activeResearch(items: KnowledgeResearchItem[]) {
  return items.filter((item) => item.status !== 'superseded');
}

function researchOption(
  research: KnowledgeResearchItem,
  parentBreadcrumb: string,
  depth: 1 | 2,
): KnowledgeScopeOption {
  return {
    key: knowledgeScopeKey('research', research.id),
    type: 'research',
    id: research.id,
    title: research.title,
    breadcrumb: `${parentBreadcrumb} / ${research.title}`,
    depth,
    researchIds: [research.id],
    indexStatus: research.index_status?.status,
  };
}

export function KnowledgeScopeSelector({
  value,
  options,
  onChange,
  placeholder = '选择知识范围',
}: KnowledgeScopeSelectorProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const selected = useMemo(() => options.find((option) => option.key === value), [options, value]);

  useEffect(() => {
    if (!open) return;
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', closeOnOutsideClick);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [open]);

  const SelectedIcon = selected ? TYPE_ICONS[selected.type] : BookOpen;

  return (
    <div ref={containerRef} className="relative min-w-0">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-11 w-full items-center justify-between gap-3 rounded-control border border-border bg-white px-3 py-2 text-left text-sm text-primary hover:bg-page focus:border-accent focus:outline-none"
      >
        <span className="flex min-w-0 items-center gap-2.5">
          <SelectedIcon className="h-4 w-4 flex-shrink-0 text-accent" />
          <span className="min-w-0">
            <span className="block truncate font-medium">{selected?.title || placeholder}</span>
            {selected && <span className="block truncate text-xs text-secondary">{selected.breadcrumb}</span>}
          </span>
        </span>
        <ChevronDown className={cn('h-4 w-4 flex-shrink-0 text-secondary transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute z-30 mt-1 max-h-80 w-full min-w-0 overflow-y-auto rounded-panel border border-border bg-white py-1 shadow-lg sm:min-w-[20rem]"
        >
          {options.length === 0 && <div className="px-3 py-2 text-sm text-secondary">暂无知识范围</div>}
          {options.map((option) => {
            const Icon = TYPE_ICONS[option.type];
            const isSelected = option.key === value;
            return (
              <button
                key={option.key}
                type="button"
                role="option"
                aria-selected={isSelected}
                onClick={() => {
                  onChange(option);
                  setOpen(false);
                }}
                className={cn(
                  'flex min-h-10 w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-page',
                  isSelected && 'bg-page text-accent',
                )}
                style={{ paddingLeft: `${12 + option.depth * 22}px` }}
              >
                <Icon className={cn('h-4 w-4 flex-shrink-0', isSelected ? 'text-accent' : 'text-secondary')} />
                <span className="min-w-0 flex-1 truncate font-medium">{option.title}</span>
                <span className="flex-shrink-0 text-[11px] text-secondary">{TYPE_LABELS[option.type]}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
