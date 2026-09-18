import { useState } from 'react';
import { Plus, RotateCcw, Trash2 } from 'lucide-react';
import { Button, IconButton } from '@/components/controls/Button';
import { ResearchPlanSection } from '@/types';

interface ResearchPlanReviewProps {
  sections: ResearchPlanSection[];
  isRevising: boolean;
  error?: string;
  onRedo: (index: number) => void;
  onDelete: (index: number) => void;
  onAdd: (title: string) => void;
}

export function ResearchPlanReview({
  sections,
  isRevising,
  error,
  onRedo,
  onDelete,
  onAdd,
}: ResearchPlanReviewProps) {
  const [newSectionTitle, setNewSectionTitle] = useState('');
  const questionCount = sections.reduce((total, section) => total + (section.questions?.length || 0), 0);

  const handleAdd = () => {
    const title = newSectionTitle.trim();
    if (!title) return;
    onAdd(title);
    setNewSectionTitle('');
  };

  return (
    <section className="mt-5 border-t border-border pt-4" aria-labelledby="research-plan-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id="research-plan-title" className="text-base font-semibold text-primary">调研计划</h3>
          <p className="mt-1 text-xs text-secondary">{sections.length} 个章节 · {questionCount} 个研究问题</p>
        </div>
        {isRevising && <span className="text-xs text-secondary">正在更新计划...</span>}
      </div>

      {error && <p className="mt-3 text-sm text-danger">{error}</p>}

      <div className="mt-4 divide-y divide-border border-y border-border">
        {sections.map((section, sectionIndex) => (
          <article key={section.id || `${section.title}-${sectionIndex}`} className="py-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h4 className="text-sm font-semibold text-primary">{section.title}</h4>
                {section.intent && <p className="mt-1 text-sm text-secondary">{section.intent}</p>}
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <IconButton
                  icon={<RotateCcw className="h-4 w-4" />}
                  label={`重新生成${section.title}`}
                  disabled={isRevising}
                  onClick={() => onRedo(sectionIndex)}
                />
                <IconButton
                  icon={<Trash2 className="h-4 w-4" />}
                  label={`删除${section.title}`}
                  disabled={isRevising || sections.length <= 1}
                  onClick={() => onDelete(sectionIndex)}
                />
              </div>
            </div>

            {section.questions?.length ? (
              <ol className="mt-3 space-y-3">
                {section.questions.map((question, questionIndex) => {
                  const questionText = question.text || question.question || '';
                  return (
                    <li key={question.id || `${sectionIndex}-${questionIndex}`} className="grid grid-cols-[24px_minmax(0,1fr)] gap-2 text-sm">
                      <span className="text-secondary">{questionIndex + 1}.</span>
                      <div>
                        <p className="text-primary">{questionText}</p>
                        {question.success_criteria?.length ? (
                          <ul className="mt-1 space-y-1 text-xs text-secondary">
                            {question.success_criteria.map((criterion, criterionIndex) => (
                              <li key={`${criterion}-${criterionIndex}`}>· {criterion}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
              </ol>
            ) : (
              <p className="mt-3 text-xs text-secondary">该章节用于汇总其他研究问题的结论。</p>
            )}
          </article>
        ))}
      </div>

      <div className="mt-4 flex flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={newSectionTitle}
          onChange={(event) => setNewSectionTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              handleAdd();
            }
          }}
          placeholder="新增研究方向"
          aria-label="新增研究方向"
          disabled={isRevising}
          className="min-w-0 flex-1 rounded-control border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none disabled:opacity-50"
        />
        <Button
          type="button"
          variant="secondary"
          disabled={isRevising || !newSectionTitle.trim()}
          onClick={handleAdd}
          icon={<Plus className="mr-1.5 h-4 w-4" />}
        >
          新增章节
        </Button>
      </div>
    </section>
  );
}
