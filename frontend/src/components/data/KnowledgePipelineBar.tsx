import { CheckCircle2, Circle, FileText, Database, Network, MessageSquare, ClipboardCheck } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface PipelineStatus {
  documents: boolean;
  metadata: boolean;
  index: boolean;
  vectors: boolean;
  qa: boolean;
  evaluation: boolean;
}

interface KnowledgePipelineBarProps {
  status: PipelineStatus;
}

const steps: Array<{ key: keyof PipelineStatus; label: string; icon: React.ReactNode }> = [
  { key: 'documents', label: '文档', icon: <FileText className="h-4 w-4" /> },
  { key: 'metadata', label: 'Metadata', icon: <Database className="h-4 w-4" /> },
  { key: 'index', label: '索引', icon: <Network className="h-4 w-4" /> },
  { key: 'vectors', label: '向量/图谱', icon: <Database className="h-4 w-4" /> },
  { key: 'qa', label: '问答', icon: <MessageSquare className="h-4 w-4" /> },
  { key: 'evaluation', label: '评估', icon: <ClipboardCheck className="h-4 w-4" /> },
];

export function KnowledgePipelineBar({ status }: KnowledgePipelineBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-panel border border-border bg-white p-3">
      {steps.map((step, idx) => {
        const active = status[step.key];
        return (
          <div key={step.key} className="flex items-center">
            <div
              className={cn(
                'flex items-center gap-1.5 rounded-control px-2.5 py-1.5 text-xs font-medium',
                active ? 'bg-green-50 text-success' : 'bg-gray-50 text-secondary'
              )}
            >
              {active ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : (
                <Circle className="h-3.5 w-3.5" />
              )}
              {step.icon}
              {step.label}
            </div>
            {idx < steps.length - 1 && (
              <span className="mx-1 text-border">→</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
