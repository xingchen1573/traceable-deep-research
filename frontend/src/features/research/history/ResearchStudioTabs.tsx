import { Button } from '@/components/controls/Button';

export type ResearchStudioTab = 'create' | 'history' | 'detail';

interface ResearchStudioTabsProps {
  active: ResearchStudioTab;
  onChange: (tab: ResearchStudioTab) => void;
}

export function ResearchStudioTabs({ active, onChange }: ResearchStudioTabsProps) {
  return (
    <div className="flex gap-2">
      <Button variant={active === 'create' ? 'primary' : 'secondary'} size="sm" onClick={() => onChange('create')}>
        新建调研
      </Button>
      <Button variant={active === 'history' ? 'primary' : 'secondary'} size="sm" onClick={() => onChange('history')}>
        历史调研
      </Button>
      {active === 'detail' && (
        <Button variant="primary" size="sm" onClick={() => onChange('detail')}>
          调研详情
        </Button>
      )}
    </div>
  );
}
