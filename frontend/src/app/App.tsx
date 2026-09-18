import { createBrowserRouter, RouterProvider, Navigate, useRouteError } from 'react-router-dom';
import { AppShell } from './AppShell';
import { ErrorBoundary } from './ErrorBoundary';
import { ErrorPanel } from '@/components/data/ErrorPanel';
import { DashboardPage } from '@/features/dashboard';
import { ChannelsPage } from '@/features/channels';
import { ResearchStudioPage } from '@/features/research';
import { KnowledgeBasesPage, KnowledgeBaseDetailPage } from '@/features/knowledge-bases';
import { KnowledgeQAPage } from '@/features/qa';
import { IndexGraphPage } from '@/features/index-graph';
import { EvaluationsPage } from '@/features/evaluations';
import { ObservabilityPage } from '@/features/observability';
import { ModelsPage } from '@/features/models';
import { JobsPage } from '@/features/jobs';

function RouteError() {
  const error = useRouteError();
  console.error('Route error:', error);
  return (
    <div className="p-6">
      <ErrorPanel error={error} onRetry={() => window.location.reload()} />
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: 'channels', element: <ChannelsPage /> },
      { path: 'research', element: <ResearchStudioPage /> },
      { path: 'knowledge-bases', element: <KnowledgeBasesPage /> },
      { path: 'knowledge-bases/:topic', element: <KnowledgeBaseDetailPage /> },
      { path: 'qa', element: <KnowledgeQAPage /> },
      { path: 'index-graph', element: <IndexGraphPage /> },
      { path: 'evaluations', element: <EvaluationsPage /> },
      { path: 'observability', element: <ObservabilityPage /> },
      { path: 'jobs', element: <JobsPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]);

export function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}
