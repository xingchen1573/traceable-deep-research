import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { TopNav } from '@/components/layout/TopNav';
import { JobDrawer } from '@/components/layout/JobDrawer';
import { MobileNav } from '@/components/layout/MobileNav';
import { installWheelScrollBoundaryGuard } from '@/utils/scrollBoundary';

export function AppShell() {
  useEffect(() => installWheelScrollBoundaryGuard(), []);

  return (
    <div className="min-h-screen bg-page">
      <TopNav />
      <main className="pb-16 md:pb-0">
        <div className="mx-auto max-w-[1240px] p-4 md:p-6">
          <Outlet />
        </div>
      </main>
      <JobDrawer />
      <MobileNav />
    </div>
  );
}
