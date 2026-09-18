import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryProvider } from './app/QueryProvider';
import { ToastProvider } from '@/components/feedback/Toast';
import { App } from './app/App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryProvider>
      <ToastProvider>
        <App />
      </ToastProvider>
    </QueryProvider>
  </React.StrictMode>
);
