import { Component, ReactNode } from 'react';
import { ErrorPanel } from '@/components/data/ErrorPanel';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: unknown;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: unknown): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6">
          <ErrorPanel
            error={this.state.error}
            onRetry={() => window.location.reload()}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
