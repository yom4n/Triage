"use client";

/**
 * React error boundary that reports render-tree crashes to the FDE triage
 * backend, AND installs the window-level crash hooks (crashReporter.ts) on
 * mount -- so wrapping your app in this one component is the entire
 * integration. See this package's README for the two-line setup.
 *
 * Why a separate error boundary at all: `window.onerror` / `unhandledrejection`
 * (crashReporter.ts) do NOT see errors thrown during React's own render --
 * React catches those itself before they reach the window, and only a
 * class-component error boundary's `componentDidCatch` can observe them.
 * This is a real React constraint, not a limitation of this SDK -- it's
 * exactly why Sentry/Bugsnag's React packages ship an error boundary too.
 */

import React from "react";
import { installCrashReporter, reportReactCrash, type CrashReporterOptions } from "./crashReporter";

interface CrashBoundaryProps {
  children: React.ReactNode;
  options: CrashReporterOptions;
  /** Rendered in place of the crashed subtree. Defaults to a plain message. */
  fallback?: React.ReactNode;
}

interface CrashBoundaryState {
  hasError: boolean;
}

export class CrashBoundary extends React.Component<CrashBoundaryProps, CrashBoundaryState> {
  constructor(props: CrashBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidMount(): void {
    // Registering here (not at module scope) ties installation to this
    // component actually being mounted client-side -- avoids any chance
    // of running during Next.js's server-side render pass.
    installCrashReporter(this.props.options);
  }

  static getDerivedStateFromError(): CrashBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    reportReactCrash(this.props.options, error, info.componentStack ?? null);
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return this.props.fallback ?? <p role="alert">Something went wrong.</p>;
    }
    return this.props.children;
  }
}
