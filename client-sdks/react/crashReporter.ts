/**
 * Minimal crash-capture hook for a React / Next.js app -- Phase 1 of the
 * FDE triage engine's monitoring loop (see the FDE-Project repo's README,
 * "Auto-ticketing from a running app's own crashes").
 *
 * Registers `window.onerror` + `unhandledrejection` listeners that POST a
 * normalized crash payload to the triage backend's
 * `POST /api/v1/ingest/crash` endpoint. These two hooks catch uncaught
 * exceptions and unhandled promise rejections anywhere in the app -- but
 * NOT errors thrown during React's own render cycle, which React
 * intercepts before they ever reach `window.onerror`. Pair this with
 * `CrashBoundary.tsx` (which calls `installCrashReporter` for you) to
 * cover that case too.
 *
 * This intentionally does the least amount of work possible to prove the
 * loop end-to-end: no batching, no retry, no offline queue, no sampling.
 * A production-grade version (Sentry, Bugsnag, ...) does all of that; this
 * is the learning-project equivalent of the same idea.
 */

export type CrashSource = "window.onerror" | "unhandledrejection" | "react-error-boundary";
export type AppEnvironment = "development" | "staging" | "production";

export interface CrashReporterOptions {
  /** Base URL of the running FDE triage backend, e.g. "http://127.0.0.1:8001". No trailing slash. */
  backendUrl: string;
  /** Forwarded as-is to the backend's TicketCreate.environment. Defaults to "production". */
  environment?: AppEnvironment;
  /** Called with the crash payload right before it's sent. Return `false` to suppress that one report. */
  onCapture?: (payload: CrashPayload) => boolean | void;
}

export interface CrashPayload {
  message: string;
  stack?: string;
  componentStack?: string;
  url?: string;
  userAgent?: string;
  source: CrashSource;
  environment?: AppEnvironment;
}

let installed = false;

/**
 * Registers the two window-level crash hooks. Safe to call multiple times
 * (only installs once) and safe to call during SSR (no-ops off the
 * browser). Call this once, as early as possible on the client -- e.g.
 * inside `CrashBoundary`'s `componentDidMount`, which is the recommended
 * integration point (see CrashBoundary.tsx / this package's README).
 */
export function installCrashReporter(options: CrashReporterOptions): void {
  if (installed || typeof window === "undefined") {
    return;
  }
  installed = true;

  window.addEventListener("error", (event: ErrorEvent) => {
    reportCrash(options, {
      message: event.message || "Unknown window error",
      stack: event.error instanceof Error ? event.error.stack : undefined,
      url: window.location.href,
      userAgent: navigator.userAgent,
      source: "window.onerror",
      environment: options.environment,
    });
  });

  window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
    const reason: unknown = event.reason;
    reportCrash(options, {
      message: reason instanceof Error ? reason.message : String(reason ?? "Unhandled promise rejection"),
      stack: reason instanceof Error ? reason.stack : undefined,
      url: window.location.href,
      userAgent: navigator.userAgent,
      source: "unhandledrejection",
      environment: options.environment,
    });
  });
}

/** Called by CrashBoundary.tsx's componentDidCatch -- the React-render-tree crash path. */
export function reportReactCrash(options: CrashReporterOptions, error: Error, componentStack: string | null): void {
  reportCrash(options, {
    message: error.message,
    stack: error.stack,
    componentStack: componentStack ?? undefined,
    url: typeof window !== "undefined" ? window.location.href : undefined,
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
    source: "react-error-boundary",
    environment: options.environment,
  });
}

function reportCrash(options: CrashReporterOptions, payload: CrashPayload): void {
  if (options.onCapture?.(payload) === false) {
    return;
  }

  // Fire-and-forget, deliberately: a crash reporter must never itself
  // throw or block the app it's instrumenting -- that would turn "report
  // the crash" into a second way to crash. `keepalive: true` lets the
  // request survive a page unload/navigation that follows the crash,
  // which a plain fetch would otherwise abort mid-flight.
  fetch(`${options.backendUrl}/api/v1/ingest/crash`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    keepalive: true,
  }).catch(() => {
    // Swallowed on purpose: if the triage backend itself is unreachable,
    // that must not become a second, user-visible failure in the app
    // being monitored.
  });
}
