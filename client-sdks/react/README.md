# FDE crash reporter -- React / Next.js

Drop-in client-side crash capture for any React or Next.js app you want the
FDE triage engine to monitor. This is Phase 1 of the auto-ticketing loop:
**app crashes → ticket appears in the triage backend, automatically, with
no manual `curl`.**

## What it catches

| Source | Hook | Catches |
| --- | --- | --- |
| `window.onerror` | global listener | Uncaught exceptions anywhere in the page |
| `unhandledrejection` | global listener | Rejected promises nobody `.catch()`ed |
| React render errors | `CrashBoundary` (error boundary) | Errors thrown while React renders a component -- `window.onerror` never sees these |

## Setup (2 steps)

1. Copy `crashReporter.ts` and `CrashBoundary.tsx` into your app (e.g. `lib/fde/`).
2. Wrap your app with `CrashBoundary`, pointed at your running triage backend:

```tsx
// app/layout.tsx (Next.js App Router) or a similar top-level wrapper
import { CrashBoundary } from "@/lib/fde/CrashBoundary";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <CrashBoundary
          options={{
            backendUrl: process.env.NEXT_PUBLIC_TRIAGE_BACKEND_URL ?? "http://127.0.0.1:8001",
            environment: (process.env.NODE_ENV as "development" | "production") ?? "development",
          }}
        >
          {children}
        </CrashBoundary>
      </body>
    </html>
  );
}
```

That's the whole integration. `CrashBoundary` installs the window-level
hooks on mount *and* catches React render errors -- one component, one prop.

3. (Optional) set `NEXT_PUBLIC_TRIAGE_BACKEND_URL` in your app's `.env.local`
   if the triage backend isn't at `http://127.0.0.1:8001`.

## What gets sent

On any capture, a `CrashPayload` is POSTed to
`{backendUrl}/api/v1/ingest/crash`:

```json
{
  "message": "Cannot read properties of undefined (reading 'map')",
  "stack": "TypeError: ...\n    at TodoList (TodoList.tsx:42:19)",
  "componentStack": "\n    at TodoList\n    at App",
  "url": "http://localhost:3000/todos",
  "userAgent": "Mozilla/5.0 ...",
  "source": "react-error-boundary",
  "environment": "development"
}
```

The backend normalizes this into a real ticket (`app/schemas.py`'s
`crash_report_to_ticket_create`) and runs it through the exact same
LangGraph triage pipeline a manually-filed bug report goes through --
severity, summary, resolution steps, RAG-grounded context, and the
confidence-gated human-escalation fallback, all identical either way.

## Deliberately out of scope for this minimal SDK

This is a learning-project-scoped SDK, not a Sentry replacement. It
intentionally skips: retry/offline queueing (a failed POST is just
dropped), batching/rate-limiting (every crash sends immediately), dedup
(the same recurring crash makes a new ticket each time), and source-map
de-minification (run your app in dev mode, or with source maps available,
so stack traces already point at real files). See the FDE-Project README's
"Notes / what's intentionally out of scope" for the parallel list on the
backend side.
