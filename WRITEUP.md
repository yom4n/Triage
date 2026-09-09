# Writeup: Autonomous Tier-1 Technical Support & Triage Engine

## The Problem

Tier-1 incident triage is repetitive but still risky. A real support queue fills with variations of the same work:

- read a stack trace,
- identify the likely root cause,
- decide whether it is urgent,
- search past incidents,
- write a short remediation note,
- decide whether a human needs to take over,
- sometimes make the obvious code/config fix,
- prove the fix did not break the monitored app.

Most automation demos stop at the middle of that list. They summarize an error and maybe draft a patch, but they do not prove the patch works. That is the dangerous part. A confident-looking bad fix is worse than no automation because it consumes reviewer attention and can create production risk.

This project treats verification as the center of the system. The agent can draft a fix, but a verified PR is only opened after the candidate rewrite is applied to the monitored repository inside Docker and the monitored repository's own tests pass. If the fix fails, the graph loops back with the real test output. If it still does not pass within the bounded retry budget, the ticket is escalated and no verified PR is opened.

The result is a local-first triage engine that can be demonstrated end to end without a paid model API, but is structured like a production incident workflow: explicit routing, durable tickets, trace IDs, structured errors, verified retrieval, proactive detection, and evals.

## System At A Glance

There are three ways an incident enters the system:

- `POST /api/v1/triage`: a human files a ticket.
- `POST /api/v1/ingest/crash`: a browser or React app reports a crash through `client-sdks/react`.
- `POST /api/v1/ingest/alert`: a metric alert is posted manually, or synthesized by the background detector.

All three paths normalize into `TicketCreate` and call the same internal triage runner. The database records the origin in `tickets.source` as `human`, `crash`, or `metric`, but source does not create separate business logic. That is intentional: the agent should reason about the failure, not about whether a person or Prometheus noticed it first.

The core stack is FastAPI, LangGraph, PostgreSQL with pgvector, Redis checkpointing, Ollama by default, Docker for sandbox verification, and OpenTelemetry for traces. The optional frontend shows tickets and live agent-run events over SSE. The demo app and Prometheus stack exist to show proactive detection without needing a real production service.

## The LangGraph Pipeline

The graph is:

```text
START
  -> log_inspector
  -> rag_lookup
  -> triage_router
       -> fallback_human_escalation -> END
       -> generate_fix
            -> verify_fix
                 -> open_pr -> END
                 -> generate_fix
                 -> fix_escalation -> END
```

The first three nodes are the triage path. They always run in order.

`log_inspector` extracts the root exception, message, affected file, affected line, and a clean semantic-search query. It uses a structured LLM call when available and deterministic regex fallback when not.

`rag_lookup` embeds the clean failure query and retrieves similar verified incidents from Postgres/pgvector.

`triage_router` classifies severity, writes the diagnostic summary, proposes remediation steps, and reports confidence.

After `triage_router`, the graph branches. If the triage LLM failed or confidence is below the configured threshold, the graph routes to `fallback_human_escalation`. If the diagnosis is confident, the graph enters the fix sub-graph.

The fix sub-graph is:

```text
generate_fix -> verify_fix -> open_pr
                         |
                         -> generate_fix
                         -> fix_escalation
```

That cycle is bounded. `verify_fix` can route back to `generate_fix` only while `SANDBOX_MAX_ATTEMPTS` has not been reached.

## RAG With A Verified Corpus Gate

The RAG corpus lives in the same `tickets` table as the operational ticket record. That keeps storage simple and avoids vector-store synchronization problems. A ticket can be embedded and persisted in the same transaction that creates the ticket.

The important guardrail is `is_verified`.

`app/services/rag.py` only retrieves rows where `is_verified=true`. Seeded historical incidents are inserted as verified. Live tickets are stored as unverified by default, even though they may have embeddings.

That prevents a self-reinforcing loop. Without the gate, the model could produce a speculative remediation, store it, retrieve it later as "history," and gradually amplify its own unreviewed guesses. With the gate, retrieval means "this resolution was allowed into the corpus," not merely "the agent said this before."

## Phase 1: Sandboxed Fix Verification

Phase 1 is the centerpiece because it draws the line between a helpful agent and an unsafe patch generator.

The fix path starts only after a confident, non-escalated triage. `generate_fix_node` resolves the affected file in the configured GitHub repository, fetches the current file content, and asks the model for a complete corrected version of the file. The model does not author the diff. The system computes the unified diff locally from original content and proposed content.

Then `verify_fix_node` applies the proposed file content in a throwaway Docker sandbox and runs the monitored repository's own tests. The test command is the verdict. The possible outcomes are deliberately explicit:

- `PASSED`: tests passed; `open_pr_node` may open a verified PR.
- `FAILED_RETRY`: tests failed; route back to `generate_fix_node` with the test output.
- `FAILED_MAX_ATTEMPTS`: tests failed through the retry budget; escalate with the diff and latest failure output.
- `SKIPPED_NO_SANDBOX`: Docker or sandbox execution was unavailable; the system may open an unverified PR with a warning.
- `SKIPPED_UNTESTABLE`: the repo or toolchain could not be tested; escalate without opening a verified PR.
- `NOT_ATTEMPTED`: no fix path ran.

The key behavior is that a bad fix never becomes a verified PR. If tests fail, the model gets the actual failure output and tries again. If it cannot make the tests pass, the ticket keeps the diff and output so a human can continue from a useful starting point, but the system does not pretend the fix is safe.

## Phase 2: Live Run View And Deployment Scaffolding

Phase 2 added the live agent-run surface. The backend exposes an SSE stream that reports graph progress as the ticket moves through extraction, retrieval, triage, fix generation, verification, PR creation, or escalation. The frontend consumes that stream and shows the run as an operational timeline rather than a black-box request.

The deployment scaffold reflects a real constraint: the backend needs Docker access for verification. `DEPLOY.md` therefore uses a VM for the FastAPI backend, Postgres, Redis, and Docker sandbox, while the Next.js frontend can live on Vercel. Caddy terminates HTTPS and is configured for unbuffered SSE proxying.

## Phase 3: Proactive Detection

Phase 3 added `app/services/detector.py`, a background Prometheus polling service. When enabled, it watches three rules:

- `high_error_rate`: `sum(rate(demo_app_errors_total[1m]))`
- `high_p95_latency`: `histogram_quantile(0.95, sum(rate(demo_app_request_latency_seconds_bucket[5m])) by (le))`
- `dependency_down`: `demo_app_dependency_up`

Each rule has a threshold setting, comparison, severity hint, last observed value, firing state, and cooldown. On a fresh breach, the detector synthesizes a `MetricAlert`, converts it to a ticket, and calls the same triage path with `source="metric"`.

This matters because proactive detection is not a side channel. A metric-triggered incident receives the same RAG grounding, confidence gate, fix-verification loop, persistence, and observability as a human-filed ticket.

The detector also exposes `GET /api/v1/detector/status`, which returns the enabled state, configured rules, thresholds, last values, firing states, last ticket IDs, and last query errors. `POST /api/v1/ingest/alert` provides a direct alert-ingestion endpoint for manual or external alertmanager-style integrations.

## Observability

The project uses OpenTelemetry across the request lifecycle:

- FastAPI route spans,
- SQLAlchemy query spans,
- LangGraph node spans through `traced_node()`,
- LLM call spans,
- domain attributes such as `rag.hits`, `llm.used`, `triage.confidence`, and `triage.status`.

By default, spans print to stdout as JSON. In deployment, a standard OTLP endpoint can send them to Jaeger, Tempo, or an APM vendor.

Errors are returned as RFC 7807 `application/problem+json` responses. Those bodies include a `trace_id`, so the operator can take the response from a failing request and jump straight to the trace. That is more useful than a generic 500 and much less fragile than timestamp-based log search.

## Phase 4: Evaluation

The eval harness lives in `evals/`. It contains 16 ground-truthed synthetic incidents and drives the real FastAPI app through `TestClient(app)`.

The committed `evals/REPORT.md` was generated on 2026-09-09 and reports 16 cases. The committed `evals/scorecard.json` is explicitly an offline fallback-path floor, generated with `mode="offline"`. It is useful because it proves the deterministic degraded path is measurable, but it is not the final live-model score.

The committed offline floor is:

- case count: `16`,
- root-cause hit rate: `0.875`,
- severity accuracy: `0.5`,
- escalation precision: `0.25`,
- escalation recall: `1.0`,
- escalation F1: `0.4`,
- fix attempted count: `0`,
- fix verification pass rate: `0.0`.

That scorecard shows the fallback path is conservative: it escalated all 16 cases, which gives perfect recall for the four cases expected to escalate, but poor precision. A full run with:

```bash
python -m evals.runner
```

is the real score for the complete system, because it exercises live infrastructure and the configured LLM/embedding backends. The committed scorecard should be read as the floor when the live path is unavailable.

`GET /api/v1/evals/scorecard` exposes the committed scorecard to the frontend and reviewers.

## What Is Deliberately Out Of Scope

This is not presented as a production support platform. The omissions are intentional and visible:

- no automatic merge,
- no built-in auth or tenant model,
- no rate limiting,
- no production Alembic migration chain,
- no shipped tracing backend,
- no admin UI to mark new resolutions verified,
- no crash fingerprinting or deduplication,
- no source-map de-minification,
- no persistent detector firing state,
- no guarantee that every arbitrary repository's test command can be inferred.

The next useful work would be an authenticated reviewer UI, an admin flow for promoting resolved tickets into the verified corpus, persistent detector state, alertmanager integration, crash deduplication, source-map support, and broader evals that include successful live fix verification against controlled repos.
