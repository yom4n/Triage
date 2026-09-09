# Autonomous Tier-1 Technical Support & Triage Engine

This repository is a local-first incident triage system for software support. It accepts tickets from three sources: a human API call, a browser crash SDK, or a proactive Prometheus metric detector. All three paths normalize into the same ticket shape and run through the same LangGraph pipeline.

The core graph is:

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

The first half classifies and explains the incident. The second half is the remediation loop: generate a candidate file rewrite, apply it in an isolated Docker sandbox, run the monitored repo's own test command, and only open a PR when that path is allowed by the verification result.

## Stack

- **FastAPI**: async HTTP layer and lifespan wiring.
- **Pydantic v2**: request validation, normalized crash and metric payloads, structured LLM output contracts.
- **PostgreSQL 16 + pgvector**: transactional ticket store and RAG corpus with HNSW cosine search.
- **Redis**: LangGraph checkpoint store for resumable agent state.
- **SQLAlchemy 2.0 async**: database access.
- **LangGraph**: multi-node agent workflow with conditional routing and a bounded fix-verification cycle.
- **Ollama**: default local chat model and embedding model.
- **Anthropic SDK**: optional remote LLM backend.
- **Docker**: infrastructure plus sandboxed verification of generated fixes.
- **OpenTelemetry**: traces across HTTP, SQLAlchemy, graph nodes, LLM calls, and error responses.
- **Next.js frontend**: optional live ticket and SSE agent-run view.
- **Prometheus demo stack**: monitored demo app plus metric detector.

## Project Layout

```text
docker-compose.yml                 # local postgres (pgvector) + redis
docker-compose.prod.yml            # production backend stack for a VM
docker-compose.demo.yml            # demo app + prometheus-oriented stack
Dockerfile                         # backend container image
requirements.txt
.env.example                       # local defaults for DB, Redis, LLM, detector, sandbox, GitHub
DEPLOY.md                          # VM + Caddy + Vercel deployment notes
README.md                          # quickstart and command index
WRITEUP.md                         # reviewer narrative
ARCHITECTURE.md                    # this file

init-db/
  01-init-extensions.sql           # CREATE EXTENSION vector

scripts/
  seed_tickets.py                  # inserts verified historical RAG tickets
  vector_index_analysis.sql        # HNSW / IVFFlat / seq scan comparison
  failure_drills.py                # DB outage, LLM failure, malformed output drills
  detection_drill.py               # metric detector drill helper

prometheus/
  prometheus.yml                   # scrape config for the demo app

demo-app/
  Dockerfile
  README.md
  main.py                          # FastAPI demo app with /work, /metrics, /admin/fault
  loadgen.py                       # traffic generator for detector demos

client-sdks/
  react/
    README.md                      # monitored-app setup
    crashReporter.ts               # window.onerror / unhandledrejection reporter
    CrashBoundary.tsx              # React error boundary wrapper

evals/
  README.md                        # eval commands
  runner.py                        # dry-run, offline, and full eval runner
  REPORT.md                        # human-readable latest committed report
  scorecard.json                   # machine-readable latest committed scorecard
  cases/                           # 16 ground-truthed incident cases

frontend/
  package.json
  app/
    page.tsx                       # dashboard
    components/AgentRun.tsx        # live SSE run timeline
    api/
      triage/route.ts
      triage/stream/route.ts
      tickets/route.ts
      health/route.ts
      detector/status/route.ts
      evals/scorecard/route.ts

app/
  __init__.py
  config.py                        # typed settings
  database.py                      # async engine/session and schema bootstrap
  graph.py                         # LangGraph nodes and routing
  main.py                          # FastAPI routes, lifespan, RFC 7807 handlers
  models.py                        # Ticket ORM model
  schemas.py                       # API schemas and source normalizers
  services/
    detector.py                    # Prometheus polling detector
    embeddings.py                  # Ollama / deterministic embeddings
    github.py                      # GitHub file fetch, branch, commit, PR helpers
    llm.py                         # structured LLM and code-fix calls
    rag.py                         # verified-ticket vector retrieval
    sandbox.py                     # Docker verification of generated fixes
    telemetry.py                   # OpenTelemetry setup and traced_node()
```

## Runtime Sources

Tickets carry a `source` column with one of:

- `human`: created through `POST /api/v1/triage`.
- `crash`: created through `POST /api/v1/ingest/crash`.
- `metric`: created through `POST /api/v1/ingest/alert` or by `DetectorService`.

The source is stored with the ticket but not used to fork the graph. A human report, a browser crash, and a metric alert all receive the same extraction, RAG, triage, escalation, and fix-generation treatment after normalization.

## HTTP Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Basic health check. |
| `POST /api/v1/triage` | Submit a human ticket. |
| `GET /api/v1/triage/stream` | Run triage as an SSE stream for the frontend. |
| `POST /api/v1/ingest/crash` | Normalize a client/browser crash into a ticket. |
| `POST /api/v1/ingest/alert` | Normalize a metric alert into a ticket. |
| `GET /api/v1/tickets` | List stored tickets. |
| `GET /api/v1/detector/status` | Return detector enabled state, rule config, last values, firing states, and last ticket IDs. |
| `GET /api/v1/evals/scorecard` | Return `evals/scorecard.json` when present. |

## LangGraph Pipeline

### 1. `log_inspector`

`log_inspector_node` receives the normalized title, environment, stack trace, and optional description. It asks the configured LLM for the root exception or error signature, exception message, affected file and line if actually present, a clean embedding query, and confidence.

The node uses strict Pydantic validation for model output. If the model is unreachable or returns malformed content after retries, the node falls back to deterministic regex extraction. It also mechanically extracts first-party file and line frames from stack traces so later fix generation has a real target.

### 2. `rag_lookup`

`rag_lookup_node` embeds the clean query and asks `app/services/rag.py` for nearby historical tickets. Retrieval is intentionally gated:

```text
is_verified = true
```

Only tickets whose resolution was verified by a human or seed data can ground future recommendations. Live tickets are persisted with embeddings, but default to `is_verified=false`, so the system cannot feed its own unreviewed guesses back into the corpus.

### 3. `triage_router`

`triage_router_node` receives the root error, original trace, environment, and verified RAG context. It returns severity, a short diagnostic summary, ordered remediation steps, and confidence.

The LLM path is schema-constrained. The fallback path is deterministic severity and remediation logic retained from the earliest implementation so the endpoint degrades instead of failing when the model backend is unavailable.

### 4. `fallback_human_escalation`

The graph routes here when the triage LLM did not succeed or its confidence is below `MIN_CONFIDENCE` (`0.70` by default). The node preserves the partial diagnosis, sets `status="ESCALATED_TO_HUMAN"`, writes an explicit `escalation_reason`, and returns a hand-off step instead of overstating certainty.

### 5. `generate_fix`

Confident, non-escalated tickets enter the fix sub-graph. `generate_fix_node` resolves the affected file in the configured GitHub repo, fetches the current content, and asks the LLM for a complete corrected file. The diff is computed locally with `difflib`, not trusted from the model.

If the system cannot identify a file, cannot reach GitHub, receives an identical rewrite, or hits an LLM failure, it records `fix_skipped_reason` and routes to `fix_escalation`.

### 6. `verify_fix`

`verify_fix_node` is the Phase 1 centerpiece. It applies the candidate file content to a fresh clone of the monitored repo inside Docker, detects the repo's test command, and runs the repo's own tests against the exact proposed change.

| Status | Route | Meaning |
| --- | --- | --- |
| `PASSED` | `open_pr` | Tests ran and passed in the sandbox. |
| `FAILED_RETRY` | `generate_fix` | Tests ran and failed, but retry budget remains. |
| `FAILED_MAX_ATTEMPTS` | `fix_escalation` | Tests failed through the bounded retry budget. |
| `SKIPPED_NO_SANDBOX` | `open_pr` | Docker/sandbox unavailable; PR may open with an unverified warning. |
| `SKIPPED_UNTESTABLE` | `fix_escalation` | The repo/file/toolchain could not be tested. |
| `NOT_ATTEMPTED` | terminal/default | No fix path was attempted. |

The `FAILED_RETRY -> generate_fix` edge is the graph's only cycle. It is bounded by `SANDBOX_MAX_ATTEMPTS`, and the prior test output is passed back into the next generation prompt. A bad fix that never goes green is escalated with its diff and failing output attached; it is not opened as a verified PR.

### 7. `open_pr`

`open_pr_node` opens a GitHub branch, commits the full-file rewrite, and creates a PR. The PR body says whether the fix was verified in the Docker sandbox or was opened unverified because the sandbox was unavailable. Nothing is merged automatically.

### 8. `fix_escalation`

`fix_escalation_node` terminates attempted fixes that cannot safely become PRs. It leaves the diagnostic result intact and records why automated remediation stopped.

## Storage Model

`Ticket` is both the system of record and the vector corpus. Important columns:

- `source`: `human`, `crash`, or `metric`.
- extraction fields: `extracted_error`, `exception_message`, `affected_file`, `affected_line`.
- triage fields: `severity`, `summary`, `resolution_steps`, `confidence`, `status`, `escalation_reason`.
- fix fields: `fix_attempted`, `fix_skipped_reason`, `fix_diff`, `fix_pr_url`, `fix_branch_name`.
- verification fields: `fix_verified`, `fix_verification_status`, `fix_verification_attempts`, `fix_test_command`, `fix_test_output_tail`.
- RAG fields: `resolution`, `is_verified`, `embedding_text`, `embedding`.

The pgvector column width is fixed by `EMBEDDING_DIM`. Changing embedding models across dimensions requires a database migration, not only an environment variable change.

## Proactive Detector

`app/services/detector.py` owns an async background polling task. When `DETECTOR_ENABLED=true`, FastAPI lifespan creates `DetectorService`, starts it, and stops it on shutdown.

The detector queries Prometheus and watches three rules:

| Rule | PromQL | Threshold setting | Comparison | Severity hint |
| --- | --- | --- | --- | --- |
| `high_error_rate` | `sum(rate(demo_app_errors_total[1m]))` | `DETECTOR_ERROR_RATE_THRESHOLD` | `>` | `HIGH` |
| `high_p95_latency` | `histogram_quantile(0.95, sum(rate(demo_app_request_latency_seconds_bucket[5m])) by (le))` | `DETECTOR_P95_LATENCY_THRESHOLD_SECONDS` | `>` | `MEDIUM` |
| `dependency_down` | `demo_app_dependency_up` | `DETECTOR_DEPENDENCY_DOWN_THRESHOLD` | `<=` | `CRITICAL` |

On a fresh breach, the detector synthesizes a `MetricAlert`, converts it with `metric_alert_to_ticket_create()`, and calls the same `run_triage()` path with `source="metric"`. Per-rule in-memory state prevents repeated ticket creation while a rule remains firing, and `DETECTOR_RULE_COOLDOWN_SECONDS` limits refiring after transitions.

`GET /api/v1/detector/status` exposes the detector state: enabled flag, PromQL expressions, thresholds, comparisons, severity hints, last values, firing flags, last ticket IDs, and last query errors.

For manual alert injection, `POST /api/v1/ingest/alert` accepts the same loose `MetricAlert` schema and files a metric-sourced ticket without waiting for a detector poll.

## Demo App And Prometheus

`demo-app/` is a small FastAPI service used to demonstrate metric detection. It exposes:

- `GET /work`: a workload endpoint.
- `GET /metrics`: Prometheus metrics.
- `POST /admin/fault`: toggles `latency`, `errors`, and `dependency_down`.

`docker-compose.demo.yml` and `prometheus/prometheus.yml` stand up the detector demo stack. `demo-app/loadgen.py` can create traffic so Prometheus has data to evaluate.

## Evals

`evals/` contains 16 ground-truthed synthetic incidents. `evals/runner.py` drives the real FastAPI app through `TestClient(app)` and compares responses with expected root-cause keywords, severity, escalation behavior, and fix-verification status.

Commands:

```bash
python -m evals.runner --dry-run
python -m evals.runner --offline
python -m evals.runner
python -m evals.runner --limit 4 --offline
```

The committed `evals/scorecard.json` is an offline fallback-path floor, not the full live-system score. It was generated in `mode="offline"` and records:

- 16 cases.
- root-cause hit rate: `0.875`.
- severity accuracy: `0.5`.
- escalation precision: `0.25`.
- escalation recall: `1.0`.
- escalation F1: `0.4`.
- fix attempted count: `0`.
- fix verification pass rate: `0.0`.

A full `python -m evals.runner` run against live infrastructure and model backends is the real score for the current machine and configuration.

`GET /api/v1/evals/scorecard` returns the committed scorecard JSON so the frontend and reviewers can inspect the baseline without running the harness.

## Observability

`app/services/telemetry.py` sets up OpenTelemetry and `traced_node()`.

Traced surfaces:

- FastAPI requests.
- SQLAlchemy queries.
- LangGraph node executions.
- LLM calls.
- domain attributes such as `llm.used`, `rag.hits`, `triage.confidence`, and `triage.status`.

By default, spans go to stdout through `ConsoleSpanExporter`. A deployment can set a standard OTLP endpoint:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Error responses use RFC 7807 `application/problem+json` bodies and include the current `trace_id`. That means an on-call engineer can take the ID from an API failure and find the exact trace in the collector instead of relying on timestamp search.

Instrumentation is installed immediately after `FastAPI(...)` is created. This matters because Starlette builds and caches middleware on the first ASGI call, which is often lifespan startup; instrumenting inside lifespan is too late for request tracing.

## Resilience

The system distinguishes normal degraded outcomes from process errors:

- LLM extraction failure: regex fallback.
- Embedding failure: continue without RAG context.
- Triage LLM failure or low confidence: human escalation.
- Fix generation failure: fix escalation.
- Sandbox unavailable: optionally open an unverified PR with an explicit warning.
- Sandbox test failure: retry with real test output, then escalate without opening a verified PR.
- Database or unexpected process error: RFC 7807 response with `trace_id`.

`scripts/failure_drills.py` exercises database outage, LLM rate limit, and corrupted LLM output paths against the API.

## Deployment Shape

The public demo is intended to run as:

- backend, Postgres, Redis, and Docker sandbox on a VM with Docker Compose,
- frontend on Vercel,
- Caddy in front of the backend for HTTPS and SSE-friendly reverse proxying.

The backend needs access to a real Docker daemon because `verify_fix_node` runs monitored-repo tests in throwaway containers. That requirement rules out many simple PaaS deployments for the backend. See [DEPLOY.md](DEPLOY.md) for the full path.

## What's Intentionally Out Of Scope

- **Automatic merge**: PRs are opened for review only.
- **Authentication and authorization**: the demo API is open unless a deployer puts a proxy or network control in front of it.
- **Rate limiting and abuse protection**: important before exposing a paid LLM-backed endpoint publicly.
- **Alembic migrations**: local dev uses `create_all` plus additive DDL bootstrap.
- **Admin verification workflow**: `is_verified` is enforced by RAG, but there is no shipped UI/API to promote live tickets into the verified corpus.
- **Source-map de-minification**: browser crash ingestion expects useful dev-mode or already-symbolicated stacks.
- **Crash deduplication/fingerprinting**: repeated crashes can create repeated tickets.
- **Persistent detector state**: detector firing state is in memory; a process restart can re-file an alert after cooldown behavior resets.
- **Production tracing backend**: stdout spans are local-friendly; Jaeger/Tempo/APM setup is deployer-owned.
- **Comprehensive metrics dashboards**: Prometheus is used for the demo detector, not a full RED/USE operations package for this service.
- **Arbitrary repo support guarantees**: the sandbox detects common test commands, but unknown toolchains can route to `SKIPPED_UNTESTABLE`.
