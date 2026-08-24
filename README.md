# Autonomous Tier-1 Technical Support & Triage Engine

**Phase 1 (Days 1–2):** infrastructure, database, input guardrails, and the
LangGraph triage pipeline (deterministic rules).

**Phase 2 (Day 3):** live local LLM reasoning (Ollama) and pgvector
semantic search (RAG), with automatic fallback to the Phase 1 deterministic
rules if the LLM/embedding backend is unavailable.

**Phase 3 (Day 4):** production observability, resilience, and chaos
testing — OpenTelemetry tracing across FastAPI/SQLAlchemy/LangGraph, a
confidence-gated human-escalation node, RFC 7807 structured error
responses, and a chaos-drill script that proves all of it under simulated
outages.

## Stack

- **FastAPI** (async) — HTTP layer
- **Pydantic v2** — strict input validation / guardrails
- **PostgreSQL 16 + pgvector** — transactional store _and_ RAG corpus (HNSW cosine index)
- **Redis** — LangGraph checkpoint store (resumable agent state)
- **SQLAlchemy 2.0 (async)** — ORM / data access
- **LangGraph** — multi-agent StateGraph pipeline: `log_inspector → rag_lookup → triage_router → [fallback_human_escalation]`
- **Ollama** (default) — fully local LLM reasoning + embeddings, no API key, no per-token cost.
  Swap to Claude (`anthropic` SDK) via one config value — see [LLM backend](#llm-backend--embeddings).
- **OpenTelemetry** — distributed tracing across every layer (HTTP routes, DB
  queries, graph node executions, LLM calls) — see [Observability](#observability-opentelemetry).

## Project layout

```
docker-compose.yml            # postgres (pgvector) + redis, with healthchecks
init-db/                      # runs once on first postgres boot: CREATE EXTENSION vector
requirements.txt
.env.example                  # copy to .env and adjust if needed
scripts/
  seed_tickets.py              # inserts 8 verified historical tickets with real embeddings
  vector_index_analysis.sql    # EXPLAIN ANALYZE walkthrough: HNSW vs IVFFlat vs seq scan
  failure_drills.py            # chaos drills: DB outage, LLM rate limit, corrupted LLM output
app/
  config.py                    # Settings (pydantic-settings): DB/Redis/LLM/embedding/RAG config
  models.py                    # Ticket ORM model (Vector(EMBEDDING_DIM) column, RAG + status fields)
  database.py                  # async engine/session (lazy init), HNSW index bootstrap
  schemas.py                   # TicketCreate / TicketResponse + guardrails
  graph.py                     # TicketState + log_inspector/rag_lookup/triage_router/fallback_human_escalation nodes
  main.py                      # FastAPI app, lifespan wiring, POST /api/v1/triage, RFC 7807 error handlers
  services/
    embeddings.py               # Ollama / deterministic embedding backends
    llm.py                      # Ollama / Anthropic structured-output LLM backends
    rag.py                      # pgvector cosine similarity retrieval
    telemetry.py                 # OpenTelemetry setup: tracer provider, FastAPI/SQLAlchemy instrumentation, traced_node()
```

## Running it

1. Start infrastructure:

   ```bash
   docker compose up -d
   ```

   Wait for both services to report healthy: `docker compose ps`.

   > **Note on ports:** Postgres is mapped to host port `55432` (not the
   > default `5432`) because Windows machines commonly have a native
   > Postgres service already bound to `5432`/`5433`, which silently
   > intercepts connections meant for the container instead of erroring —
   > the app connects with the wrong credentials and it looks like a
   > password problem. If `55432` is free on your machine you don't need
   > to change anything; `.env.example` already matches this mapping. If
   > it collides too, pick another free port and update both
   > `docker-compose.yml`'s `postgres.ports` entry and `POSTGRES_PORT` in
   > your `.env` together.

2. Install and start [Ollama](https://ollama.com) on the host (not in
   Docker — the app talks to `http://localhost:11434` by default), then
   pull the two local models this project uses:

   ```bash
   ollama pull qwen2.5:7b-instruct   # chat model: extraction + triage reasoning
   ollama pull nomic-embed-text      # embedding model: 768-dim, for RAG
   ```

   `ollama serve` typically starts automatically after install; if not,
   run it in its own terminal.

3. Create a virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

4. (Optional) copy `.env.example` to `.env` — the defaults already match
   `docker-compose.yml` and a stock local Ollama install, so this is only
   needed if you change ports/creds/models.
5. Seed the RAG corpus with historical tickets (idempotent — safe to re-run):

   ```bash
   python scripts/seed_tickets.py
   ```

6. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

   On startup the app creates the `tickets` table + HNSW vector index (dev
   convenience — swap for Alembic migrations in production), installs
   OpenTelemetry instrumentation, and opens the Redis-backed LangGraph
   checkpointer. Every span prints to stdout as JSON by default — see
   [Observability](#observability-opentelemetry) to point it at a real
   collector instead.

7. Try it — this stack trace closely matches a seeded ticket, so watch
   `resolution_steps` come back grounded in that historical resolution:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/triage \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Checkout fails intermittently under load",
       "environment": "production",
       "stack_trace": "psycopg2.OperationalError: connection to server at \"db-primary\" port 5432 failed: timeout expired\n  File \"checkout.py\", line 91, in charge_card\n    cursor.execute(query, params)",
       "description": "Started after last night'\''s deploy."
     }'
   ```

   Expect a `201` with a body like:

   ```json
   {
     "ticket_id": "…",
     "title": "Checkout fails intermittently under load",
     "environment": "production",
     "extracted_error": "OperationalError",
     "affected_file": "checkout.py",
     "affected_line": 91,
     "severity": "CRITICAL",
     "summary": "[CRITICAL] Database connection pool exhaustion during high load caused checkout requests to time out.",
     "resolution_steps": [
       "Confirm connection pool exhaustion via SELECT count(*) FROM pg_stat_activity.",
       "Increase PgBouncer pool_size and set a hard statement_timeout on the checkout role.",
       "Add a circuit breaker around the payment DB call so it fails fast under saturation."
     ],
     "confidence": 0.9,
     "similar_tickets_considered": 1,
     "status": "COMPLETED",
     "escalation_reason": null,
     "created_at": "…"
   }
   ```

   A low-confidence or LLM-outage run instead comes back `201` with
   `"status": "ESCALATED_TO_HUMAN"`, a populated `escalation_reason`, and a
   single hand-off instruction in `resolution_steps` — see
   [Resilience](#resilience-human-escalation--structured-errors).

   Interactive docs: http://127.0.0.1:8000/docs

8. Inspect vector index performance:

   ```bash
   docker exec -i triage_postgres psql -U triage_user -d triage_engine < scripts/vector_index_analysis.sql
   ```

9. Run the chaos drills (optional, proves the resilience story end to end):

   ```bash
   python scripts/failure_drills.py
   ```

   See [Chaos engineering drills](#chaos-engineering-drills).

## LLM backend & embeddings

Selected entirely via `.env` — no code changes:

| Setting              | Default                                       | Alternative                                                                                     |
| -------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `LLM_PROVIDER`       | `ollama` (`qwen2.5:7b-instruct`, local)       | `anthropic` (`claude-opus-5`, needs `ANTHROPIC_API_KEY`)                                        |
| `EMBEDDING_PROVIDER` | `ollama` (`nomic-embed-text`, 768-dim, local) | `deterministic` (offline hashing vectorizer — CI/dry-run only, **not** semantically meaningful) |

Both LLM-backed graph nodes (`log_inspector_node`, `triage_router_node` in
[app/graph.py](app/graph.py)) enforce a strict JSON-schema contract on the
model's output (Ollama via grammar-constrained decoding, Anthropic via
`messages.parse`) and **fall back to the original Phase 1 deterministic
regex/keyword logic** if the LLM backend is unreachable or its output fails
validation after retry. As of Phase 3, that fallback path is also what
feeds the confidence-gated escalation node described below — a degraded
answer is never returned dressed up as a fully-reasoned one.

`EMBEDDING_DIM` (768 for `nomic-embed-text`, 1536 for OpenAI-compatible
models) is a **schema constraint**, not just config — pgvector fixes a
`Vector(N)` column's width at table-creation time. Changing embedding
models to a different dimension requires dropping/recreating the
`tickets.embedding` column, not just an env var edit.

## RAG corpus & verification

`rag_lookup_node` only retrieves tickets where `is_verified = true` (see
`app/services/rag.py`). Seed data is inserted pre-verified; tickets the
live API triages are inserted as `is_verified = false` — they're embedded
and stored so they _can_ ground future retrievals, but only once a human
confirms the recorded resolution actually worked. This prevents the corpus
from being able to amplify its own unverified guesses across tickets.

## Observability (OpenTelemetry)

Every request is traced end to end through `app/services/telemetry.py`:

- **FastAPI routes** — auto-instrumented; the root span for
  `POST /api/v1/triage` is what every span below nests under.
- **SQLAlchemy** — every SQL statement (the pgvector cosine search, the
  ticket insert, `init_models()`'s DDL) shows up as its own child span
  with its own duration.
- **LangGraph nodes** — `traced_node()` wraps `log_inspector_node`,
  `rag_lookup_node`, `triage_router_node`, and
  `fallback_human_escalation_node`, each recording wall-clock duration
  plus domain attributes (`llm.used`, `triage.confidence`, `rag.hits`,
  `triage.status`).
- **LLM calls** — `app/services/llm.py` opens a nested `llm.call` span
  around each Ollama/Anthropic request and records the provider's own
  reported token counts (`llm.tokens.prompt` / `.completion` / `.total`),
  so a slow or expensive ticket can be diagnosed by opening its trace and
  reading off exactly which span was the bottleneck — a slow pgvector
  query and a slow/expensive LLM call are siblings in the same trace, not
  one lumped "triage took 8s" number.

By default, spans print to stdout as JSON (`ConsoleSpanExporter`) — zero
extra infrastructure needed for local dev. Point it at a real collector
(Jaeger, Tempo, an APM vendor, …) with the standard OTel env var:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

**Instrumentation-timing gotcha:** `instrument_fastapi_app(app)` is called
immediately after `FastAPI(...)` is constructed in `app/main.py`, not from
inside `lifespan()`. Starlette builds and caches its middleware stack on
the very first ASGI call it receives — which is the `lifespan` call itself
— so instrumenting from inside the startup hook patches a method that has
already been read, and every request silently gets zero tracing with no
error anywhere. `setup_telemetry()` (the TracerProvider + SQLAlchemy
instrumentation) can safely stay in `lifespan()` since it has no such
caching trap.

## Resilience: human escalation & structured errors

**Confidence-gated escalation.** `triage_router_node` now routes through a
conditional edge (`_route_after_triage` in [app/graph.py](app/graph.py))
instead of going straight to `END`:

```
... → triage_router_node ─┬─ (confident, LLM-grounded) ──────────→ END
                           └─ (LLM failed, or confidence < MIN_CONFIDENCE) → fallback_human_escalation_node → END
```

`fallback_human_escalation_node` fires when either the LLM call never
succeeded (timeout, malformed output after retries, backend unreachable —
`used_llm_triage_router=False`) or it succeeded but self-reported
confidence below `MIN_CONFIDENCE` (`0.70` by default). It preserves
whatever partial diagnosis exists, sets `status="ESCALATED_TO_HUMAN"` with
a human-readable `escalation_reason`, and replaces `resolution_steps` with
a single honest hand-off note — the request still returns `201`, the
ticket is still persisted, and no data is dropped; it's just flagged for a
human instead of auto-resolved.

**Structured errors.** Global exception handlers in `app/main.py` turn any
`SQLAlchemyError`, `LLMTimeoutError`, `LLMMalformedOutputError`, or
unhandled exception into an RFC 7807 (`application/problem+json`) body
carrying the live OpenTelemetry trace ID, instead of a bare 500:

```json
{
  "type": "https://triage-engine.internal/errors/database-error",
  "title": "Database Unavailable",
  "status": 503,
  "detail": "The triage engine could not complete a database operation. This is very likely transient (a dropped connection or an exhausted pool) -- retry with exponential backoff.",
  "instance": "http://127.0.0.1:8000/api/v1/triage",
  "trace_id": "6f248c031044c4e4f3a84c5afcd01417"
}
```

An on-call engineer can paste `trace_id` straight into the tracing backend
and land on the exact failing request — no timestamp-based log grepping.

## Chaos engineering drills

`scripts/failure_drills.py` simulates three production failure modes
against `POST /api/v1/triage`, in-process via `TestClient` (real Postgres +
Redis, no Ollama required — each drill patches the DB or LLM call site
directly with `unittest.mock.patch`):

| Drill | Simulates | Proves |
| ----- | --------- | ------ |
| **A — Database Outage** | `AsyncSession.commit` raises `OperationalError` mid-request | Global handler returns a structured `503` RFC 7807 body with a real `trace_id`, not a bare 500 |
| **B — API Rate Limit** | `call_structured` raises an `LLMError` mimicking a `429` | Request still returns `201` with `status=ESCALATED_TO_HUMAN`, confidence `0.0` — no crash, no hang, no dropped ticket |
| **C — Corrupted Response** | `call_structured` raises `LLMMalformedOutputError` with unparseable JSON | Same graceful degradation as Drill B, proving the fallback is keyed on "the LLM call didn't succeed," not one specific exception type |

```bash
python scripts/failure_drills.py
```

Requires `docker compose up -d` (Postgres + Redis) already running. Exits
non-zero if any drill's assertions fail.

## Notes / what's intentionally out of scope

- **Alembic migrations** — table/index creation still uses
  `Base.metadata.create_all` + raw DDL for zero-setup local dev. A real
  deployment should switch to versioned migrations (this is also why an
  existing dev database needs a manual `ALTER TABLE` if you pull schema
  changes onto a volume created before this phase — the new `status` /
  `escalation_reason` columns won't appear via `create_all` alone on a
  table that already exists).
- **No shipped tracing backend** — spans export to stdout by default;
  wiring up an actual Jaeger/Tempo/APM collector is a deploy-time choice,
  not something this repo stands up for you.
- **No metrics or log correlation beyond trace_id** — this phase covers
  traces only, not a metrics pipeline (RED/USE dashboards) or structured
  log-to-trace correlation beyond stamping `trace_id` on error responses.
- **No auth, rate limiting, or admin "verify ticket" endpoint** —
  `is_verified` is a real DB column with real RAG-gating behavior, but
  nothing yet flips it from `false` to `true`; that's a human-in-the-loop
  admin action still to be built.
