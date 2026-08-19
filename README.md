# Autonomous Tier-1 Technical Support & Triage Engine

**Phase 1 (Days 1–2):** infrastructure, database, input guardrails, and the
LangGraph triage pipeline (deterministic rules).

**Phase 2 (Day 3):** live local LLM reasoning (Ollama) and pgvector
semantic search (RAG), with automatic fallback to the Phase 1 deterministic
rules if the LLM/embedding backend is unavailable.

## Stack

- **FastAPI** (async) — HTTP layer
- **Pydantic v2** — strict input validation / guardrails
- **PostgreSQL 16 + pgvector** — transactional store _and_ RAG corpus (HNSW cosine index)
- **Redis** — LangGraph checkpoint store (resumable agent state)
- **SQLAlchemy 2.0 (async)** — ORM / data access
- **LangGraph** — multi-agent StateGraph pipeline: `log_inspector → rag_lookup → triage_router`
- **Ollama** (default) — fully local LLM reasoning + embeddings, no API key, no per-token cost.
  Swap to Claude (`anthropic` SDK) via one config value — see [LLM backend](#llm-backend--embeddings).

## Project layout

```
docker-compose.yml            # postgres (pgvector) + redis, with healthchecks
init-db/                      # runs once on first postgres boot: CREATE EXTENSION vector
requirements.txt
.env.example                  # copy to .env and adjust if needed
scripts/
  seed_tickets.py              # inserts 8 verified historical tickets with real embeddings
  vector_index_analysis.sql    # EXPLAIN ANALYZE walkthrough: HNSW vs IVFFlat vs seq scan
app/
  config.py                    # Settings (pydantic-settings): DB/Redis/LLM/embedding/RAG config
  models.py                    # Ticket ORM model (Vector(EMBEDDING_DIM) column, RAG fields)
  database.py                  # async engine/session (lazy init), HNSW index bootstrap
  schemas.py                   # TicketCreate / TicketResponse + guardrails
  graph.py                     # TicketState + log_inspector/rag_lookup/triage_router nodes
  main.py                      # FastAPI app, lifespan wiring, POST /api/v1/triage
  services/
    embeddings.py               # Ollama / deterministic embedding backends
    llm.py                      # Ollama / Anthropic structured-output LLM backends
    rag.py                      # pgvector cosine similarity retrieval
```

## Running it

1. Start infrastructure:

   ```bash
   docker compose up -d
   ```

   Wait for both services to report healthy: `docker compose ps`.

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
   convenience — swap for Alembic migrations in production) and opens the
   Redis-backed LangGraph checkpointer.

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
     "created_at": "…"
   }
   ```

   Interactive docs: http://127.0.0.1:8000/docs

8. Inspect vector index performance:

   ```bash
   docker exec -i triage_postgres psql -U triage_user -d triage_engine < scripts/vector_index_analysis.sql
   ```

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
validation after retry — the endpoint degrades gracefully instead of 502ing
the whole request when a local model server is briefly down.

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

## Notes / what's intentionally out of scope for Phase 2

- No OpenTelemetry tracing, confidence-based human-escalation node, or RFC
  7807 error responses yet — the `confidence` field is captured on every
  response in preparation for that, but nothing routes on it yet (Phase 3).
- Table/index creation uses `Base.metadata.create_all` + raw DDL for
  zero-setup local dev; a real deployment should switch to Alembic
  migrations.
