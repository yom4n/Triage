# Autonomous Tier-1 Technical Support & Triage Engine

This is a working local support-triage agent for software incidents: it accepts human tickets, browser crashes, and Prometheus metric alerts, then runs the same LangGraph pipeline to classify the failure, retrieve verified prior fixes, decide whether to escalate, and, when configured, draft a code fix. The centerpiece is the sandboxed verification loop: a generated fix is applied to the monitored repo in Docker and must pass that repo's own tests before the system opens a PR.

One-command detector demo:

```bash
docker compose -f docker-compose.demo.yml up --build
```

Core API demo path:

```bash
docker compose up -d
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
pip install -r requirements.txt
python scripts/seed_tickets.py
uvicorn app.main:app --reload
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Checkout fails intermittently under load",
    "environment": "production",
    "stack_trace": "OperationalError: connection to server timed out\n  File \"checkout.py\", line 91, in charge_card"
  }'
```

## Feature Map

| Phase | What shipped |
| --- | --- |
| Phase 1 | Sandboxed fix verification: `generate_fix -> verify_fix -> open_pr` or `fix_escalation`, with Docker running the monitored repo's own tests before a verified PR is opened. |
| Phase 2 | Live SSE agent-run view, frontend scaffolding, production compose, and deployment notes for VM + Vercel. |
| Phase 3 | Proactive Prometheus detector: three metric rules synthesize `MetricAlert` objects and enter the same triage pipeline with `source="metric"`. |
| Phase 4 | Eval harness with 16 ground-truthed incident cases, `evals/scorecard.json`, `evals/REPORT.md`, and `GET /api/v1/evals/scorecard`. |
| Phase 5 | Reviewer-facing documentation: [WRITEUP.md](WRITEUP.md), this README, and the updated [ARCHITECTURE.md](ARCHITECTURE.md). |

## What You Need Installed First

- **Docker**: runs Postgres/pgvector, Redis, the demo stack, and the fix-verification sandbox.
- **Python 3.11+**
- **Ollama**: runs the default local chat and embedding model.
- **Node.js**: only needed for the Next.js dashboard.
- **GitHub token**: only needed if you want the system to push branches and open PRs.

## Quickstart

1. Start infrastructure:

   ```bash
   docker compose up -d
   ```

   Wait for both services to report healthy:

   ```bash
   docker compose ps
   ```

   Postgres is mapped to host port `55432`, not `5432`, because many Windows machines already have a local Postgres service on the default port. `.env.example` already matches this mapping. If `55432` collides too, change both `docker-compose.yml` and `POSTGRES_PORT` together.

2. Install and start Ollama on the host, then pull the models:

   ```bash
   ollama pull qwen2.5:7b-instruct   # chat model: extraction + triage + fix reasoning
   ollama pull nomic-embed-text      # embedding model: 768-dim, for RAG
   ```

   `ollama serve` normally starts automatically after install. If it does not, run it in its own terminal.

3. Create a virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` if you need to change defaults:

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Seed the verified RAG corpus:

   ```bash
   python scripts/seed_tickets.py
   ```

6. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

   Interactive docs: http://127.0.0.1:8000/docs

7. Send a ticket:

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

8. Inspect vector index performance:

   ```bash
   docker exec -i triage_postgres psql -U triage_user -d triage_engine < scripts/vector_index_analysis.sql
   ```

9. Run resilience drills:

   ```bash
   python scripts/failure_drills.py
   ```

10. Run evals:

   ```bash
   python -m evals.runner --dry-run
   python -m evals.runner --offline
   python -m evals.runner
   ```

## Demo App And Detector

Run the monitored demo app, Prometheus, and detector-oriented compose stack:

```bash
docker compose -f docker-compose.demo.yml up --build
```

The demo app exposes `/work`, `/metrics`, and `POST /admin/fault` with JSON like:

```json
{"name":"latency","enabled":true}
```

Supported faults are `latency`, `errors`, and `dependency_down`. The detector watches Prometheus and files metric-sourced tickets when configured thresholds breach.

## Frontend

The optional dashboard shows tickets and live agent-run progress.

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## Important Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/triage` | File a human ticket. |
| `GET /api/v1/triage/stream` | Run triage with a live SSE event stream for the UI. |
| `POST /api/v1/ingest/crash` | Ingest a browser/client crash from `client-sdks/react`. |
| `POST /api/v1/ingest/alert` | Ingest a Prometheus-style metric alert. |
| `GET /api/v1/tickets` | List persisted tickets. |
| `GET /api/v1/detector/status` | Inspect detector rule state and last observed values. |
| `GET /api/v1/evals/scorecard` | Return the committed eval scorecard JSON. |

## Environment Notes

The local defaults work without paid APIs:

- `LLM_PROVIDER=ollama`
- `OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct`
- `EMBEDDING_PROVIDER=ollama`
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`
- `EMBEDDING_DIM=768`

To open real PRs, configure:

- `GITHUB_REPO`, for example `your-username/your-repo`
- `GITHUB_TOKEN`, a fine-grained token with repository Contents read/write and Pull Requests read/write
- `GITHUB_BASE_BRANCH`, usually `main`

Without GitHub settings, triage and evals still work. The fix sub-graph records why it stopped instead of pretending a PR was opened.

## Read More

- [WRITEUP.md](WRITEUP.md): reviewer narrative and phase-by-phase explanation.
- [ARCHITECTURE.md](ARCHITECTURE.md): system internals, graph routing, storage, observability, detector, evals, and scope.
- [DEPLOY.md](DEPLOY.md): VM + Caddy + Vercel deployment path.
- [evals/README.md](evals/README.md): eval commands and scoring modes.
- [demo-app/README.md](demo-app/README.md): detector demo app.

## Intentionally Out Of Scope

This is a demo/portfolio system, not a production support platform. It intentionally omits auth, rate limiting, source-map de-minification, crash deduplication, a ticket-verification admin workflow, and production migrations. Nothing is merged automatically; every PR is for human review.
