# Deployment

The public demo runs in two pieces:

| Piece | Where | Why |
| --- | --- | --- |
| **Backend** (FastAPI + Postgres/pgvector + Redis + the Docker sandbox) | one small cloud VM running `docker compose` | `verify_fix_node` needs a real Docker daemon to run the monitored repo's tests in a throwaway container. That rules out a normal PaaS. |
| **Frontend** (Next.js) | Vercel | Static/SSR, zero infra, free tier is plenty. It only talks to the backend over HTTPS. |

`SANDBOX_ENABLED` stays **on** in the hosted demo — the sandboxed-verification loop is the whole point of the project, so the demo must run it live, not fake it.

---

## 1. Provision the VM

Any provider works. Minimum realistic spec, because the sandbox pulls `node:20-slim` + `postgres:16` and runs `npm ci`:

- **2 vCPU / 4 GB RAM / 40 GB disk** (Hetzner CX22, DigitalOcean `s-2vcpu-4gb`, etc.)
- Ubuntu 24.04 LTS
- Open inbound TCP **22** (SSH) and **443** (HTTPS). Do **not** expose 8000 publicly — the reverse proxy handles TLS and forwards to it locally.

Install Docker Engine + compose plugin:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"   # log out/in so the group applies
docker compose version            # sanity check
```

---

## 2. Get the code + config onto the VM

```bash
git clone https://github.com/<you>/FDE-Project.git
cd FDE-Project
git checkout phase-1-sandbox-verification   # until this merges to main
cp .env.example .env
```

Edit `.env` for the compose-prod topology:

```dotenv
APP_ENV=production
LOG_LEVEL=INFO

# The backend container reaches Postgres/Redis by SERVICE NAME, not localhost.
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
REDIS_HOST=redis
REDIS_PORT=6379
POSTGRES_USER=triage_user
POSTGRES_PASSWORD=<a real password>
POSTGRES_DB=triage_engine

# LLM: Ollama is not practical on a 4 GB VM. Use Anthropic.
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-5
ANTHROPIC_API_KEY=<your key>
# Embeddings still need a backend. Simplest: run Ollama on the VM just for
# nomic-embed-text (small, CPU-fine), or switch EMBEDDING_PROVIDER later.
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768

# Sandbox
SANDBOX_ENABLED=true

# GitHub auto-fix — the monitored repo + a fine-grained PAT (Contents RW + PRs RW)
GITHUB_TOKEN=<fine-grained PAT scoped to the monitored repo>
GITHUB_REPO=yom4n/todotest
GITHUB_BASE_BRANCH=main

# CORS — lock to the Vercel origin once you have it
CORS_ALLOW_ORIGINS=https://<your-frontend>.vercel.app
```

> **Embeddings note:** the 4 GB VM can run `ollama serve` with only `nomic-embed-text` pulled (~275 MB, CPU inference is fine for one embedding per ticket). Install Ollama on the host (`curl -fsSL https://ollama.com/install.sh | sh`), `ollama pull nomic-embed-text`, and the compose file's `host.docker.internal` mapping lets the container reach it. If the VM is too small, set `EMBEDDING_PROVIDER=deterministic` — RAG retrieval degrades to lexical matching but the demo still runs end to end.

---

## 3. Bring the stack up

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps        # all healthy?
docker compose -f docker-compose.prod.yml logs -f backend
```

On first boot the backend runs `init_models()` (creates the `tickets` table, the HNSW index, the Phase 1 columns).

Seed the RAG corpus:

```bash
docker compose -f docker-compose.prod.yml exec backend python scripts/seed_tickets.py
```

Smoke-test locally on the VM:

```bash
curl -s localhost:8000/health
curl -sN "localhost:8000/api/v1/triage/stream?title=test&stack_trace=$(python3 -c 'import urllib.parse;print(urllib.parse.quote("KeyError: '\''x'\''\n  File \"h.py\", line 3"))')&environment=staging"
```

---

## 4. TLS + reverse proxy (Caddy — one file, auto HTTPS)

Point a DNS `A` record (e.g. `fde-api.yourdomain.com`) at the VM, then:

```bash
sudo apt install -y caddy
```

`/etc/caddy/Caddyfile`:

```
fde-api.yourdomain.com {
    reverse_proxy localhost:8000 {
        # SSE needs unbuffered proxying
        flush_interval -1
    }
}
```

```bash
sudo systemctl reload caddy
```

Caddy fetches a Let's Encrypt cert automatically. Verify: `curl https://fde-api.yourdomain.com/health`.

> `flush_interval -1` is essential — without it Caddy buffers the response and the live agent-run timeline arrives all at once instead of streaming.

---

## 5. Deploy the frontend to Vercel

```bash
cd frontend
npx vercel            # first run links the project
npx vercel --prod
```

In the Vercel project settings → Environment Variables:

```
BACKEND_API_URL = https://fde-api.yourdomain.com
```

Redeploy. The Next.js route handlers (`/api/triage`, `/api/triage/stream`, `/api/tickets`, `/api/health`) proxy to that.

Then tighten the backend's `CORS_ALLOW_ORIGINS` in `.env` to the exact Vercel URL and `docker compose -f docker-compose.prod.yml up -d` to reload.

---

## 6. Keeping it running

- `restart: unless-stopped` is already set on every service, so a VM reboot brings the stack back.
- **Cost guard:** `LLM_PROVIDER=anthropic` means every triage run costs tokens. The demo has no auth or rate limiting yet (see the README's "intentionally out of scope"). For a public link, either put Caddy `basicauth` in front of it, or accept the spend and watch it.
- **Disk:** the sandbox leaves no containers behind (they're `--rm`'d), but pulled images accumulate. `docker image prune -a` monthly.
- **Updates:** `git pull && docker compose -f docker-compose.prod.yml up -d --build`.

---

## Alternative: no public deploy

If you don't want to run a VM, the project still demos fully on a laptop:

```bash
docker compose up -d          # postgres + redis
ollama serve                  # + ollama pull qwen2.5:7b-instruct nomic-embed-text
uvicorn app.main:app --port 8000
cd frontend && npm run dev
```

For sharing, record a screen capture of the Live Run tab firing the todotest 504 incident end to end (triage → generate → sandbox verify → PR). The eval-harness numbers (Phase 4) and the writeup (Phase 5) carry the rest of the story in a DM without needing a live link.
