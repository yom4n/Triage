# Autonomous Tier-1 Technical Support & Triage Engine

This project watches an app for crashes and handles the boring first steps of fixing them automatically:

1. Something breaks in the monitored app (a user's browser throws an error).
2. A support ticket is created automatically — no one had to file it.
3. A local AI model reads the error, figures out how serious it is, and writes up what's wrong and how to fix it. It also checks past tickets to see if something similar happened before.
4. If the AI is confident about the fix, it writes the corrected code itself and opens a real Pull Request on GitHub for a human to review.
5. If the AI isn't confident, it skips the fix and flags the ticket for a human instead — it never guesses and pretends to be sure.

Nothing gets merged automatically. A person always reviews the PR before it goes anywhere.

## How it works, step by step

```
crash happens  →  ticket created  →  AI reads the error  →  AI checks similar past tickets
                                                                        │
                                                confident?  ────────────┤
                                                   │                    │
                                                  yes                   no
                                                   │                    │
                                     AI rewrites the file        ticket flagged for
                                     and opens a GitHub PR        a human to look at
```

There's also a small dashboard (a web page) where you can see every ticket, its status, and the PR link if one was opened.

## What you need installed first

- **Docker** — runs the database (Postgres) and a small in-memory store (Redis)
- **Python 3.11+**
- **[Ollama](https://ollama.com)** — a free program that runs the AI model on your own machine, no API key needed
- **Node.js** — only if you want to run the dashboard web page
- A **GitHub account + token** — only if you want the "open a Pull Request" part to actually work (optional, explained below)

## Setup

```bash
# 1. Start the database and Redis
docker compose up -d

# 2. Install Ollama, then download the two models this project uses
ollama pull qwen2.5:7b-instruct     # the AI model that reads errors and reasons about them
ollama pull nomic-embed-text        # the model that compares tickets for similarity

# 3. Install the Python dependencies
pip install -r requirements.txt

# 4. Create your own .env file from the template
cp .env.example .env
# now open .env and fill in anything mentioned below

# 5. (optional) load a few sample tickets so the AI has history to compare against
python scripts/seed_tickets.py

# 6. Start the server
uvicorn app.main:app --reload
```

Once it's running:
- API docs (test it in the browser): http://127.0.0.1:8000/docs
- Dashboard (optional): `cd frontend`, then `npm install`, then `npm run dev` → http://localhost:3000

## The .env file — what each key means

Open `.env.example`, copy it to `.env`, and fill it in. Most values already have working defaults — here's what actually matters:

**You don't need to touch these** — they already match the Docker setup:
`POSTGRES_*`, `REDIS_*`

**AI model settings — defaults work out of the box, no key needed:**
- `LLM_PROVIDER=ollama` — uses the free local AI model. Leave this alone unless you'd rather pay for Claude instead (see below).
- Only if you want to use Claude instead of the free local model: set `LLM_PROVIDER=anthropic` and add `ANTHROPIC_API_KEY=sk-ant-...` (get one at [console.anthropic.com](https://console.anthropic.com)).

**GitHub settings — optional, only needed for the "open a Pull Request" step:**
Without these, everything still works — the AI still drafts a fix, you just won't get an actual PR, only the suggested code change.

- `GITHUB_REPO` — the repo the AI is allowed to fix, written as `your-username/your-repo`.
- `GITHUB_TOKEN` — a key that lets this app push code and open PRs on your behalf. To create one:
  1. Go to GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
  2. Under "Repository access", pick the one repo you're monitoring (the one in `GITHUB_REPO`).
  3. Under "Permissions", turn on:
     - **Contents: Read and write**
     - **Pull requests: Read and write**
  4. Generate the token and paste it in as `GITHUB_TOKEN=github_pat_...`.

  Without both of those two permissions checked, GitHub will reject the request with a `403` error when the app tries to open a branch.

## Try it

```bash
curl -X POST http://127.0.0.1:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Checkout fails intermittently under load",
    "environment": "production",
    "stack_trace": "OperationalError: connection to server timed out\n  File \"checkout.py\", line 91, in charge_card"
  }'
```

You'll get back a ticket with a severity, a plain-English summary, and next steps — or a PR link if it was confident enough to fix it itself.

## The three endpoints that matter

| Endpoint | What it's for |
|---|---|
| `POST /api/v1/triage` | File a ticket by hand |
| `POST /api/v1/ingest/crash` | Used by the crash-reporting script (`client-sdks/react`) to file tickets automatically |
| `GET /api/v1/tickets` | List every ticket — this is what the dashboard reads from |

## What this project intentionally doesn't do

This is a demo/portfolio project, not something ready for production traffic:

- No login or access control on the API
- No automated test suite
- If the same crash happens 1,000 times, it makes 1,000 tickets (no deduplication)

The reasoning behind each design choice is explained in comments in the code itself.

For the full technical write-up (architecture, observability, chaos drills, phase-by-phase history), see [ARCHITECTURE.md](ARCHITECTURE.md).
