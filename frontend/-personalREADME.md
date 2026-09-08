# FDE Incident Triage Frontend

Next.js frontend for the current FastAPI triage backend.

## Run

```bash
npm install
npm run dev
```

The app runs at http://127.0.0.1:3000 by default.

## Backend Connection

The frontend calls its own server routes:

- `POST /api/triage`
- `GET /api/health`

Those routes proxy to the FastAPI backend at:

```bash
BACKEND_API_URL=http://127.0.0.1:8000
```

Set `BACKEND_API_URL` in `.env.local` if the FastAPI service runs elsewhere.

## Current Backend Limits

The current backend only exposes ticket creation/triage, so the dashboard stores submitted incidents in browser local storage. When the backend adds incident list/detail/status APIs, this UI can switch from local history to persisted records.
