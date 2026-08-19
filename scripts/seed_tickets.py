"""
Seed the `tickets` table with verified historical bug tickets so
`rag_lookup_node` (app/graph.py) has a real corpus to search against.

Each seed ticket is embedded with the exact same `build_embedding_text` /
`generate_embedding` path the live API uses (app/services/embeddings.py),
so cosine similarity between a *new* incoming ticket and these rows is
meaningful from the very first run -- not a placeholder that needs
re-embedding later.

Usage
-----
    # from the project root, with the venv activated and Postgres/Ollama running
    python scripts/seed_tickets.py            # insert any seed tickets not already present
    python scripts/seed_tickets.py --reset     # delete existing seed tickets first, then re-insert

Requires `ollama serve` running with the embedding model pulled, unless
EMBEDDING_PROVIDER=deterministic is set (see app/config.py) -- the
deterministic backend produces reproducible but non-semantic vectors,
useful for a dependency-free smoke test of the pipeline wiring.
"""
import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Allow running as `python scripts/seed_tickets.py` from the repo root
# without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.database import get_session_maker, init_models  # noqa: E402
from app.models import Ticket  # noqa: E402
from app.services.embeddings import EmbeddingError, build_embedding_text, generate_embedding  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_tickets")


# Eight verified historical tickets spanning every severity and all three
# environments, so the seeded corpus can demonstrate CRITICAL/HIGH/MEDIUM/LOW
# retrieval and cross-environment behavior out of the box. `resolution` is
# the payload RAG retrieval exists to surface -- it's what triage_router_node
# actually reads and grounds its recommendation in.
SEED_TICKETS: list[dict] = [
    {
        "title": "Checkout service hangs under peak load",
        "environment": "production",
        "stack_trace": (
            "psycopg2.OperationalError: connection to server at \"db-primary\" "
            "port 5432 failed: timeout expired\n"
            "  File \"checkout.py\", line 88, in charge_card\n"
            "    cursor.execute(query, params)\n"
            "  File \"db/pool.py\", line 41, in get_connection\n"
            "    return self._pool.getconn(timeout=5)"
        ),
        "extracted_error": "OperationalError",
        "exception_message": "connection to server timeout expired",
        "affected_file": "checkout.py",
        "affected_line": 88,
        "severity": "CRITICAL",
        "resolution_steps": [
            "Confirm connection pool exhaustion via `SELECT count(*) FROM pg_stat_activity`.",
            "Increase PgBouncer pool_size and set a hard statement_timeout on the checkout role.",
            "Add a circuit breaker around the payment DB call so it fails fast under saturation instead of queueing.",
        ],
        "resolution": (
            "Root cause was connection pool exhaustion during a flash-sale traffic spike: "
            "a slow reporting query was holding connections open. Fixed by adding PgBouncer "
            "in transaction-pooling mode in front of Postgres, capping pool_size at 40, and "
            "moving the reporting query to a read replica."
        ),
    },
    {
        "title": "Null pointer crash on order confirmation screen",
        "environment": "production",
        "stack_trace": (
            "Exception in thread \"main\" java.lang.NullPointerException: "
            "Cannot invoke \"Order.getShippingAddress()\" because \"order\" is null\n"
            "\tat com.flodata.orders.OrderConfirmationService.render(OrderConfirmationService.java:142)\n"
            "\tat com.flodata.orders.OrderController.confirm(OrderController.java:57)"
        ),
        "extracted_error": "NullPointerException",
        "exception_message": "Cannot invoke Order.getShippingAddress() because order is null",
        "affected_file": "OrderConfirmationService.java",
        "affected_line": 142,
        "severity": "HIGH",
        "resolution_steps": [
            "Add a null-guard around order lookup in OrderConfirmationService.render().",
            "Return a 404 with a user-facing 'order not found' message instead of crashing the request.",
            "Add a regression test for a confirmation request with an already-cancelled order ID.",
        ],
        "resolution": (
            "The order had been cancelled and purged by a background job between page load and "
            "confirmation click, so the lookup returned null but the code assumed it always "
            "succeeded. Fixed with an explicit null check that returns a friendly 'this order is "
            "no longer available' response."
        ),
    },
    {
        "title": "Redis connection refused across all app pods",
        "environment": "production",
        "stack_trace": (
            "redis.exceptions.ConnectionError: Error 111 connecting to redis-cache:6379. "
            "Connection refused.\n"
            "  File \"cache/client.py\", line 23, in get\n"
            "    return self._client.get(key)"
        ),
        "extracted_error": "ConnectionError",
        "exception_message": "Error 111 connecting to redis-cache:6379. Connection refused.",
        "affected_file": "cache/client.py",
        "affected_line": 23,
        "severity": "CRITICAL",
        "resolution_steps": [
            "Check `redis-cli -h redis-cache ping` and Redis maxclients (`CONFIG GET maxclients`).",
            "Audit worker code for unclosed Redis connections leaking across requests.",
            "Add explicit connection pooling with a bounded max size and pool health checks.",
        ],
        "resolution": (
            "A recent deploy introduced a code path that opened a new Redis connection per "
            "request without closing it, exhausting Redis's maxclients limit under normal "
            "traffic. Fixed by switching to a shared, bounded connection pool and bumping "
            "maxclients as a secondary safety margin."
        ),
    },
    {
        "title": "Gateway timeouts during nightly batch export",
        "environment": "production",
        "stack_trace": (
            "504 Gateway Timeout\n"
            "upstream timed out (110: Connection timed out) while reading response header "
            "from upstream, request: \"POST /api/exports/batch HTTP/1.1\""
        ),
        "extracted_error": "504 Gateway Timeout",
        "exception_message": "upstream timed out while reading response header",
        "affected_file": None,
        "affected_line": None,
        "severity": "CRITICAL",
        "resolution_steps": [
            "Raise nginx proxy_read_timeout for the /api/exports/ location block.",
            "Move batch export generation to an async job with a polling/webhook completion pattern.",
            "Scale the export-worker deployment's replica count during the nightly window.",
        ],
        "resolution": (
            "The batch export endpoint was synchronous and occasionally took longer than nginx's "
            "60s proxy_read_timeout once the export volume grew. Converted the endpoint to enqueue "
            "a background job and return a job ID immediately; the client polls for completion."
        ),
    },
    {
        "title": "Deadlock detected during concurrent inventory updates",
        "environment": "production",
        "stack_trace": (
            "sqlalchemy.exc.OperationalError: (psycopg2.errors.DeadlockDetected) "
            "deadlock detected\n"
            "DETAIL:  Process 4821 waits for ShareLock on transaction 9931; "
            "Process 4903 waits for ShareLock on transaction 9928.\n"
            "  File \"inventory/service.py\", line 61, in reserve_stock\n"
            "    session.execute(update_stmt)"
        ),
        "extracted_error": "DeadlockDetected",
        "exception_message": "deadlock detected",
        "affected_file": "inventory/service.py",
        "affected_line": 61,
        "severity": "CRITICAL",
        "resolution_steps": [
            "Always acquire row locks on inventory rows in a consistent order (sort by product_id).",
            "Wrap the update in a retry-with-backoff decorator for DeadlockDetected specifically.",
            "Add a load test that simulates concurrent stock reservations for the same SKU set.",
        ],
        "resolution": (
            "Two code paths updated inventory rows for the same set of SKUs in opposite order "
            "under concurrent load, producing a classic deadlock. Fixed by always sorting SKUs "
            "before locking and adding an automatic retry on DeadlockDetected."
        ),
    },
    {
        "title": "KeyError when processing webhook payloads missing optional field",
        "environment": "staging",
        "stack_trace": (
            "KeyError: 'customer_id'\n"
            "  File \"webhooks/handler.py\", line 34, in process_payload\n"
            "    customer_id = payload['customer_id']"
        ),
        "extracted_error": "KeyError",
        "exception_message": "'customer_id'",
        "affected_file": "webhooks/handler.py",
        "affected_line": 34,
        "severity": "MEDIUM",
        "resolution_steps": [
            "Replace payload['customer_id'] with payload.get('customer_id') and handle the None case explicitly.",
            "Add schema validation for inbound webhook payloads before processing.",
        ],
        "resolution": (
            "A partner's webhook payload schema changed to make `customer_id` optional for "
            "guest checkouts, which the handler assumed was always present. Fixed with a "
            "defensive `.get()` and a documented fallback for guest-checkout payloads."
        ),
    },
    {
        "title": "Cart total sometimes shows NaN on the frontend",
        "environment": "staging",
        "stack_trace": (
            "Uncaught TypeError: Cannot read properties of undefined (reading 'price')\n"
            "    at calculateTotal (cart.js:19)\n"
            "    at updateCartSummary (cart.js:44)"
        ),
        "extracted_error": "TypeError",
        "exception_message": "Cannot read properties of undefined (reading 'price')",
        "affected_file": "cart.js",
        "affected_line": 19,
        "severity": "MEDIUM",
        "resolution_steps": [
            "Guard calculateTotal() against cart line items still resolving from an async fetch.",
            "Add a loading state so the UI doesn't render cart totals before line items are populated.",
        ],
        "resolution": (
            "calculateTotal() ran against a cart item array that hadn't finished loading yet, so "
            "some entries were `undefined`. Fixed by gating total calculation behind a "
            "cartItemsLoaded flag and rendering a skeleton state until then."
        ),
    },
    {
        "title": "Misaligned checkout button on mobile Safari",
        "environment": "development",
        "stack_trace": (
            "No runtime error -- visual regression caught in review.\n"
            "CSS: .checkout-button { display: flex; } renders with a 12px offset on iOS Safari "
            "15.x due to a flexbox gap property fallback difference.\n"
            "File: styles/checkout.css, selector .checkout-button, line 27."
        ),
        "extracted_error": "CSS layout regression",
        "exception_message": "flexbox gap fallback offset on iOS Safari 15.x",
        "affected_file": "styles/checkout.css",
        "affected_line": 27,
        "severity": "LOW",
        "resolution_steps": [
            "Replace `gap` with explicit margins on .checkout-button children for Safari <16 compatibility.",
            "Add an iOS Safari 15 case to the visual regression test matrix.",
        ],
        "resolution": (
            "iOS Safari 15 doesn't support `gap` inside flex containers, only grid. Replaced the "
            "flex `gap` with explicit child margins, which is supported everywhere in the target "
            "browser matrix."
        ),
    },
]


async def seed(reset: bool) -> None:
    await init_models()  # idempotent: ensures pgvector extension, tables, HNSW index exist
    session_maker = get_session_maker()

    async with session_maker() as session:
        if reset:
            titles = [t["title"] for t in SEED_TICKETS]
            existing = (
                await session.execute(select(Ticket).where(Ticket.title.in_(titles)))
            ).scalars().all()
            for row in existing:
                await session.delete(row)
            await session.commit()
            logger.info("Deleted %d existing seed ticket(s)", len(existing))

        inserted = 0
        skipped = 0
        for seed_data in SEED_TICKETS:
            already_present = (
                await session.execute(select(Ticket.id).where(Ticket.title == seed_data["title"]))
            ).scalar_one_or_none()
            if already_present is not None:
                skipped += 1
                logger.info("Skipping (already seeded): %s", seed_data["title"])
                continue

            embedding_text = build_embedding_text(
                title=seed_data["title"],
                extracted_error=seed_data["extracted_error"],
                stack_trace=seed_data["stack_trace"],
                description="",
            )
            try:
                embedding = await generate_embedding(embedding_text)
            except EmbeddingError as exc:
                logger.error(
                    "Failed to embed seed ticket %r: %s\n"
                    "Is Ollama running (`ollama serve`) with the embedding model pulled "
                    "(`ollama pull <model>`)? Or set EMBEDDING_PROVIDER=deterministic for "
                    "a dependency-free dry run.",
                    seed_data["title"], exc,
                )
                raise

            ticket = Ticket(
                id=uuid.uuid4(),
                title=seed_data["title"],
                stack_trace=seed_data["stack_trace"],
                environment=seed_data["environment"],
                description=None,
                extracted_error=seed_data["extracted_error"],
                exception_message=seed_data["exception_message"],
                affected_file=seed_data["affected_file"],
                affected_line=seed_data["affected_line"],
                severity=seed_data["severity"],
                summary=f"[{seed_data['severity']}] {seed_data['exception_message']} ({seed_data['title']})",
                resolution_steps=seed_data["resolution_steps"],
                confidence=1.0,  # human-authored seed data, not an LLM guess
                resolution=seed_data["resolution"],
                is_verified=True,  # seed data represents confirmed, real-world fixes
                embedding_text=embedding_text,
                embedding=embedding,
            )
            session.add(ticket)
            inserted += 1
            logger.info("Seeded: %s [%s]", seed_data["title"], seed_data["severity"])

        await session.commit()

    logger.info("Done. Inserted %d, skipped %d (already present).", inserted, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing seed tickets (matched by title) before re-inserting them.",
    )
    args = parser.parse_args()
    asyncio.run(seed(reset=args.reset))


if __name__ == "__main__":
    main()
