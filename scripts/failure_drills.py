"""
Chaos-engineering / failure-mode drills for the triage engine.

Simulates three production failure scenarios against POST /api/v1/triage,
entirely in-process (via FastAPI's TestClient, which drives app.main:app
through its real lifespan -- Postgres, Redis, and the compiled LangGraph
pipeline all come up exactly as they would under uvicorn) with each
failure injected via `unittest.mock.patch` rather than an external chaos
tool, so these drills run repeatably in CI with no separate infrastructure
to stand up or tear down.

Usage
-----
    # from the project root, with the venv activated and Postgres/Redis
    # running (`docker compose up -d`). Ollama is NOT required -- every
    # drill below patches the DB or LLM call site directly, so none of
    # them actually reach Ollama or any live network dependency.
    python scripts/failure_drills.py

Drill A -- Database Outage
    Patches `AsyncSession.commit` to raise a SQLAlchemy `OperationalError`
    mid-request -- after the graph has already produced a full triage
    result, exactly like a real connection drop during the final ticket
    persist. Proves the global `database_error_handler` (app/main.py)
    returns a structured 503 RFC 7807 body with a `trace_id` instead of
    leaking a bare, unstructured 500.

Drill B -- API Rate Limit
    Patches `app.graph.call_structured` (the single choke point both
    LLM-backed nodes call through) to raise `LLMError` mimicking a 429
    from the provider. Proves log_inspector_node / triage_router_node's
    internal fallback (app/graph.py) degrades gracefully: the request
    still returns 201, with `status=ESCALATED_TO_HUMAN` and confidence
    0.0, rather than 5xx-ing or hanging on the outage.

Drill C -- Corrupted Response
    Patches the same choke point to raise `LLMMalformedOutputError`
    carrying an unparseable JSON fragment, simulating a model that broke
    its structured-output contract. Proves the same graceful-degradation
    path fires for a *different* failure flavor than Drill B -- the
    fallback is keyed on "the LLM call didn't succeed", not on one
    specific exception subtype.

Each drill also implicitly exercises the Redis-backed LangGraph
checkpointer: every ticket gets a fresh `thread_id` (its UUID), and a
checkpoint is written after every node regardless of which branch the
graph took -- `redis-cli --scan --pattern 'checkpoint:*'` after running
this script will show one checkpoint stream per drill, proving state was
held even for the request that ended in an exception.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Allow running as `python scripts/failure_drills.py` from the repo root
# without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

import app.graph as graph_module  # noqa: E402
from app.main import app  # noqa: E402
from app.services.llm import LLMError, LLMMalformedOutputError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("failure_drills")


def _sample_payload(title: str) -> dict:
    """A realistic, schema-valid ticket payload -- see app/schemas.py's TicketCreate constraints."""
    return {
        "title": title,
        "stack_trace": (
            "Traceback (most recent call last):\n"
            '  File "app/worker.py", line 88, in process_job\n'
            "    conn = pool.acquire(timeout=5)\n"
            "TimeoutError: connection pool exhausted after 5s"
        ),
        "environment": "production",
        "description": "Chaos drill synthetic ticket -- not a real incident.",
    }


def _print_result(drill_name: str, response) -> None:
    logger.info("=" * 78)
    logger.info("%s -- HTTP %s", drill_name, response.status_code)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    logger.info("Response body: %s", body)
    logger.info("=" * 78)


def drill_a_database_outage(client: TestClient) -> None:
    """Drill A: simulate a DB connection drop mid-triage (during the final commit)."""
    logger.info("Drill A: Database Outage -- patching AsyncSession.commit to raise OperationalError")

    async def _raise_operational_error(self, *args, **kwargs):
        # OperationalError is what asyncpg/SQLAlchemy actually raises for
        # a dropped connection / server restart mid-transaction -- using
        # the real exception class (not a generic RuntimeError) is what
        # proves `database_error_handler`'s `SQLAlchemyError` catch
        # actually matches a realistic failure, not a contrived one.
        raise OperationalError("COMMIT", {}, Exception("simulated connection drop"))

    with patch.object(AsyncSession, "commit", new=_raise_operational_error):
        response = client.post("/api/v1/triage", json=_sample_payload("Drill A: DB outage during ticket persist"))

    _print_result("Drill A (Database Outage)", response)

    assert response.status_code == 503, f"expected 503, got {response.status_code}"
    body = response.json()
    assert body["type"].endswith("/database-error"), body
    assert body.get("trace_id") and body["trace_id"] != "unavailable", "expected a real OTel trace_id"
    assert response.headers["content-type"].startswith("application/problem+json"), response.headers
    logger.info("Drill A PASSED: DB outage returned a structured 503 RFC 7807 body, trace_id=%s", body["trace_id"])


def drill_b_rate_limit(client: TestClient) -> None:
    """Drill B: simulate a 429 rate-limit response from the LLM provider."""
    logger.info("Drill B: API Rate Limit -- patching call_structured to raise a 429-style LLMError")

    async def _raise_rate_limited(**kwargs):
        raise LLMError(
            "Ollama chat request failed with 429: Too Many Requests -- "
            "simulated provider rate limit for chaos drill B"
        )

    # Patched on the graph module, not app.services.llm: log_inspector_node
    # and triage_router_node imported `call_structured` by name (`from
    # app.services.llm import call_structured`), so the name they actually
    # call at runtime lives in app.graph's namespace -- patching the
    # origin module would leave their already-bound reference untouched.
    with patch.object(graph_module, "call_structured", new=AsyncMock(side_effect=_raise_rate_limited)):
        response = client.post("/api/v1/triage", json=_sample_payload("Drill B: LLM provider rate limited"))

    _print_result("Drill B (API Rate Limit)", response)

    assert response.status_code == 201, f"expected 201 (graceful degradation), got {response.status_code}"
    body = response.json()
    assert body["status"] == "ESCALATED_TO_HUMAN", body
    assert body["confidence"] == 0.0, body
    assert body["resolution_steps"], "escalation must still return an actionable hand-off step"
    logger.info("Drill B PASSED: rate-limited LLM degraded to ESCALATED_TO_HUMAN, no 5xx, no dropped ticket data")


def drill_c_corrupted_response(client: TestClient) -> None:
    """Drill C: simulate an unparseable/malformed JSON payload returned by an LLM node."""
    logger.info("Drill C: Corrupted Response -- patching call_structured to raise LLMMalformedOutputError")

    async def _raise_malformed(**kwargs):
        raise LLMMalformedOutputError(
            'Ollama returned output that failed schema validation: '
            '\'{"severity": "CRITICAL", "summary": \' (truncated/unparseable JSON) -- '
            "simulated corrupted response for chaos drill C"
        )

    with patch.object(graph_module, "call_structured", new=AsyncMock(side_effect=_raise_malformed)):
        response = client.post("/api/v1/triage", json=_sample_payload("Drill C: corrupted LLM JSON payload"))

    _print_result("Drill C (Corrupted Response)", response)

    assert response.status_code == 201, f"expected 201 (graceful degradation), got {response.status_code}"
    body = response.json()
    assert body["status"] == "ESCALATED_TO_HUMAN", body
    assert body["confidence"] == 0.0, body
    logger.info("Drill C PASSED: corrupted LLM output degraded to ESCALATED_TO_HUMAN, no 5xx, no dropped ticket data")


def main() -> None:
    logger.info("Starting chaos drills against app.main:app (in-process, real Postgres + Redis)")

    # `with TestClient(app) as client:` drives the real FastAPI lifespan
    # (app/main.py's `lifespan`) -- init_models(), OpenTelemetry setup, and
    # the Redis checkpointer all come up exactly as they would under
    # uvicorn, so these drills exercise the actual production wiring
    # rather than a stub of it.
    with TestClient(app) as client:
        failures: list[tuple[str, Exception]] = []
        for name, drill in (
            ("Drill A", drill_a_database_outage),
            ("Drill B", drill_b_rate_limit),
            ("Drill C", drill_c_corrupted_response),
        ):
            try:
                drill(client)
            except AssertionError as exc:
                failures.append((name, exc))
                logger.error("%s FAILED: %s", name, exc)

    logger.info("=" * 78)
    if failures:
        logger.error("%d/3 drills FAILED: %s", len(failures), [n for n, _ in failures])
        sys.exit(1)
    logger.info("All 3 chaos drills PASSED -- the engine degrades gracefully under every simulated failure mode.")


if __name__ == "__main__":
    main()
