"""
FastAPI application entrypoint.

Wires config, the database layer, and the LangGraph RAG pipeline together
behind a single webhook-style endpoint: callers submit a raw ticket and
get back a structured, LLM-reasoned, historically-grounded triage result.
The client system never needs to know an LLM/agent pipeline sits behind
the endpoint -- it just POSTs JSON and receives JSON.

Run with:  uvicorn app.main:app --reload
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import Ticket, dispose_engine, get_db, init_models
from app.graph import TicketState, build_triage_graph
from app.schemas import (
    CrashReport,
    EnvironmentEnum,
    SeverityEnum,
    TicketCreate,
    TicketResponse,
    TriageStatusEnum,
    crash_report_to_ticket_create,
)
from app.services.embeddings import EmbeddingError, build_embedding_text, generate_embedding, warn_if_deterministic
from app.services.llm import LLMError, LLMMalformedOutputError, LLMTimeoutError
from app.services.telemetry import current_trace_id, instrument_fastapi_app, setup_telemetry, shutdown_telemetry

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("triage_engine")


# ===========================================================================
# RFC 7807 (application/problem+json) error contract
# ===========================================================================
#
# Every handler below returns the same shape so a client system only has to
# write one deserializer for every error this API can produce, and an
# on-call engineer can always find `trace_id` in the same place regardless
# of which exception fired. `type` is a stable, greppable slug (not a real
# dereferenceable URL -- RFC 7807 only requires it be a URI *identifier*,
# not that it resolves) an engineer can search runbooks/dashboards for.

_PROBLEM_BASE_TYPE = "https://triage-engine.internal/errors"


def _problem_response(*, status_code: int, title: str, detail: str, type_slug: str, request: Request) -> JSONResponse:
    """
    Build one RFC 7807 problem-details response, stamped with the current
    OpenTelemetry trace ID.

    `trace_id` is the whole point of pairing structured error logging with
    tracing: this same ID is on the span FastAPIInstrumentor opened for
    this request (see app/services/telemetry.py), so an on-call engineer
    can copy it straight from a client-reported 500 into the tracing
    backend and land on the exact request -- no timestamp-based log
    grepping required.
    """
    trace_id = current_trace_id()
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content={
            "type": f"{_PROBLEM_BASE_TYPE}/{type_slug}",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": str(request.url),
            "trace_id": trace_id or "unavailable",
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook -- the one place real network connections are
    allowed to open.

    Postgres tables/index and the Redis-backed LangGraph checkpointer are
    both brought up here rather than at module-import time. Keeping
    imports connection-free means unit tests can `import app.main` without
    a database, Redis, or Ollama instance available, and only pay the
    connection cost when the server actually starts serving traffic.
    """
    settings = get_settings()

    # 1. Ensure the pgvector extension, tables, and HNSW index exist. (Dev
    #    convenience -- a real deployment runs versioned Alembic
    #    migrations in CI/CD instead of calling this on every boot.)
    await init_models()
    logger.info(
        "Postgres schema ready at %s:%s/%s", settings.postgres_host, settings.postgres_port, settings.postgres_db
    )

    # 1b. Build the TracerProvider/exporter and instrument the SQLAlchemy
    #     engine `init_models()` just built. FastAPI's own instrumentation
    #     is NOT done here -- `instrument_fastapi_app(app)` already ran at
    #     module import time, right after `app = FastAPI(...)` below, for
    #     reasons documented on that function (a Starlette middleware-
    #     stack caching trap that silently no-ops tracing if instrumented
    #     from inside this startup hook instead).
    setup_telemetry()

    # 2. Loudly flag a non-semantic embedding configuration -- silent
    #    misconfiguration here would make RAG retrieval look "broken" for
    #    reasons that are hard to trace back to a config value.
    warn_if_deterministic()
    logger.info(
        "LLM provider=%s (model=%s) | embedding provider=%s (model=%s, dim=%d)",
        settings.llm_provider,
        settings.ollama_chat_model if settings.llm_provider == "ollama" else settings.anthropic_model,
        settings.embedding_provider,
        settings.ollama_embedding_model,
        settings.embedding_dim,
    )

    # 3. Open the Redis connection backing LangGraph's checkpointer.
    #    AsyncRedisSaver.from_conn_string(...) returns an async context
    #    manager because the underlying redis client needs a running event
    #    loop to connect; asetup() creates the indices RedisSaver uses to
    #    store/query checkpoint state. We enter the context manager
    #    manually (rather than via `async with`) so the connection stays
    #    open for the app's entire lifetime and is torn down in `finally`.
    redis_checkpointer_cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
    checkpointer = await redis_checkpointer_cm.__aenter__()
    await checkpointer.asetup()

    # Compile the graph once per process and hand it off via app.state so
    # every request reuses the same compiled graph + checkpointer instead
    # of rebuilding either per-call.
    app.state.triage_graph = build_triage_graph(checkpointer=checkpointer)
    logger.info("Triage graph compiled with Redis checkpointer at %s", settings.redis_url)

    try:
        yield
    finally:
        # Reverse order of acquisition: close Redis first, then drain the
        # Postgres connection pool, then flush any spans still buffered in
        # the OTel BatchSpanProcessor so a graceful shutdown doesn't lose
        # the trace for whatever request was in flight when it started.
        await redis_checkpointer_cm.__aexit__(None, None, None)
        await dispose_engine()
        shutdown_telemetry()


app = FastAPI(
    title="Autonomous Tier-1 Technical Support & Triage Engine",
    description=(
        "Phase 3 (Day 4): LLM reasoning + pgvector RAG over the LangGraph "
        "triage pipeline, with OpenTelemetry tracing, a human-escalation "
        "fallback node, and RFC 7807 structured error responses."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# Must run immediately after construction, before this app ever handles its
# first ASGI call (including the "lifespan" call itself) -- see
# `instrument_fastapi_app`'s docstring in app/services/telemetry.py for why
# calling this from inside `lifespan()` above would silently produce zero
# request tracing.
instrument_fastapi_app(app)

# Phase 4: /api/v1/ingest/crash is called directly from the monitored app's
# browser runtime (see client-sdks/), which is a different origin than this
# API -- without this, the browser blocks the request before it ever leaves
# the client. `cors_allow_origin_list` defaults to "*" (see app/config.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allow_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe. Deliberately avoids touching Postgres/Redis/Ollama so it stays cheap."""
    return {"status": "ok"}


# ===========================================================================
# Global exception handlers (Day 4)
# ===========================================================================
#
# Registration order does not matter to Starlette's dispatcher -- it always
# picks the most specific matching handler for the exception's actual type
# (walking the MRO), so `SQLAlchemyError`/`LLMTimeoutError`/`LLMMalformedOutputError`
# below are each tried before the catch-all `Exception` handler regardless
# of where they're declared. Most of these exception types are already
# caught *inside* the LangGraph nodes that can raise them (see
# app/graph.py's per-node try/except LLMError blocks) as part of the
# graceful-degradation design -- these handlers exist for whatever gets
# past that: a DB failure during the final `db.commit()`/`db.refresh()`
# below (outside the graph's own try/except), a future endpoint that
# doesn't wrap its own DB/LLM calls, or a genuinely unanticipated bug.


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    Any Postgres-layer failure -- a dropped connection, a pool exhausted,
    a constraint violation -- surfaces here as a 503: the *engine* is fine,
    but this specific dependency is unavailable right now, which is what
    tells a well-behaved client "retry with backoff" rather than "give up".
    """
    trace_id = current_trace_id()
    logger.error("Database error [trace_id=%s]: %s", trace_id, exc, exc_info=True)
    return _problem_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        title="Database Unavailable",
        detail=(
            "The triage engine could not complete a database operation. "
            "This is very likely transient (a dropped connection or an "
            "exhausted pool) -- retry with exponential backoff."
        ),
        type_slug="database-error",
        request=request,
    )


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
    """
    The configured LLM backend didn't answer within `settings.llm_timeout_seconds`
    (or the backend was flat-out unreachable). Note: inside the triage
    pipeline itself this is caught by log_inspector_node/triage_router_node
    and degrades to the rule-based fallback path instead of ever reaching
    here (see app/graph.py) -- this handler is what fires if an LLM call
    is ever made *outside* that guarded path.
    """
    trace_id = current_trace_id()
    logger.error("LLM timeout [trace_id=%s]: %s", trace_id, exc, exc_info=True)
    return _problem_response(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        title="LLM Backend Timeout",
        detail=f"The upstream LLM provider did not respond in time: {exc}",
        type_slug="llm-timeout",
        request=request,
    )


@app.exception_handler(LLMMalformedOutputError)
async def llm_schema_validation_handler(request: Request, exc: LLMMalformedOutputError) -> JSONResponse:
    """
    The LLM backend responded, but its output didn't validate against the
    Pydantic schema the calling node required (app/services/llm.py's
    `call_structured` already retried `settings.llm_max_retries` times
    before giving up). Treated as a 502: the upstream dependency returned
    a response we cannot trust, not a fault of the caller's request.
    """
    trace_id = current_trace_id()
    logger.error("LLM schema validation failed [trace_id=%s]: %s", trace_id, exc, exc_info=True)
    return _problem_response(
        status_code=status.HTTP_502_BAD_GATEWAY,
        title="LLM Output Schema Validation Failed",
        detail=f"The upstream LLM provider returned output that failed schema validation: {exc}",
        type_slug="llm-schema-validation-error",
        request=request,
    )


@app.exception_handler(PydanticValidationError)
async def schema_validation_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """
    Catches Pydantic `ValidationError` raised *outside* FastAPI's own
    request-body validation (which already returns its own 422 via
    `RequestValidationError`/`fastapi.exception_handlers.request_validation_exception_handler`
    and is left untouched) -- e.g. a `TicketResponse` or an internal
    `BaseModel` failing to construct from data this service itself
    produced. That is always a server-side contract bug, so it is a 500,
    not a 422 -- the caller's request was fine.
    """
    trace_id = current_trace_id()
    logger.error("Internal schema validation error [trace_id=%s]: %s", trace_id, exc, exc_info=True)
    return _problem_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="Internal Schema Validation Error",
        detail=f"A response failed internal schema validation: {exc}",
        type_slug="schema-validation-error",
        request=request,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort catch-all: guarantees this API *never* returns a bare,
    unstructured 500 with a stack trace leaked into the response body.
    Every unanticipated failure still gets a well-formed RFC 7807 body and
    a trace_id an on-call engineer can search for -- the alternative is a
    generic ASGI error page that reveals nothing actionable.
    """
    trace_id = current_trace_id()
    logger.exception("Unhandled exception [trace_id=%s]", trace_id)
    return _problem_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        title="Internal Server Error",
        detail="An unexpected error occurred while processing this request.",
        type_slug="internal-error",
        request=request,
    )


def _ticket_to_response(ticket: Ticket, *, similar_tickets_considered: int = 0) -> TicketResponse:
    """
    Build a `TicketResponse` from a persisted `Ticket` row.

    Deliberately not `TicketResponse.model_validate(ticket)`: `from_attributes`
    only matches identically-named attributes, and the ORM's primary key
    column is `id`, not `ticket_id` -- a bare `model_validate` call silently
    raises "ticket_id: Field required" instead of picking it up. Shared by
    `_run_triage_pipeline` (one ticket, right after creating it) and
    `list_tickets` (many, read back from the DB) so this mapping exists in
    exactly one place instead of drifting between two hand-written copies.

    `similar_tickets_considered` has no column on `Ticket` (it's a fact
    about the triage *run*, not the stored record -- see `TicketState`'s
    `retrieved_context` in app/graph.py) so it isn't derivable from `ticket`
    alone; callers reading a ticket back from storage have no run to ask,
    so it defaults to 0 there.
    """
    return TicketResponse(
        ticket_id=ticket.id,
        title=ticket.title,
        environment=EnvironmentEnum(ticket.environment),
        stack_trace=ticket.stack_trace,
        extracted_error=ticket.extracted_error,
        affected_file=ticket.affected_file,
        affected_line=ticket.affected_line,
        severity=SeverityEnum(ticket.severity),
        summary=ticket.summary,
        resolution_steps=ticket.resolution_steps,
        confidence=ticket.confidence or 0.0,
        similar_tickets_considered=similar_tickets_considered,
        status=TriageStatusEnum(ticket.status),
        escalation_reason=ticket.escalation_reason,
        fix_attempted=ticket.fix_attempted,
        fix_skipped_reason=ticket.fix_skipped_reason,
        fix_diff=ticket.fix_diff,
        fix_pr_url=ticket.fix_pr_url,
        fix_branch_name=ticket.fix_branch_name,
        fix_verified=ticket.fix_verified,
        fix_verification_status=ticket.fix_verification_status,
        fix_verification_attempts=ticket.fix_verification_attempts,
        fix_test_command=ticket.fix_test_command,
        fix_test_output_tail=ticket.fix_test_output_tail,
        created_at=ticket.created_at,
    )


async def _run_triage_pipeline(
    payload: TicketCreate,
    request: Request,
    db: AsyncSession,
) -> TicketResponse:
    """
    End-to-end triage flow, shared by every ingestion route (a human filing
    a ticket via `POST /api/v1/triage`, or a monitored app's own crash
    reaching `POST /api/v1/ingest/crash`). Both routes do nothing but
    produce a valid `TicketCreate` and hand it here -- this function has no
    idea whether a person or a browser crash hook is the ultimate source,
    which is exactly the point: the pipeline stays one single code path
    regardless of how many ingestion adapters feed it.

    1. `payload` has already been validated/sanitized against `TicketCreate`
       (app/schemas.py) by the time it reaches here.
    2. The validated fields seed a `TicketState` and run through the
       compiled LangGraph pipeline (app/graph.py):
         log_inspector_node  -- LLM extracts the root exception + a clean
                                 semantic-search query string.
         rag_lookup_node     -- embeds that query and retrieves verified
                                 historical tickets via pgvector cosine search.
         triage_router_node  -- LLM assigns severity/summary/resolution
                                 steps, grounded in whatever was retrieved.
       This request's own `db` session is passed into the graph via
       `config["configurable"]["db_session"]` so retrieval reads share a
       transaction with the insert below rather than opening a second
       pooled connection.
    3. The enriched ticket is persisted to Postgres. It is also embedded
       and stored (unverified) so it can itself be retrieved by *future*
       tickets once a human reviews and verifies its resolution -- see the
       `is_verified` gate in app/services/rag.py.
    4. A typed `TicketResponse` is returned to the caller.
    """
    ticket_id = uuid.uuid4()

    initial_state: TicketState = {
        "ticket_id": str(ticket_id),
        "title": payload.title,
        "stack_trace": payload.stack_trace,
        "environment": payload.environment.value,
        "description": payload.description or "",
        "extracted_error": "",
        "exception_message": "",
        "affected_file": None,
        "affected_line": None,
        "embedding_query": "",
        "log_inspector_confidence": 0.0,
        "used_llm_log_inspector": False,
        "retrieved_context": [],
        "severity": "",
        "summary": "",
        "resolution_steps": [],
        "triage_confidence": 0.0,
        "used_llm_triage_router": False,
        "status": "COMPLETED",
        "escalation_reason": "",
        "fix_attempted": False,
        "fix_skipped_reason": "",
        "fix_diff": "",
        "fix_pr_url": "",
        "fix_branch_name": "",
        "fix_verified": False,
        "fix_verification_status": "NOT_ATTEMPTED",
        "fix_verification_attempts": 0,
        "fix_test_command": "",
        "fix_test_output_tail": "",
        "fix_candidate_content": "",
        "fix_explanation": "",
        "fix_llm_confidence": 0.0,
        "fix_affected_path": "",
        "fix_original_content": "",
        "fix_original_sha": "",
    }

    graph = request.app.state.triage_graph
    run_config = {
        "configurable": {
            # thread_id scopes checkpoint state per-ticket in Redis, so
            # concurrent triage runs for different tickets never read or
            # clobber each other's in-progress state.
            "thread_id": str(ticket_id),
            # rag_lookup_node reads this straight out of config -- see the
            # docstring on that node in app/graph.py for why the session
            # is threaded through here instead of opened inside the node.
            "db_session": db,
        }
    }

    try:
        # ainvoke is the async entrypoint into the compiled graph -- it
        # awaits each node in turn (log_inspector -> rag_lookup ->
        # triage_router), checkpointing state to Redis after every node
        # completes. LLM failures never reach here: log_inspector_node and
        # triage_router_node each catch LLMError internally and degrade to
        # their rule-based fallback (see app/graph.py), and a low- or
        # zero-confidence result is diverted to fallback_human_escalation_node
        # by the graph's own conditional routing rather than raising.
        result_state: TicketState = await graph.ainvoke(initial_state, config=run_config)
    except SQLAlchemyError:
        # A genuine DB-layer failure (e.g. rag_lookup_node's pgvector
        # query hitting a dropped connection) is NOT something a node can
        # gracefully fall back from -- there is no rule-based substitute
        # for "the database is unreachable". Re-raise so it's handled by
        # `database_error_handler` above instead of being flattened into
        # this function's generic 502, so callers get the correct 503 +
        # RFC 7807 body and this failure mode is distinguishable from an
        # LLM/pipeline-logic failure in logs and traces alike.
        raise
    except Exception:
        logger.exception("Triage pipeline failed for ticket %s", ticket_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Triage pipeline failed to process this ticket.",
        )

    ticket_record = Ticket(
        id=ticket_id,
        title=payload.title,
        stack_trace=payload.stack_trace,
        environment=payload.environment.value,
        description=payload.description,
        extracted_error=result_state["extracted_error"],
        exception_message=result_state.get("exception_message") or None,
        affected_file=result_state.get("affected_file"),
        affected_line=result_state.get("affected_line"),
        severity=result_state["severity"],
        summary=result_state["summary"],
        resolution_steps=result_state.get("resolution_steps") or [],
        confidence=result_state.get("triage_confidence"),
        # Newly-created tickets join the RAG corpus as *unverified* --
        # they only start grounding other tickets' recommendations once a
        # human confirms the resolution actually worked. See
        # app/services/rag.py's `require_verified` gate.
        is_verified=False,
        # Day 4: whatever fallback_human_escalation_node's conditional
        # routing decided (app/graph.py's `_route_after_triage`) is
        # persisted verbatim -- an escalated ticket is still a completed,
        # storable record, just one flagged for human follow-up rather
        # than treated as an automated resolution.
        status=result_state.get("status", "COMPLETED"),
        escalation_reason=result_state.get("escalation_reason") or None,
        # The fix sub-graph's outcome (app/graph.py) -- absent (all
        # falsy/None) for an escalated ticket, since that sub-graph is
        # never reached for one. See app/schemas.py's TicketResponse for
        # what each field means.
        fix_attempted=result_state.get("fix_attempted", False),
        fix_skipped_reason=result_state.get("fix_skipped_reason") or None,
        fix_diff=result_state.get("fix_diff") or None,
        fix_pr_url=result_state.get("fix_pr_url") or None,
        fix_branch_name=result_state.get("fix_branch_name") or None,
        # Phase 1: sandboxed verification outcome.
        fix_verified=result_state.get("fix_verified", False),
        fix_verification_status=result_state.get("fix_verification_status") or "NOT_ATTEMPTED",
        fix_verification_attempts=result_state.get("fix_verification_attempts", 0),
        fix_test_command=result_state.get("fix_test_command") or None,
        fix_test_output_tail=result_state.get("fix_test_output_tail") or None,
    )

    # Embed the ticket we just triaged so it becomes retrievable once
    # verified. This never fails the request: an embedding-backend outage
    # should not block ticket creation, it should just leave this row out
    # of the RAG corpus until a retry/backfill job re-embeds it.
    try:
        embedding_text = build_embedding_text(
            title=payload.title,
            extracted_error=result_state["extracted_error"],
            stack_trace=payload.stack_trace,
            description=payload.description or "",
        )
        ticket_record.embedding_text = embedding_text
        ticket_record.embedding = await generate_embedding(embedding_text)
    except EmbeddingError as exc:
        logger.warning("Failed to embed ticket %s for future RAG retrieval: %s", ticket_id, exc)

    db.add(ticket_record)
    await db.commit()
    await db.refresh(ticket_record)

    return _ticket_to_response(
        ticket_record, similar_tickets_considered=len(result_state.get("retrieved_context", []))
    )


@app.post(
    "/api/v1/triage",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["triage"],
    summary="Submit a bug report for automated, RAG-grounded Tier-1 triage",
)
async def triage_ticket(
    payload: TicketCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """Human/manual entry point -- see `_run_triage_pipeline` for the actual flow."""
    return await _run_triage_pipeline(payload, request, db)


@app.post(
    "/api/v1/ingest/crash",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["triage"],
    summary="Auto-ticket a crash reported by a monitored app's own client-side capture hook",
)
async def ingest_crash_report(
    report: CrashReport,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TicketResponse:
    """
    Phase 4 entry point: the automated counterpart to `triage_ticket` above.

    A monitored app (see `client-sdks/`) POSTs its own uncaught
    error/promise-rejection/React-boundary payload here, in whatever loose
    shape its runtime handed it (`CrashReport`, app/schemas.py). This route's
    only job is `crash_report_to_ticket_create()` -- normalizing that into a
    real `TicketCreate` -- before handing off to the exact same
    `_run_triage_pipeline` a human-filed ticket goes through. No separate
    pipeline, no separate persistence path: from this point on, an
    auto-detected crash and a manually-filed bug report are indistinguishable
    to the rest of the system.
    """
    payload = crash_report_to_ticket_create(report)
    return await _run_triage_pipeline(payload, request, db)


@app.get(
    "/api/v1/tickets",
    response_model=list[TicketResponse],
    tags=["triage"],
    summary="List triaged tickets, most recent first",
)
async def list_tickets(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[TicketResponse]:
    """
    Dashboard-backing endpoint (frontend/): every ticket this instance has
    triaged, newest first, paginated via `limit`/`offset`. Returns the same
    `TicketResponse` shape `_run_triage_pipeline` hands back on creation --
    a dashboard row and a just-submitted result are the same object, so the
    frontend needs only one renderer for both. Read-only: no filtering
    beyond pagination, since the frontend's own status/severity filters
    already operate over whatever page is loaded.
    """
    result = await db.execute(
        select(Ticket).order_by(Ticket.created_at.desc()).limit(limit).offset(offset)
    )
    tickets = result.scalars().all()
    return [_ticket_to_response(ticket) for ticket in tickets]
