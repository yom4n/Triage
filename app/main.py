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

from fastapi import Depends, FastAPI, HTTPException, Request, status
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import Ticket, dispose_engine, get_db, init_models
from app.graph import TicketState, build_triage_graph
from app.schemas import SeverityEnum, TicketCreate, TicketResponse
from app.services.embeddings import EmbeddingError, build_embedding_text, generate_embedding, warn_if_deterministic

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("triage_engine")


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
        # Postgres connection pool.
        await redis_checkpointer_cm.__aexit__(None, None, None)
        await dispose_engine()


app = FastAPI(
    title="Autonomous Tier-1 Technical Support & Triage Engine",
    description="Phase 2: LLM reasoning + pgvector RAG over the LangGraph triage pipeline.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness probe. Deliberately avoids touching Postgres/Redis/Ollama so it stays cheap."""
    return {"status": "ok"}


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
    """
    End-to-end triage flow:

    1. FastAPI + Pydantic validate/sanitize `payload` against `TicketCreate`
       (app/schemas.py) before this function body even runs.
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
        # completes.
        result_state: TicketState = await graph.ainvoke(initial_state, config=run_config)
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

    return TicketResponse(
        ticket_id=ticket_record.id,
        title=ticket_record.title,
        environment=payload.environment,
        extracted_error=ticket_record.extracted_error,
        affected_file=ticket_record.affected_file,
        affected_line=ticket_record.affected_line,
        severity=SeverityEnum(ticket_record.severity),
        summary=ticket_record.summary,
        resolution_steps=ticket_record.resolution_steps,
        confidence=ticket_record.confidence or 0.0,
        similar_tickets_considered=len(result_state.get("retrieved_context", [])),
        created_at=ticket_record.created_at,
    )
