"""
OpenTelemetry instrumentation for the triage engine.

Three signal sources are wired together here, all feeding one process-wide
`TracerProvider` so a single trace ID threads across every layer of a
request:

  1. **FastAPI routes** -- `FastAPIInstrumentor` auto-instruments every
     request; the root span for `POST /api/v1/triage` is what everything
     below nests under.
  2. **SQLAlchemy DB sessions** -- `SQLAlchemyInstrumentor` wraps the async
     engine so every SQL statement (pgvector cosine search, the ticket
     insert, `init_models()`'s DDL) shows up as a child span with its own
     duration, without a single line of tracing code inside app/database.py
     or app/services/rag.py.
  3. **LangGraph node executions** -- there is no off-the-shelf LangGraph
     instrumentor, so `traced_node()` below is a hand-rolled decorator that
     wraps each node coroutine in its own span, recording wall-clock
     duration and whichever domain-specific attributes that node produced
     (LLM-used flag, confidence, RAG hit count). app/services/llm.py uses
     the same tracer to open a nested `llm.call` span around each Ollama/
     Anthropic request and records the provider's own reported token
     counts on it, so a slow ticket can be diagnosed by opening its trace
     and reading off exactly which span was expensive: a slow pgvector
     query looks like a long `db.query` span, a slow/expensive LLM call
     looks like a long `llm.call` span with a large `llm.tokens.total`,
     and the two are never confused because they are siblings, not one
     lumped "triage took 8s" number.

Span context propagation: `start_as_current_span` (used throughout this
module and its callers) sets the new span as the *current* span on an
internal `contextvars.ContextVar`, which `asyncio` automatically copies
into every task/coroutine spawned from that point onward. That is what
lets `llm.call` (opened deep inside app/services/llm.py) automatically
attach itself as a *child* of whichever `graph.node.*` span called it,
and why that in turn nests under the FastAPI-instrumented request span --
none of these layers pass a trace/span ID around manually; it all rides
the ambient async context.
"""
import logging
import os
from functools import wraps
from time import perf_counter
from typing import Any, Awaitable, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from app.config import get_settings

logger = logging.getLogger("triage_engine.telemetry")

_TRACER_NAME = "triage_engine"
_tracer_provider: TracerProvider | None = None


def instrument_fastapi_app(app) -> None:
    """
    Wrap `app`'s ASGI middleware stack with OTel's request-span
    instrumentation.

    This is deliberately split out from `setup_telemetry()` below and must
    be called immediately after `FastAPI(...)` is constructed -- e.g. right
    at the bottom of app/main.py's module body -- rather than from the
    lifespan startup hook. The reason is a Starlette implementation detail:
    `Starlette.__call__` builds and *caches* `self.middleware_stack` on the
    very first ASGI call it receives, and that first call is the
    "lifespan" scope itself (TestClient/uvicorn send it before any HTTP
    request). `FastAPIInstrumentor.instrument_app()` works by monkey-
    patching `app.build_middleware_stack`, so if it is called from inside
    the lifespan startup function -- i.e. *during* that first ASGI call --
    the stack has already been built from the unpatched method and the
    patch has no effect: every request after that silently gets zero
    tracing, with no error raised anywhere to say so.
    #
    # It is safe to call this before `setup_telemetry()` installs the real
    # TracerProvider: `get_tracer()` calls made at instrument-time bind to
    # the OTel API's proxy tracer provider, which transparently starts
    # forwarding to the real provider the moment `trace.set_tracer_provider()`
    # runs -- this is a documented, intentional part of the API surface for
    # exactly this "instrument early, configure exporters later" ordering.
    """
    FastAPIInstrumentor.instrument_app(app)


def setup_telemetry() -> None:
    """
    Build and install the process-wide TracerProvider, then instrument
    SQLAlchemy against it.

    Called once from app/main.py's lifespan startup (after `init_models()`
    has built the engine, so `SQLAlchemyInstrumentor` has a real engine
    instance to attach to). FastAPI instrumentation is intentionally NOT
    done here -- see `instrument_fastapi_app()`'s docstring for why it
    must run before the app's first ASGI call instead. Idempotent-by-
    construction is not attempted here deliberately -- calling this twice
    in one process would double-instrument SQLAlchemy and double-export
    spans, so it must only ever be called once, which the single lifespan
    call site guarantees.
    """
    global _tracer_provider
    settings = get_settings()

    resource = Resource.create(
        {
            "service.name": "triage-engine",
            "service.version": "0.3.0",
            "deployment.environment": settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)

    # OTEL_EXPORTER_OTLP_ENDPOINT is the standard OTel env var for pointing
    # at a real collector (Jaeger, Tempo, an APM vendor, ...). Left unset,
    # this falls back to a console exporter -- every span prints as JSON to
    # stdout -- so the service is fully traceable out of the box with zero
    # extra infrastructure to stand up for local dev/CI.
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        logger.info("OpenTelemetry: exporting spans via OTLP/HTTP to %s", otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()
        logger.info(
            "OpenTelemetry: OTEL_EXPORTER_OTLP_ENDPOINT not set -- exporting spans to "
            "stdout (console exporter). Set it to point at a real collector in prod."
        )

    # BatchSpanProcessor buffers spans and exports them off the request
    # path in a background thread -- a request never blocks on export I/O,
    # which matters most for the OTLP exporter (a network call) but is
    # good hygiene even for the console exporter.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    # Instrument the same engine app/database.py's session factory is
    # bound to, so every statement issued through a request-scoped
    # AsyncSession shows up as a child "db.query"-style span with its own
    # duration -- this is what makes "slow pgvector query vs. slow LLM
    # call" a one-glance answer instead of a guess. Unlike FastAPI's
    # instrumentation, this has no first-call caching trap -- SQLAlchemy
    # event listeners can be attached any time before the first query --
    # so it's fine to do this here, after init_models() has already run
    # its own DDL queries un-instrumented.
    from app.database import get_engine

    SQLAlchemyInstrumentor().instrument(engine=get_engine().sync_engine)

    logger.info("OpenTelemetry instrumentation active (FastAPI + SQLAlchemy)")


def shutdown_telemetry() -> None:
    """Flush and shut down the span processor. Called from lifespan teardown."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()


def get_tracer():
    """Process-wide tracer used by every span opened outside FastAPI/SQLAlchemy's own instrumentation."""
    return trace.get_tracer(_TRACER_NAME)


def current_trace_id() -> str | None:
    """
    Return the current span's trace ID as a 32-hex-char string, or None if
    no span is active (e.g. tracing never got set up, or this runs outside
    a request).

    This is what lets app/main.py's global exception handlers stamp every
    RFC 7807 problem-details response with a `trace_id` field: an on-call
    engineer can paste that ID straight into the tracing backend and land
    on the exact request that produced the 500, instead of grepping logs
    by timestamp.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def record_llm_usage(span: Span, *, provider: str, model: str, prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """
    Attach token-count attributes to the given span using OTel's
    semantic-convention-style `llm.*` attribute names.

    Called from app/services/llm.py's backend implementations, which are
    the only place the raw provider response (carrying token usage) is
    available -- `call_structured`'s caller (a graph node) only ever sees
    the validated Pydantic object, never the provider's raw JSON, so usage
    has to be recorded here or it's lost.
    """
    span.set_attribute("llm.provider", provider)
    span.set_attribute("llm.model", model)
    if prompt_tokens is not None:
        span.set_attribute("llm.tokens.prompt", prompt_tokens)
    if completion_tokens is not None:
        span.set_attribute("llm.tokens.completion", completion_tokens)
    if prompt_tokens is not None and completion_tokens is not None:
        span.set_attribute("llm.tokens.total", prompt_tokens + completion_tokens)


F = TypeVar("F", bound=Callable[..., Awaitable[dict]])


def traced_node(node_name: str) -> Callable[[F], F]:
    """
    Decorator that wraps a LangGraph node coroutine in its own span.

    LangGraph itself has no OTel integration, so every node in
    app/graph.py is wrapped with this instead of relying on
    auto-instrumentation: it records the node's wall-clock duration (the
    thing that actually answers "which of the three nodes was slow for
    this ticket?") plus whatever domain-specific signal that node's return
    dict carries -- LLM-used flag, self-reported confidence, RAG hit
    count -- as span attributes, and marks the span as errored (with the
    exception attached) if the node raises instead of returning.

    Because `func` is awaited *inside* `start_as_current_span`'s context
    manager, this span is active for the node's entire async body,
    including any further spans it opens (e.g. `llm.call` inside
    app/services/llm.py) -- those attach as children automatically via the
    same ambient-context propagation described in the module docstring,
    with no span/trace ID threaded through function arguments by hand.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict:
            tracer = get_tracer()
            start = perf_counter()
            with tracer.start_as_current_span(f"graph.node.{node_name}") as span:
                span.set_attribute("graph.node.name", node_name)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    # record_exception + an explicit ERROR status is what
                    # makes this span (and every ancestor span, since most
                    # backends roll error status up the trace) visibly red
                    # in a tracing UI -- a bare `raise` alone leaves the
                    # span looking like it completed normally.
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
                duration_ms = (perf_counter() - start) * 1000
                span.set_attribute("graph.node.duration_ms", round(duration_ms, 2))
                if isinstance(result, dict):
                    _annotate_node_result(span, result)
                span.set_status(Status(StatusCode.OK))
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _annotate_node_result(span: Span, result: dict) -> None:
    """Best-effort extraction of interesting fields a node returned, for span attributes."""
    if "used_llm_log_inspector" in result:
        span.set_attribute("llm.used", bool(result["used_llm_log_inspector"]))
        span.set_attribute("triage.confidence", float(result.get("log_inspector_confidence", 0.0)))
    if "used_llm_triage_router" in result:
        span.set_attribute("llm.used", bool(result["used_llm_triage_router"]))
        span.set_attribute("triage.confidence", float(result.get("triage_confidence", 0.0)))
    if "retrieved_context" in result:
        span.set_attribute("rag.hits", len(result["retrieved_context"]))
    if "status" in result:
        span.set_attribute("triage.status", str(result["status"]))
    if "escalation_reason" in result and result["escalation_reason"]:
        span.set_attribute("triage.escalation_reason", str(result["escalation_reason"]))
