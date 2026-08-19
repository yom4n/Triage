"""
LangGraph multi-agent triage pipeline.

    START -> log_inspector_node -> rag_lookup_node -> triage_router_node -> END

Every node communicates exclusively through the shared `TicketState` dict --
no node calls another directly, and none holds private state. That's what
makes the pipeline resilient: if the process restarts mid-run, the
Redis-backed checkpointer (wired up in app/main.py) can resume execution
from the last completed node using only what's already sitting in state.

Phase 2 additions over Phase 1:
  * log_inspector_node and triage_router_node now call a live LLM
    (local Ollama by default, or Claude -- see app/services/llm.py) under
    a strict JSON-schema contract, instead of pure regex/keyword rules.
  * rag_lookup_node sits between them: it embeds the cleaned error
    signature and retrieves verified historical resolutions via pgvector
    cosine search (app/services/rag.py), so triage_router_node reasons
    over real prior outcomes instead of inventing a fix from scratch.
  * Both LLM-backed nodes fall back to the original Phase 1 deterministic
    logic (kept below, renamed with a `_fallback_` prefix) if the LLM
    backend is unreachable or returns output that fails validation after
    retries -- a production triage endpoint must degrade gracefully, not
    502 because a local model server is down.
"""
import logging
import re
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas import EnvironmentEnum, SeverityEnum
from app.services.embeddings import EmbeddingError, generate_embedding
from app.services.llm import LLMError, call_structured
from app.services.rag import find_similar_tickets

logger = logging.getLogger("triage_engine.graph")


class RetrievedTicket(TypedDict):
    """One RAG hit, as threaded through TicketState (see rag_lookup_node)."""

    ticket_id: str
    extracted_error: str
    resolution: str
    similarity: float


class TicketState(TypedDict):
    """
    Shared state threaded through every node in the graph.

    A TypedDict (rather than a class with methods) is used deliberately:
    LangGraph nodes are plain functions that take the current state dict
    and return a *partial* dict of updates, which the graph runtime merges
    back into state. There is no mutable object passed by reference
    between nodes -- each node only ever sees what earlier nodes wrote.
    """

    ticket_id: str
    title: str
    stack_trace: str
    environment: str
    description: str

    # -- Populated by log_inspector_node -------------------------------
    extracted_error: str
    exception_message: str
    affected_file: str | None
    affected_line: int | None
    embedding_query: str
    log_inspector_confidence: float
    used_llm_log_inspector: bool

    # -- Populated by rag_lookup_node -----------------------------------
    retrieved_context: list[RetrievedTicket]

    # -- Populated by triage_router_node ---------------------------------
    severity: str
    summary: str
    resolution_steps: list[str]
    triage_confidence: float
    used_llm_triage_router: bool


# ===========================================================================
# Node 1: log_inspector_node
# ===========================================================================


class LogInspectorOutput(BaseModel):
    """
    Structured extraction target for log_inspector_node.

    Field types are deliberately flat (no Enum, no nested model): Ollama's
    structured-output grammar wants a self-contained schema with no
    `$defs`, and `severity`/`root_exception` classification lives entirely
    in `enum`/plain-string constraints rather than a Python Enum class.
    See `_pydantic_to_ollama_schema` in app/services/llm.py.
    """

    model_config = ConfigDict(extra="forbid")

    root_exception: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "The specific exception/error class or HTTP status that is the "
            "root cause, e.g. 'NullPointerException' or '504 Gateway Timeout'."
        ),
    )
    exception_message: str = Field(
        default="",
        max_length=500,
        description="The human-readable message that accompanied the exception, if any.",
    )
    affected_file: str | None = Field(
        default=None,
        max_length=500,
        description="The file path most directly implicated in the failure, if the trace names one.",
    )
    affected_line: int | None = Field(
        default=None,
        ge=0,
        description="The line number in affected_file most directly implicated, if the trace names one.",
    )
    embedding_query: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "A short, clean natural-language sentence describing this failure, "
            "suitable for semantic search against past tickets. Strip file "
            "paths, line numbers, and timestamps."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your confidence that root_exception correctly identifies the true root cause, 0.0-1.0.",
    )


_LOG_INSPECTOR_SYSTEM_PROMPT = """You are a log analysis assistant for a Tier-1 technical support triage system.

Given a raw stack trace or log excerpt, identify:
1. The specific root-cause exception or error signature.
2. The file and line most directly implicated, ONLY if the trace actually names one -- never invent a file/line that isn't present.
3. A short, clean natural-language sentence describing the failure, suitable for semantic search against a database of past tickets (no file paths, line numbers, or timestamps).

Be conservative with confidence: only report confidence above 0.85 when the root cause is unambiguous from the text given."""


# Kept from Phase 1 as the offline fallback path -- see log_inspector_node.
# Checked in order from most to least specific, so a fully-qualified
# exception (java.lang.NullPointerException) is preferred over a bare
# word match found later in the same trace.
_ERROR_PATTERNS = [
    re.compile(r"\b(?:[a-zA-Z_][\w]*\.)+([A-Z][\w]*(?:Exception|Error))\b"),
    re.compile(r"\b([A-Z][\w]*(?:Error|Exception|Timeout|Warning))\s*:"),
    re.compile(r"\b([A-Z][\w]*(?:Error|Exception))\b"),
    re.compile(r"\b(\d{3}\s+[A-Z][\w ]+)\b"),
]


def _fallback_extract_error(trace: str) -> str:
    """Phase 1's deterministic regex extractor, used when the LLM path fails."""
    for pattern in _ERROR_PATTERNS:
        match = pattern.search(trace)
        if match:
            return match.group(1).strip()
    first_line = next((line.strip() for line in trace.splitlines() if line.strip()), "")
    return first_line[:200] or "UnknownError"


async def log_inspector_node(state: TicketState) -> dict:
    """
    Node 1: use LLM reasoning to isolate the root exception, the file/line
    it originated from, and a clean natural-language query string for the
    RAG retrieval step that follows.

    On LLM failure (backend unreachable, or output that fails schema
    validation after retries -- see app/services/llm.py's `call_structured`),
    falls back to `_fallback_extract_error`, the Phase 1 regex extractor.
    A novel or oddly-formatted trace should degrade to "best guess from
    patterns", never crash the request.
    """
    settings = get_settings()
    trace = state["stack_trace"][: settings.llm_max_trace_chars]
    user_prompt = (
        f"Title: {state['title']}\n"
        f"Environment: {state['environment']}\n"
        f"Stack trace / log excerpt:\n{trace}"
    )

    try:
        result = await call_structured(
            system=_LOG_INSPECTOR_SYSTEM_PROMPT,
            user=user_prompt,
            schema_model=LogInspectorOutput,
            node_name="log_inspector_node",
        )
        return {
            "extracted_error": result.root_exception,
            "exception_message": result.exception_message,
            "affected_file": result.affected_file,
            "affected_line": result.affected_line,
            "embedding_query": result.embedding_query,
            "log_inspector_confidence": result.confidence,
            "used_llm_log_inspector": True,
        }
    except LLMError as exc:
        logger.warning(
            "log_inspector_node: LLM call failed (%s); falling back to regex extraction", exc
        )
        extracted_error = _fallback_extract_error(trace)
        return {
            "extracted_error": extracted_error,
            "exception_message": "",
            "affected_file": None,
            "affected_line": None,
            "embedding_query": f"{extracted_error} {state['title']}",
            "log_inspector_confidence": 0.0,
            "used_llm_log_inspector": False,
        }


# ===========================================================================
# Node 2: rag_lookup_node
# ===========================================================================


async def rag_lookup_node(state: TicketState, config: RunnableConfig) -> dict:
    """
    Node 2: retrieve verified historical tickets whose root cause resembles
    this one, so triage_router_node can ground its recommendation instead
    of inventing a fix from scratch.

    The DB session is threaded in via LangGraph's `config["configurable"]`
    rather than a module-level global or a second connection opened here:
    `build_triage_graph()` stays a pure function of `TicketState`, and
    app/main.py passes the *same* request-scoped AsyncSession it already
    holds via `Depends(get_db)` -- so retrieval reads happen in the same
    transaction as the eventual ticket insert, and this node never opens
    its own pooled connection. LangGraph passes `config` automatically to
    any node function with a second parameter named `config` -- but only
    if it is annotated `RunnableConfig` (bare `dict` is not recognized by
    the runtime's introspection and silently drops the argument, which is
    why the type hint below is not optional).
    """
    settings = get_settings()
    db: AsyncSession = config["configurable"]["db_session"]

    query_text = state.get("embedding_query") or state["extracted_error"]

    try:
        query_vector = await generate_embedding(query_text)
    except EmbeddingError as exc:
        # A dead embedding backend degrades the pipeline -- triage still
        # proceeds, just ungrounded -- rather than crashing the request.
        logger.warning(
            "rag_lookup_node: embedding failed (%s); proceeding with no retrieved context", exc
        )
        return {"retrieved_context": []}

    hits = await find_similar_tickets(db, query_vector, limit=settings.rag_top_k)

    retrieved_context: list[RetrievedTicket] = [
        {
            "ticket_id": hit.ticket_id,
            "extracted_error": hit.extracted_error,
            "resolution": hit.resolution,
            "similarity": round(hit.similarity, 4),
        }
        for hit in hits
    ]
    return {"retrieved_context": retrieved_context}


# ===========================================================================
# Node 3: triage_router_node
# ===========================================================================


class TriageOutput(BaseModel):
    """Structured decision target for triage_router_node. See LogInspectorOutput for why fields are flat."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        ..., description="Final Tier-1 severity classification."
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="A one-to-two sentence diagnostic summary of the failure and its impact.",
    )
    resolution_steps: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "Ordered, concrete remediation steps. If similar historical "
            "tickets were provided, ground these in what actually worked "
            "before rather than inventing a new approach."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your genuine confidence in this severity/resolution assessment, 0.0-1.0.",
    )


_TRIAGE_ROUTER_SYSTEM_PROMPT = """You are a Tier-1 triage router for a software support desk.

Given a ticket's environment, its root-cause error, the full stack trace, and (if available) similar historical tickets with verified resolutions, decide:
1. severity: CRITICAL, HIGH, MEDIUM, or LOW.
2. summary: a one-to-two sentence diagnostic explanation of the failure and its production impact.
3. resolution_steps: concrete, ordered remediation steps.

Severity guidance:
- Production outages involving data loss, database timeouts, deadlocks, connection exhaustion, or crashes are CRITICAL.
- Other production-environment errors are HIGH.
- Staging-environment errors are MEDIUM if they resemble a past critical/high issue, otherwise LOW.
- Development-environment errors are LOW.

If historical tickets are provided and genuinely relevant (check the similarity score and whether the error actually matches), ground resolution_steps in what verifiably worked before. If none are relevant, reason from the stack trace itself -- do not fabricate a historical precedent that wasn't given to you.

Report your genuine confidence -- do not default to a high number."""


# Kept from Phase 1 as the offline fallback path -- see triage_router_node.
_CRITICAL_KEYWORDS = {
    "timeout", "deadlock", "outofmemory", "out of memory", "connection refused",
    "connectionerror", "databaseerror", "segmentationfault", "segfault",
    "corruption", "data loss", "dataloss", "disk full", "too many connections",
}
_HIGH_KEYWORDS = {
    "nullpointerexception", "typeerror", "keyerror", "indexerror",
    "attributeerror", "referenceerror", "500 internal server error",
    "unauthorized", "permissiondenied", "stackoverflow",
}


def _matches_any(haystack: str, keywords: set[str]) -> bool:
    return any(keyword in haystack for keyword in keywords)


def _fallback_severity(*, extracted_error: str, stack_trace: str, environment: str) -> SeverityEnum:
    """Phase 1's deterministic keyword rules, used when the LLM path fails."""
    signal = f"{extracted_error} {stack_trace}".lower()

    if environment == EnvironmentEnum.PRODUCTION.value:
        return SeverityEnum.CRITICAL if _matches_any(signal, _CRITICAL_KEYWORDS) else SeverityEnum.HIGH
    if environment == EnvironmentEnum.STAGING.value:
        if _matches_any(signal, _CRITICAL_KEYWORDS):
            return SeverityEnum.HIGH
        if _matches_any(signal, _HIGH_KEYWORDS):
            return SeverityEnum.MEDIUM
        return SeverityEnum.LOW
    return SeverityEnum.LOW  # development


def _format_retrieved_context(retrieved_context: list[RetrievedTicket]) -> str:
    if not retrieved_context:
        return "No similar historical tickets were found."
    lines = ["Similar historical tickets (verified resolutions), most similar first:"]
    for i, hit in enumerate(retrieved_context, start=1):
        lines.append(
            f"{i}. [similarity={hit['similarity']:.2f}] error={hit['extracted_error']!r} "
            f"-> resolution: {hit['resolution']}"
        )
    return "\n".join(lines)


async def triage_router_node(state: TicketState) -> dict:
    """
    Node 3: use LLM reasoning -- grounded in the historical tickets
    rag_lookup_node retrieved -- to assign final severity, a diagnostic
    summary, and concrete resolution steps.

    On LLM failure, falls back to `_fallback_severity`, the Phase 1
    keyword-rule classifier, same failure-handling shape as
    log_inspector_node. The fallback path deliberately does not attempt to
    fabricate resolution_steps it has no grounding for -- it returns a
    single generic escalation step instead of guessing.
    """
    settings = get_settings()
    trace = state["stack_trace"][: settings.llm_max_trace_chars]
    context_block = _format_retrieved_context(state.get("retrieved_context", []))
    user_prompt = (
        f"Title: {state['title']}\n"
        f"Environment: {state['environment']}\n"
        f"Extracted root error: {state['extracted_error']}\n"
        f"Stack trace / log excerpt:\n{trace}\n\n"
        f"{context_block}"
    )

    try:
        result = await call_structured(
            system=_TRIAGE_ROUTER_SYSTEM_PROMPT,
            user=user_prompt,
            schema_model=TriageOutput,
            node_name="triage_router_node",
        )
        summary = f"[{result.severity}] {result.summary}"
        return {
            "severity": result.severity,
            "summary": summary,
            "resolution_steps": result.resolution_steps,
            "triage_confidence": result.confidence,
            "used_llm_triage_router": True,
        }
    except LLMError as exc:
        logger.warning(
            "triage_router_node: LLM call failed (%s); falling back to keyword rules", exc
        )
        severity = _fallback_severity(
            extracted_error=state["extracted_error"],
            stack_trace=state["stack_trace"],
            environment=state["environment"],
        )
        summary = (
            f"[{severity.value}] {state['extracted_error']} detected in "
            f"{state['environment']} environment (fallback rule-based triage -- "
            f"LLM reasoning was unavailable). Ticket: '{state['title']}'."
        )
        return {
            "severity": severity.value,
            "summary": summary,
            "resolution_steps": [
                "Escalate to an on-call engineer for manual investigation; "
                "automated LLM triage was unavailable and this ticket was "
                "classified by rule-based fallback only."
            ],
            "triage_confidence": 0.0,
            "used_llm_triage_router": False,
        }


# ===========================================================================
# Graph assembly
# ===========================================================================


def build_triage_graph(checkpointer=None):
    """
    Assemble and compile the StateGraph:

        START -> log_inspector_node -> rag_lookup_node -> triage_router_node -> END

    The linear edges (rather than conditional routing) guarantee LangGraph
    never runs a node until every edge feeding it has fired: rag_lookup_node
    never embeds a raw, un-sanitized stack trace (it only sees the cleaned
    `embedding_query`/`extracted_error` log_inspector_node already wrote),
    and triage_router_node never reasons without first having a chance at
    retrieved historical context.

    `checkpointer` is injected rather than constructed here because its
    backing Redis connection must be opened inside an async context
    manager (see app/main.py's lifespan) -- keeping this function
    synchronous and side-effect-free makes it safe to call at import time
    or from tests without a Redis instance running.
    """
    workflow = StateGraph(TicketState)

    workflow.add_node("log_inspector", log_inspector_node)
    workflow.add_node("rag_lookup", rag_lookup_node)
    workflow.add_node("triage_router", triage_router_node)

    workflow.add_edge(START, "log_inspector")
    workflow.add_edge("log_inspector", "rag_lookup")
    workflow.add_edge("rag_lookup", "triage_router")
    workflow.add_edge("triage_router", END)

    return workflow.compile(checkpointer=checkpointer)
