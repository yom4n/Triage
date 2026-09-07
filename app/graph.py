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
import difflib
import logging
import re
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas import EnvironmentEnum, SeverityEnum
from app.services.embeddings import EmbeddingError, generate_embedding
from app.services.github import (
    GitHubError,
    GitHubFile,
    commit_file_update,
    create_branch,
    find_file_by_basename,
    get_branch_head_sha,
    get_file_content,
    open_pull_request,
)
from app.services.llm import LLMError, LLMMalformedOutputError, call_code_fix_text, call_structured
from app.services.rag import find_similar_tickets
from app.services.sandbox import SandboxError, verify_patch_in_sandbox
from app.services.telemetry import traced_node

logger = logging.getLogger("triage_engine.graph")


def _normalize_confidence(value: object) -> object:
    """
    `mode="before"` validator shared by every structured-output schema
    below that has a `confidence: float` field constrained to [0, 1].

    In practice (observed repeatedly with qwen2.5:7b-instruct, both here
    and in the Day 4 chaos-drill runs), a small local model occasionally
    reports confidence as a percentage-like integer -- e.g. `85` instead
    of `0.85` -- despite the field description explicitly asking for a
    0.0-1.0 fraction. Rejecting that outright just burns a retry (see
    app/services/llm.py's `call_structured`) and, having now seen it
    trigger an unnecessary human-escalation fallback multiple times in a
    row, it is worth normalizing rather than retrying: the model's
    *intent* ("85% confident") is unambiguous, so silently dividing by
    100 recovers a perfectly good answer instead of discarding it.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1:
        return value / 100
    return value


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

    # -- Populated by triage_router_node (default) or overwritten by
    #    fallback_human_escalation_node when routed there -------------------
    status: str  # "COMPLETED" | "ESCALATED_TO_HUMAN"
    escalation_reason: str

    # -- Populated by the fix sub-graph: generate_fix -> verify_fix ->
    #    open_pr | fix_escalation (only reached for confident, non-escalated
    #    tickets -- see build_triage_graph's routing) -----------------------
    fix_attempted: bool
    fix_skipped_reason: str
    fix_diff: str
    fix_pr_url: str
    fix_branch_name: str

    # Phase 1: sandboxed verification. `fix_verification_status` is one of
    # PASSED | FAILED_RETRY | FAILED_MAX_ATTEMPTS | SKIPPED_NO_SANDBOX |
    # SKIPPED_UNTESTABLE | NOT_ATTEMPTED -- see verify_fix_node.
    fix_verified: bool
    fix_verification_status: str
    fix_verification_attempts: int
    fix_test_command: str
    fix_test_output_tail: str
    # Working fields threaded between the sub-graph's nodes (not persisted
    # to the ticket row): the candidate file content awaiting a PR, the
    # resolved repo path, and the original file's content+sha for the commit.
    fix_candidate_content: str
    fix_explanation: str
    fix_llm_confidence: float
    fix_affected_path: str
    fix_original_content: str
    fix_original_sha: str


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

    _normalize_confidence = field_validator("confidence", mode="before")(_normalize_confidence)


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


# `at functionName (file:line:col)` (V8/Node/browser JS traces) or the
# bare `at file:line:col` form -- matches the exact shape crashReporter.ts
# forwards from `Error.stack`. Applied unconditionally as a backstop below,
# not just in the LLM-failure branch: in practice the model reliably finds
# the *exception*, but is inconsistent about also populating
# affected_file/affected_line even when the trace clearly names one --
# and propose_fix_node has nothing to work with at all without one, so
# this mechanical, well-defined extraction is worth doing regardless of
# whether the LLM path otherwise succeeded.
_FILE_LINE_PATTERN = re.compile(r"\(([^\s()]+):(\d+):\d+\)|(?:^|\s)at\s+([^\s()]+):(\d+):\d+")


def _fallback_extract_file_line(trace: str) -> tuple[str | None, int | None]:
    match = _FILE_LINE_PATTERN.search(trace)
    if not match:
        return None, None
    file_path = match.group(1) or match.group(3)
    line_str = match.group(2) or match.group(4)
    return file_path, int(line_str)


@traced_node("log_inspector")
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
        affected_file, affected_line = result.affected_file, result.affected_line
        if affected_file is None:
            # Backstop, not a fight with the model -- see
            # _fallback_extract_file_line's comment just above.
            affected_file, affected_line = _fallback_extract_file_line(trace)
        return {
            "extracted_error": result.root_exception,
            "exception_message": result.exception_message,
            "affected_file": affected_file,
            "affected_line": affected_line,
            "embedding_query": result.embedding_query,
            "log_inspector_confidence": result.confidence,
            "used_llm_log_inspector": True,
        }
    except LLMError as exc:
        logger.warning(
            "log_inspector_node: LLM call failed (%s); falling back to regex extraction", exc
        )
        extracted_error = _fallback_extract_error(trace)
        affected_file, affected_line = _fallback_extract_file_line(trace)
        return {
            "extracted_error": extracted_error,
            "exception_message": "",
            "affected_file": affected_file,
            "affected_line": affected_line,
            "embedding_query": f"{extracted_error} {state['title']}",
            "log_inspector_confidence": 0.0,
            "used_llm_log_inspector": False,
        }


# ===========================================================================
# Node 2: rag_lookup_node
# ===========================================================================


@traced_node("rag_lookup")
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

    _normalize_confidence = field_validator("confidence", mode="before")(_normalize_confidence)


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


@traced_node("triage_router")
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
            # Provisional -- `_route_after_triage` below overrides this to
            # ESCALATED_TO_HUMAN if `result.confidence` is still under
            # `settings.min_confidence` despite the LLM call succeeding.
            "status": "COMPLETED",
            "escalation_reason": "",
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
            # `used_llm_triage_router: False` alone is what
            # `_route_after_triage` keys off of to divert to
            # fallback_human_escalation_node -- status/escalation_reason
            # here are just provisional placeholders it overwrites.
            "status": "COMPLETED",
            "escalation_reason": "",
        }


# ===========================================================================
# Node 4: fallback_human_escalation_node
# ===========================================================================


@traced_node("fallback_human_escalation")
async def fallback_human_escalation_node(state: TicketState) -> dict:
    """
    Terminal safety-net node: reached only via the conditional edge out of
    triage_router_node (see `_route_after_triage`), never directly wired
    from START.

    This is the "never crash or hang on upstream API outages" guarantee
    made concrete: whatever partial diagnosis the pipeline managed to
    produce (extracted error, RAG hits, a rule-based severity guess) is
    preserved rather than discarded, but the ticket is explicitly marked
    `ESCALATED_TO_HUMAN` and its resolution_steps are replaced with a
    single honest instruction to get a human involved -- callers must
    never receive a low-confidence or rule-based guess dressed up to look
    like a fully-reasoned LLM triage result. No exception is raised and no
    ticket data is dropped: main.py still persists this state to Postgres
    exactly like a normal completion, just with `status="ESCALATED_TO_HUMAN"`.
    """
    settings = get_settings()

    reasons: list[str] = []
    if not state.get("used_llm_triage_router", False):
        reasons.append(
            "LLM triage reasoning was unavailable (timeout, malformed output, "
            "or backend outage) -- the pipeline fell back to rule-based "
            "classification, which is not trustworthy enough to auto-resolve."
        )
    triage_confidence = state.get("triage_confidence", 0.0)
    if triage_confidence < settings.min_confidence:
        reasons.append(
            f"triage confidence {triage_confidence:.2f} is below the "
            f"{settings.min_confidence:.2f} floor required for an automated resolution."
        )
    escalation_reason = " ".join(reasons) or "Escalated per policy (reason unspecified)."

    logger.warning(
        "fallback_human_escalation_node: ticket %s escalated to human review -- %s",
        state["ticket_id"], escalation_reason,
    )

    escalation_note = (
        "This ticket has been escalated to a human Tier-2/3 engineer for manual "
        f"review. Reason: {escalation_reason} No automated resolution steps were applied."
    )

    return {
        "status": "ESCALATED_TO_HUMAN",
        "escalation_reason": escalation_reason,
        # Deliberately overwrite whatever resolution_steps/summary the
        # failed or low-confidence attempt produced -- a caller reading
        # only `resolution_steps` must never mistake a discarded guess for
        # an actionable fix.
        "resolution_steps": [escalation_note],
        "summary": state.get("summary") or escalation_note,
    }


# ===========================================================================
# Node 5: generate_fix_node  (fix sub-graph 1/3)
# ===========================================================================


class CodeFixOutput(BaseModel):
    """Structured decision target for generate_fix_node. See LogInspectorOutput for why fields are flat."""

    model_config = ConfigDict(extra="forbid")

    fixed_file_content: str = Field(
        ...,
        min_length=1,
        max_length=30_000,
        description=(
            "The COMPLETE corrected contents of the file, ready to replace the "
            "original verbatim. No markdown code fences, no commentary -- raw "
            "source code only, exactly what should be written to disk."
        ),
    )
    explanation: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A one-to-three sentence explanation of the root cause and what this fix changes.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your genuine confidence that this fix is correct and safe to merge, 0.0-1.0.",
    )

    _normalize_confidence = field_validator("confidence", mode="before")(_normalize_confidence)


_CODE_FIX_SYSTEM_PROMPT = """You are an automated code-fix assistant for a Tier-1 support triage system.

You will be given the full current contents of one source file, plus the error/stack trace that was traced back to it. Produce a corrected version of the ENTIRE file that fixes the root cause.

Rules:
1. Return the complete file, not a diff or a snippet -- fixed_file_content replaces the original verbatim.
2. Preserve the file's existing style, structure, and every unrelated line exactly as-is -- change only what's needed to fix the reported error.
3. Do not wrap fixed_file_content in markdown code fences or add commentary inside it -- it must be directly-usable source code and nothing else.
4. If you cannot confidently identify a fix, still return your best attempt, but report a low confidence score honestly -- do not fabricate certainty."""

# Ollama-only text-mode counterpart to _CODE_FIX_SYSTEM_PROMPT (see
# call_code_fix_text's docstring in app/services/llm.py for why this
# exists: grammar-constrained JSON decoding is dramatically slower than
# free text for a large open-ended field). Same rules, different
# response contract -- delimited plain text instead of a JSON schema.
_CODE_FIX_TEXT_SYSTEM_PROMPT = """You are an automated code-fix assistant for a Tier-1 support triage system.

You will be given the full current contents of one source file, plus the error/stack trace that was traced back to it. Produce a corrected version of the ENTIRE file that fixes the root cause.

Respond in EXACTLY this plain-text format and nothing else -- no markdown code fences, no commentary before or after it:

===FIXED_FILE_START===
<the complete corrected file content, verbatim, ready to replace the original>
===FIXED_FILE_END===
===EXPLANATION===
<a one-to-three sentence explanation of the root cause and what this fix changes>
===CONFIDENCE===
<your genuine confidence this fix is correct and safe to merge, a number from 0.0 to 1.0>

Rules:
1. Preserve the file's existing style, structure, and every unrelated line exactly as-is -- change only what's needed to fix the reported error.
2. The FIXED_FILE section must contain nothing but raw source code -- no markdown fences, no commentary inside it.
3. If you cannot confidently identify a fix, still return your best attempt, but report a low confidence score honestly -- do not fabricate certainty."""

_CODE_FIX_TEXT_PATTERN = re.compile(
    r"===FIXED_FILE_START===\s*\n?(.*?)\n?===FIXED_FILE_END===\s*"
    r"===EXPLANATION===\s*\n?(.*?)\s*"
    r"===CONFIDENCE===\s*\n?([0-9.]+)",
    re.DOTALL,
)


def _parse_code_fix_text(raw: str) -> "CodeFixOutput":
    """
    Parse `call_code_fix_text`'s delimited plain-text response into a
    `CodeFixOutput` -- the free-text counterpart to Ollama's grammar-
    validated JSON path (`schema_model.model_validate_json` in
    app/services/llm.py's `_call_ollama_structured`). Raises
    `LLMMalformedOutputError` on anything unparseable so it folds into
    `call_code_fix_text`'s existing retry loop identically to a JSON
    validation failure.
    """
    match = _CODE_FIX_TEXT_PATTERN.search(raw)
    if not match:
        raise LLMMalformedOutputError(f"Could not parse code-fix text response: {raw[:300]!r}")

    fixed_content, explanation, confidence_str = match.groups()
    # Defensive: strip a markdown fence if the model wrapped the file in
    # one anyway despite rule 2 above -- cheap insurance, not a rule violation.
    fixed_content = fixed_content.strip("\n")
    fixed_content = re.sub(r"^```[^\n]*\n|\n```\s*$", "", fixed_content.strip())

    try:
        confidence = float(confidence_str.strip())
    except ValueError as exc:
        raise LLMMalformedOutputError(f"Could not parse confidence value: {confidence_str!r}") from exc

    try:
        return CodeFixOutput(
            fixed_file_content=fixed_content,
            explanation=explanation.strip() or "No explanation provided.",
            confidence=confidence,
        )
    except ValidationError as exc:
        raise LLMMalformedOutputError(f"Parsed code-fix text failed schema validation: {exc}") from exc


async def _resolve_and_fetch_file(state: TicketState) -> tuple[str, "GitHubFile"] | tuple[None, str]:
    """
    Shared prerequisite for the fix sub-graph: turn `state["affected_file"]`
    into a concrete (repo-relative path, current file content + blob sha)
    pair, resolving a bare browser filename to its real repo path if needed.

    Returns `(resolved_path, GitHubFile)` on success, or `(None, reason)` if
    the file can't be located/fetched -- the caller (generate_fix_node)
    turns the reason into a `fix_skipped_reason` and skips cleanly.
    """
    settings = get_settings()
    affected_file = state["affected_file"]
    owner, repo = settings.github_owner_repo  # caller has already checked this is set

    try:
        original = await get_file_content(owner, repo, affected_file, ref=settings.github_base_branch)
        return affected_file, original
    except GitHubError:
        # A browser stack trace only ever names the *served* filename
        # (e.g. "index.js"), not its repo-relative path -- see
        # find_file_by_basename's docstring. Try resolving before giving up.
        resolved_path = await find_file_by_basename(
            owner, repo, affected_file.rsplit("/", 1)[-1], ref=settings.github_base_branch
        )
        if resolved_path is None:
            return None, f"Could not find {affected_file} anywhere in {owner}/{repo}."
        try:
            original = await get_file_content(owner, repo, resolved_path, ref=settings.github_base_branch)
            return resolved_path, original
        except GitHubError as exc:
            return None, f"Resolved {affected_file} to {resolved_path} but could not fetch it: {exc}"


async def _generate_one_fix(
    *, affected_file: str, file_content: str, state: TicketState, prior_failure: str | None
) -> "CodeFixOutput":
    """
    One LLM call producing a full corrected file. Shared by every attempt of
    generate_fix_node. `prior_failure`, when set, is the actual test output
    from the previous sandbox run -- fed back so the model fixes what it
    actually broke instead of re-proposing the same patch.
    """
    settings = get_settings()
    file_excerpt = file_content[: settings.code_fix_max_file_chars]
    retry_block = ""
    if prior_failure:
        retry_block = (
            "\n\nYour PREVIOUS attempt at fixing this file was applied and its "
            "test suite was run in a sandbox. It FAILED with the following output. "
            "Produce a corrected version that makes these tests pass -- do not "
            "repeat the same mistake:\n"
            f"{prior_failure[: settings.sandbox_output_tail_chars]}"
        )
    user_prompt = (
        f"File: {affected_file}\n"
        f"Root error: {state['extracted_error']}\n"
        f"Stack trace:\n{state['stack_trace'][: settings.llm_max_trace_chars]}\n\n"
        f"Current file content:\n{file_excerpt}"
        f"{retry_block}"
    )

    if settings.llm_provider == "ollama":
        # Free-text path, not call_structured -- see call_code_fix_text's
        # docstring in app/services/llm.py: grammar-constrained JSON
        # decoding of a whole source file is dramatically slower than free
        # text on this backend.
        raw = await call_code_fix_text(
            system=_CODE_FIX_TEXT_SYSTEM_PROMPT, user=user_prompt, node_name="generate_fix_node"
        )
        return _parse_code_fix_text(raw)
    return await call_structured(
        system=_CODE_FIX_SYSTEM_PROMPT,
        user=user_prompt,
        schema_model=CodeFixOutput,
        node_name="generate_fix_node",
    )


@traced_node("generate_fix")
async def generate_fix_node(state: TicketState) -> dict:
    """
    Fix sub-graph node 1 of 3 (generate -> verify -> open_pr / escalate).

    Reached only for a confident, non-escalated triage. Fetches the
    implicated file from GitHub and asks the LLM for a complete corrected
    version, then computes a difflib diff against the original. On a retry
    (routed back here from verify_fix_node after a failed sandbox run) the
    prompt carries the previous attempt's real test output so the model
    corrects what it broke.

    Never fails the ticket: a missing affected_file, unconfigured
    GITHUB_REPO, a GitHub/LLM error, or an identical-to-original fix all
    set `fix_skipped_reason` and leave `fix_diff` empty -- `_route_after_generate`
    then sends the ticket straight to fix_escalation_node. This is the same
    graceful-degradation shape as every other node in this file.
    """
    settings = get_settings()
    attempt = state.get("fix_verification_attempts", 0) + 1

    base: dict = {
        "fix_attempted": True,
        "fix_verification_attempts": attempt,
        # carried forward / overwritten below
        "fix_skipped_reason": "",
        "fix_diff": state.get("fix_diff", ""),
        "fix_candidate_content": "",
        "fix_explanation": state.get("fix_explanation", ""),
        "fix_llm_confidence": state.get("fix_llm_confidence", 0.0),
        "fix_affected_path": state.get("fix_affected_path", ""),
    }

    if not state.get("affected_file"):
        return {**base, "fix_attempted": False,
                "fix_skipped_reason": "No affected file was identified in the stack trace."}
    if not settings.github_owner_repo:
        return {**base, "fix_attempted": False, "fix_skipped_reason": "GITHUB_REPO is not configured."}

    # Resolve the file once (attempt 1) and reuse the path/content on retries.
    affected_path = state.get("fix_affected_path") or ""
    original_content = state.get("fix_original_content") or ""
    original_sha = state.get("fix_original_sha") or ""
    carry: dict = {}
    if not affected_path:
        resolved = await _resolve_and_fetch_file(state)
        if resolved[0] is None:
            logger.warning("generate_fix_node: %s", resolved[1])
            return {**base, "fix_skipped_reason": resolved[1]}
        affected_path, original = resolved
        original_content, original_sha = original.content, original.sha
        carry = {
            "fix_affected_path": affected_path,
            "fix_original_content": original_content,
            "fix_original_sha": original_sha,
        }
    base.update(carry)
    base["fix_affected_path"] = affected_path

    prior_failure = state.get("fix_test_output_tail") or None
    try:
        fix = await _generate_one_fix(
            affected_file=affected_path,
            file_content=original_content,
            state=state,
            prior_failure=prior_failure,
        )
    except LLMError as exc:
        logger.warning("generate_fix_node: LLM fix generation failed: %s", exc)
        return {**base, "fix_skipped_reason": f"LLM fix generation failed: {exc}"}

    # Diff computed locally (difflib), never by the LLM -- guarantees a
    # syntactically valid unified diff every time.
    diff_text = "".join(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            fix.fixed_file_content.splitlines(keepends=True),
            fromfile=f"a/{affected_path}",
            tofile=f"b/{affected_path}",
        )
    )
    if not diff_text.strip():
        return {**base, "fix_skipped_reason": "LLM returned a fix identical to the current file -- nothing to propose."}

    return {
        **base,
        "fix_diff": diff_text,
        "fix_candidate_content": fix.fixed_file_content,
        "fix_explanation": fix.explanation,
        "fix_llm_confidence": fix.confidence,
    }


# ===========================================================================
# Node 6: verify_fix_node
# ===========================================================================


@traced_node("verify_fix")
async def verify_fix_node(state: TicketState) -> dict:
    """
    Fix sub-graph node 2 of 3: apply the candidate fix in a throwaway Docker
    container, run the repo's own test suite, and record the verdict.

    Outcomes (`_route_after_verify` keys off `fix_verification_status`):
      * PASSED               -> open_pr_node opens the PR, body says "verified".
      * FAILED_RETRY         -> back to generate_fix_node with the test output
                                (only while fix_verification_attempts < max).
      * FAILED_MAX_ATTEMPTS  -> fix_escalation_node: diff + last failure attached, no PR.
      * SKIPPED_NO_SANDBOX   -> open_pr_node, but body says "NOT verified"
                                (SANDBOX_ENABLED=false, or no Docker daemon).
      * SKIPPED_UNTESTABLE   -> fix_escalation_node: couldn't determine how to
                                test the repo, or the file wasn't in it.

    A test *failure* is a normal result, not an error. `SandboxError` (the
    verification couldn't be run at all) is caught here and mapped to
    SKIPPED_NO_SANDBOX so a Docker outage degrades to the honest-disclaimer
    PR path rather than 500-ing the request.
    """
    settings = get_settings()

    # Nothing to verify -- generate_fix_node already set a skip reason.
    if not state.get("fix_candidate_content"):
        return {"fix_verification_status": "SKIPPED_UNTESTABLE",
                "fix_verified": False, "fix_test_output_tail": "", "fix_test_command": ""}

    if not settings.sandbox_enabled:
        logger.info("verify_fix_node: SANDBOX_ENABLED=false -- skipping verification for ticket %s", state["ticket_id"])
        return {"fix_verification_status": "SKIPPED_NO_SANDBOX",
                "fix_verified": False, "fix_test_output_tail": "", "fix_test_command": ""}

    owner, repo = settings.github_owner_repo
    repo_url = f"https://github.com/{owner}/{repo}.git"

    try:
        result = await verify_patch_in_sandbox(
            repo_url=repo_url,
            ref=settings.github_base_branch,
            file_path=state["fix_affected_path"],
            new_content=state["fix_candidate_content"],
        )
    except SandboxError as exc:
        logger.warning("verify_fix_node: sandbox could not run (%s) -- degrading to unverified PR", exc)
        return {"fix_verification_status": "SKIPPED_NO_SANDBOX", "fix_verified": False,
                "fix_test_output_tail": str(exc)[: settings.sandbox_output_tail_chars],
                "fix_test_command": ""}

    if not result.ran:
        return {"fix_verification_status": "SKIPPED_UNTESTABLE", "fix_verified": False,
                "fix_test_output_tail": result.output_tail or result.skipped_reason,
                "fix_test_command": result.test_command}

    if result.passed:
        logger.info("verify_fix_node: fix PASSED tests for ticket %s (%.1fs)", state["ticket_id"], result.duration_s)
        return {"fix_verification_status": "PASSED", "fix_verified": True,
                "fix_test_output_tail": result.output_tail, "fix_test_command": result.test_command}

    attempts = state.get("fix_verification_attempts", 1)
    if attempts < settings.sandbox_max_attempts:
        logger.info("verify_fix_node: fix FAILED tests (attempt %d/%d) -- retrying",
                    attempts, settings.sandbox_max_attempts)
        return {"fix_verification_status": "FAILED_RETRY", "fix_verified": False,
                "fix_test_output_tail": result.output_tail, "fix_test_command": result.test_command}

    logger.info("verify_fix_node: fix FAILED tests after %d attempts -- escalating", attempts)
    return {"fix_verification_status": "FAILED_MAX_ATTEMPTS", "fix_verified": False,
            "fix_test_output_tail": result.output_tail, "fix_test_command": result.test_command}


# ===========================================================================
# Node 7: open_pr_node
# ===========================================================================


@traced_node("open_pr")
async def open_pr_node(state: TicketState) -> dict:
    """
    Fix sub-graph node 3 of 3: branch + commit + PR for a fix that either
    passed sandbox verification (PASSED) or was accepted unverified because
    the sandbox was unavailable (SKIPPED_NO_SANDBOX).

    The PR body states plainly which of those it is -- a "verified: N tests
    passed in an isolated sandbox" block for a real green run, or an
    explicit "NOT verified" disclaimer otherwise. It never merges (see
    app/services/github.py). A GitHub write failure sets `fix_skipped_reason`
    and does not fail the ticket.
    """
    settings = get_settings()
    result: dict = {"fix_skipped_reason": "", "fix_pr_url": "", "fix_branch_name": ""}

    if not settings.github_token:
        result["fix_skipped_reason"] = (
            "GITHUB_TOKEN is not configured -- computed"
            f"{' and verified' if state.get('fix_verified') else ''} a diff, "
            "but cannot open a branch/PR without write access."
        )
        return result

    owner, repo = settings.github_owner_repo
    affected_file = state["fix_affected_path"]
    short_id = state["ticket_id"][:8]
    branch_name = f"fde-fix/{short_id}"
    verified = state.get("fix_verified", False)

    if verified:
        verification_block = (
            f"**Verification:** ✅ the repo's test suite was run against this exact "
            f"change in an isolated Docker sandbox and passed (exit 0, "
            f"`{state.get('fix_test_command', 'tests')}`).\n\n"
            f"<details><summary>sandbox test output</summary>\n\n```\n"
            f"{state.get('fix_test_output_tail', '')[:3000]}\n```\n</details>\n\n"
        )
    else:
        verification_block = (
            "**Verification:** ⚠️ this fix was **NOT** verified -- the sandbox test "
            "runner was unavailable when it was generated. Run the test suite "
            "against this branch before merging.\n\n"
        )

    try:
        base_sha = await get_branch_head_sha(owner, repo, settings.github_base_branch)
        await create_branch(owner, repo, new_branch=branch_name, from_sha=base_sha)
        await commit_file_update(
            owner, repo,
            path=affected_file,
            new_content=state["fix_candidate_content"],
            sha=state["fix_original_sha"],
            branch=branch_name,
            message=f"Auto-fix: {state['extracted_error']} (ticket {short_id})",
        )
        pr_url = await open_pull_request(
            owner, repo,
            title=f"[FDE auto-fix] {state['extracted_error']} (ticket {short_id})",
            body=(
                f"Automated fix proposed by the FDE triage engine for ticket `{state['ticket_id']}`.\n\n"
                f"**Root cause:** {state['extracted_error']}\n\n"
                f"**LLM explanation:** {state.get('fix_explanation', 'n/a')}\n\n"
                f"**LLM confidence in this fix:** {state.get('fix_llm_confidence', 0.0):.2f}\n\n"
                f"{verification_block}"
                f"Attempts to a passing fix: {state.get('fix_verification_attempts', 1)}.\n\n"
                "This PR was opened automatically and is **not** merged -- review before merging."
            ),
            head=branch_name,
            base=settings.github_base_branch,
        )
    except GitHubError as exc:
        logger.warning("open_pr_node: GitHub branch/commit/PR failed: %s", exc)
        result["fix_skipped_reason"] = f"GitHub write failed: {exc}"
        return result

    logger.info("open_pr_node: opened PR %s for ticket %s", pr_url, state["ticket_id"])
    result["fix_pr_url"] = pr_url
    result["fix_branch_name"] = branch_name
    return result


# ===========================================================================
# Node 8: fix_escalation_node
# ===========================================================================


@traced_node("fix_escalation")
async def fix_escalation_node(state: TicketState) -> dict:
    """
    Terminal node for a fix that was attempted but must not become a PR:
    it never passed the sandbox after `sandbox_max_attempts`
    (FAILED_MAX_ATTEMPTS), or the repo couldn't be tested / the file
    wasn't found (SKIPPED_UNTESTABLE).

    The ticket's *triage* result is untouched and still valid -- this only
    records that automated remediation did not produce a merge-ready fix.
    The diff (if any) and the last failing test output are kept on the
    ticket so a human picks up exactly where the agent stopped, rather than
    from nothing. No PR is opened.
    """
    status = state.get("fix_verification_status", "SKIPPED_UNTESTABLE")
    if status == "FAILED_MAX_ATTEMPTS":
        reason = (
            f"A fix was generated and tested in a sandbox {state.get('fix_verification_attempts', 0)} "
            "time(s) but never passed the repo's test suite. The latest diff and its "
            "failing test output are attached for a human to take over -- no PR was opened."
        )
    else:
        reason = (
            state.get("fix_test_output_tail")
            or "The affected repo could not be tested automatically (unrecognized toolchain "
            "or the file was not present), so no verified fix could be produced. No PR was opened."
        )
    logger.info("fix_escalation_node: ticket %s -- %s", state["ticket_id"], status)
    return {
        "fix_skipped_reason": reason,
        "fix_pr_url": "",
        "fix_branch_name": "",
    }


# ===========================================================================
# Graph assembly
# ===========================================================================


def _route_after_triage(state: TicketState) -> str:
    """
    Conditional edge after triage_router_node.

    Diverts to fallback_human_escalation_node when the LLM triage call
    never succeeded (`used_llm_triage_router=False`) or self-reported
    confidence below `settings.min_confidence`. Otherwise the ticket is a
    confident, LLM-grounded triage and proceeds into the fix sub-graph at
    generate_fix_node -- a fix is only ever drafted for a diagnosis the
    pipeline itself trusts.
    """
    settings = get_settings()
    llm_failed = not state.get("used_llm_triage_router", False)
    low_confidence = state.get("triage_confidence", 0.0) < settings.min_confidence
    if llm_failed or low_confidence:
        return "fallback_human_escalation"
    return "generate_fix"


def _route_after_generate(state: TicketState) -> str:
    """
    Conditional edge after generate_fix_node. If a candidate fix (and diff)
    was produced, verify it; otherwise (no affected file, LLM error,
    no-op diff) there is nothing to test -- go straight to escalation.
    """
    if state.get("fix_candidate_content"):
        return "verify_fix"
    return "fix_escalation"


def _route_after_verify(state: TicketState) -> str:
    """
    Conditional edge after verify_fix_node -- the retry loop lives here.

      PASSED | SKIPPED_NO_SANDBOX -> open_pr  (PR body says which)
      FAILED_RETRY                -> generate_fix  (with the test output)
      FAILED_MAX_ATTEMPTS | SKIPPED_UNTESTABLE -> fix_escalation

    The FAILED_RETRY -> generate_fix edge is the only cycle in the whole
    graph. It is bounded by verify_fix_node itself, which only ever returns
    FAILED_RETRY while `fix_verification_attempts < settings.sandbox_max_attempts`
    and returns FAILED_MAX_ATTEMPTS after -- so the loop can execute at most
    `sandbox_max_attempts` times and always terminates.
    """
    status = state.get("fix_verification_status", "SKIPPED_UNTESTABLE")
    if status in ("PASSED", "SKIPPED_NO_SANDBOX"):
        return "open_pr"
    if status == "FAILED_RETRY":
        return "generate_fix"
    return "fix_escalation"


def build_triage_graph(checkpointer=None):
    """
    Assemble and compile the StateGraph:

        START -> log_inspector -> rag_lookup -> triage_router --+--> fallback_human_escalation -> END
                                                                |
                                                                +--> generate_fix --> verify_fix --+--> open_pr -------> END
                                                                        ^                           |
                                                                        +------(FAILED_RETRY)-------+--> fix_escalation -> END

    The first three edges stay linear/unconditional so LangGraph never runs
    a node until every edge feeding it has fired (rag_lookup never embeds a
    raw stack trace; triage_router never reasons without a chance at
    retrieved context).

    The fix sub-graph (generate_fix -> verify_fix -> open_pr | fix_escalation)
    is Phase 1's addition: a proposed fix is applied in a Docker sandbox and
    the repo's tests are run against it before any PR is opened. A failing
    fix loops back to generate_fix with the real test output, up to
    `settings.sandbox_max_attempts` times (the graph's only cycle, bounded
    by verify_fix_node), then escalates with the diff + failure attached.

    `checkpointer` is injected rather than constructed here because its
    backing Redis connection must be opened inside an async context manager
    (see app/main.py's lifespan) -- keeping this function synchronous and
    side-effect-free makes it safe to call at import time or from tests
    without a Redis instance running.
    """
    workflow = StateGraph(TicketState)

    workflow.add_node("log_inspector", log_inspector_node)
    workflow.add_node("rag_lookup", rag_lookup_node)
    workflow.add_node("triage_router", triage_router_node)
    workflow.add_node("fallback_human_escalation", fallback_human_escalation_node)
    workflow.add_node("generate_fix", generate_fix_node)
    workflow.add_node("verify_fix", verify_fix_node)
    workflow.add_node("open_pr", open_pr_node)
    workflow.add_node("fix_escalation", fix_escalation_node)

    workflow.add_edge(START, "log_inspector")
    workflow.add_edge("log_inspector", "rag_lookup")
    workflow.add_edge("rag_lookup", "triage_router")
    workflow.add_conditional_edges(
        "triage_router",
        _route_after_triage,
        {"fallback_human_escalation": "fallback_human_escalation", "generate_fix": "generate_fix"},
    )
    workflow.add_conditional_edges(
        "generate_fix",
        _route_after_generate,
        {"verify_fix": "verify_fix", "fix_escalation": "fix_escalation"},
    )
    workflow.add_conditional_edges(
        "verify_fix",
        _route_after_verify,
        {"generate_fix": "generate_fix", "open_pr": "open_pr", "fix_escalation": "fix_escalation"},
    )
    workflow.add_edge("fallback_human_escalation", END)
    workflow.add_edge("open_pr", END)
    workflow.add_edge("fix_escalation", END)

    return workflow.compile(checkpointer=checkpointer)
