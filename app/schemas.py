"""
Pydantic v2 request/response contracts and input guardrails.

These models are the first line of defense in the pipeline: malformed,
low-signal, or adversarial payloads are rejected here -- at the API
boundary -- before they can reach the LangGraph pipeline and burn LLM
tokens on garbage input. Failing fast here is strictly cheaper than
failing inside an agent node three steps into a graph run.
"""
import re
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class EnvironmentEnum(str, Enum):
    """Deployment environment a ticket originated in. Drives severity routing."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SeverityEnum(str, Enum):
    """Tier-1 severity classification assigned by triage_router_node."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TriageStatusEnum(str, Enum):
    """
    Final disposition of a triage run.

    COMPLETED         -- an LLM-grounded triage result at or above
                          `settings.min_confidence`; safe to act on directly.
    ESCALATED_TO_HUMAN -- fallback_human_escalation_node fired (see
                          app/graph.py's `_route_after_triage`): the LLM
                          call timed out / returned malformed output /
                          yielded low confidence. `resolution_steps` holds
                          a single hand-off instruction, not an automated fix.
    """

    COMPLETED = "COMPLETED"
    ESCALATED_TO_HUMAN = "ESCALATED_TO_HUMAN"


# Extremely low-effort reports ("it broke", "bug pls fix") almost never
# contain a real error signature. Requiring a minimum length forces the
# reporter to paste an actual stack trace / log excerpt worth analyzing.
_MIN_STACK_TRACE_LEN = 30
# Bounds how much text can ever reach the LLM/embedding stage per ticket,
# capping worst-case token cost and latency for a single request.
_MAX_STACK_TRACE_LEN = 8_000

# Cheap denylist for obviously hostile payloads (stored-XSS / markup
# injection). This is not a substitute for output encoding wherever this
# text is later rendered (e.g. an admin dashboard) -- it just stops
# garbage from ever reaching the LLM or the database in the first place.
_SUSPICIOUS_MARKUP = re.compile(
    r"<\s*script|javascript:|on\w+\s*=\s*['\"]", re.IGNORECASE
)


class TicketCreate(BaseModel):
    """Inbound payload for POST /api/v1/triage."""

    # extra="forbid" rejects unknown fields outright instead of silently
    # dropping them -- a caller sending a typo'd field name gets a 422
    # instead of confusingly-ignored input.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Short human-readable summary of the issue.",
    )
    stack_trace: str = Field(
        ...,
        min_length=_MIN_STACK_TRACE_LEN,
        max_length=_MAX_STACK_TRACE_LEN,
        description="Raw stack trace or log excerpt reproducing the failure.",
    )
    environment: EnvironmentEnum = Field(
        ..., description="Deployment environment the error was observed in."
    )
    description: str | None = Field(
        default=None,
        max_length=2_000,
        description="Optional free-text context supplied by the reporter.",
    )
    reporter_email: EmailStr | None = Field(
        default=None,
        description="Optional email of the person filing the ticket.",
    )

    @field_validator("stack_trace")
    @classmethod
    def validate_stack_trace(cls, value: str) -> str:
        # Strip embedded NUL bytes: Postgres TEXT columns reject them
        # outright, and they're a classic truncation/injection trick.
        cleaned = value.replace("\x00", "")

        if _SUSPICIOUS_MARKUP.search(cleaned):
            raise ValueError("stack_trace contains disallowed markup/script content")

        # Reject payloads that pad past min_length with whitespace to game
        # the length check (e.g. "bug" followed by 30 spaces).
        if len(cleaned.strip()) < _MIN_STACK_TRACE_LEN:
            raise ValueError(
                f"stack_trace must contain at least {_MIN_STACK_TRACE_LEN} "
                "non-whitespace characters of real log content"
            )

        return cleaned

    @field_validator("title", "description")
    @classmethod
    def validate_no_markup(cls, value: str | None) -> str | None:
        if value and _SUSPICIOUS_MARKUP.search(value):
            raise ValueError("field contains disallowed markup/script content")
        return value


class CrashReport(BaseModel):
    """
    Raw crash payload posted by a client-side error-capture hook (see
    `client-sdks/` at the repo root) -- Phase 4 (auto-ticketing from a
    running app's own crashes), the automated counterpart to a human
    filling out the `TicketCreate` form by hand.

    Deliberately looser than `TicketCreate`: a browser's `window.onerror`/
    `unhandledrejection`/React error-boundary handlers hand you whatever
    the runtime gives them, not a guaranteed 30-character stack trace, so
    this model accepts a minimal, mostly-optional shape and
    `crash_report_to_ticket_create()` below does the work of turning it
    into something that clears `TicketCreate`'s guardrails.
    """

    # populate_by_name + aliases: the SDK is plain JS/TS and sends
    # camelCase (`componentStack`, `userAgent`) -- the natural convention
    # on that side -- while the rest of this backend is snake_case.
    # extra="ignore" (not "forbid" like TicketCreate) because this is a
    # cross-repo wire contract: a newer SDK version sending one extra
    # field should never 422 an otherwise-valid crash report.
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True, extra="ignore")

    message: str = Field(..., min_length=1, max_length=2000, description="Error.message, or the thrown value's string form.")
    stack: str | None = Field(default=None, max_length=8000, description="Error.stack, if the runtime populated one.")
    component_stack: str | None = Field(
        default=None, max_length=8000, alias="componentStack",
        description="React error-boundary componentStack, if this came from CrashBoundary.tsx.",
    )
    url: str | None = Field(default=None, max_length=2000, description="window.location.href at the time of the crash.")
    user_agent: str | None = Field(default=None, max_length=500, alias="userAgent")
    source: str = Field(
        default="window.onerror", max_length=50,
        description="Which hook fired: window.onerror | unhandledrejection | react-error-boundary.",
    )
    environment: EnvironmentEnum = Field(
        default=EnvironmentEnum.PRODUCTION,
        description="Deployment tier the crashed app is running as. Defaults to production since that's the typical deployment for a monitored app.",
    )


def crash_report_to_ticket_create(report: CrashReport) -> TicketCreate:
    """
    Normalize a loose, runtime-provided `CrashReport` into a valid
    `TicketCreate` -- the one adapter function every client-side capture
    hook's output must pass through before it can enter the same pipeline
    a human-filed ticket does. Centralizing this here (rather than
    duplicating ad hoc mapping logic at each ingestion route) is what
    keeps the language/framework-specific bit (a browser crash shape)
    from leaking into the pipeline itself -- see the note in README.md
    about keeping ingestion adapters thin and swappable.
    """
    message = report.message.strip()
    title = message[:200]
    if len(title) < 5:
        # TicketCreate requires a 5-char title; a bare thrown value like
        # "x" or "" clears CrashReport's min_length=1 but not that floor.
        title = (f"Crash: {title}" if title else "Unhandled client-side crash")[:200]

    trace_parts = [part.strip() for part in (report.stack, report.component_stack) if part and part.strip()]
    if not trace_parts:
        # No real stack available (some thrown values carry none) --
        # fall back to the message itself so there's still real content
        # to extract from, rather than 422ing a crash we could triage.
        trace_parts.append(message)
    stack_trace = "\n\nComponent stack:\n".join(trace_parts) if len(trace_parts) > 1 else trace_parts[0]

    context_line = f"source: {report.source} | url: {report.url or 'unknown'}"
    stack_trace = f"{stack_trace}\n{context_line}"[:8000]
    if len(stack_trace.strip()) < _MIN_STACK_TRACE_LEN:
        # Still short (e.g. a one-word message, no stack, no url) -- pad
        # deterministically with real metadata rather than junk filler.
        stack_trace = f"{stack_trace}\nreported_at: client-side crash capture"[:8000]

    description_parts = [context_line]
    if report.user_agent:
        description_parts.append(f"user_agent: {report.user_agent}")
    description = " | ".join(description_parts)[:2000]

    return TicketCreate(
        title=title,
        stack_trace=stack_trace,
        environment=report.environment,
        description=description,
    )


class TicketResponse(BaseModel):
    """Structured triage result returned to the calling system."""

    # from_attributes lets this model be built directly from the SQLAlchemy
    # `Ticket` ORM instance (response = TicketResponse.model_validate(orm_obj))
    # in addition to being built from plain kwargs.
    model_config = ConfigDict(from_attributes=True)

    ticket_id: uuid.UUID
    title: str
    environment: EnvironmentEnum
    # Round-tripped so a dashboard fetching this via GET /api/v1/tickets can
    # render the original trace without a second request -- POST callers
    # already have it (they sent it), but a list/detail view reading a
    # ticket back has no other source for it.
    stack_trace: str
    extracted_error: str
    affected_file: str | None = None
    affected_line: int | None = None
    severity: SeverityEnum
    summary: str
    # Ordered remediation steps -- from LLM reasoning grounded in retrieved
    # historical tickets when available, or a single generic escalation
    # step if the LLM backend was unavailable (see triage_router_node's
    # fallback path in app/graph.py).
    resolution_steps: list[str]
    # The router's self-reported confidence in [0, 1]. Phase 2 only
    # records this; Phase 3 (Day 4) is what routes low-confidence results
    # to human escalation instead of returning them as-is.
    confidence: float
    # How many verified historical tickets were retrieved and actually
    # used to ground this recommendation -- 0 means the LLM reasoned from
    # the stack trace alone, with no matching precedent in the corpus.
    similar_tickets_considered: int = 0
    # Day 4: final disposition after fallback_human_escalation_node's
    # conditional routing (app/graph.py's `_route_after_triage`). Callers
    # should branch on this before trusting `resolution_steps` as an
    # automated fix -- ESCALATED_TO_HUMAN means it is a hand-off note.
    status: TriageStatusEnum = TriageStatusEnum.COMPLETED
    escalation_reason: str | None = None
    # The fix sub-graph's outcome (app/graph.py: generate_fix -> verify_fix
    # -> open_pr | fix_escalation). fix_attempted distinguishes "never
    # tried" (escalated ticket, no affected_file) from "tried" -- check
    # fix_skipped_reason for why it stopped short of a PR, or fix_pr_url for
    # the PR itself. fix_diff is populated whenever the LLM produced a
    # change, even when no PR was opened.
    fix_attempted: bool = False
    fix_skipped_reason: str | None = None
    fix_diff: str | None = None
    fix_pr_url: str | None = None
    fix_branch_name: str | None = None
    # Phase 1: sandboxed verification. fix_verified is True only when the
    # monitored repo's own test suite passed against this exact change in a
    # Docker sandbox before the PR was opened. fix_verification_status is
    # one of PASSED | FAILED_MAX_ATTEMPTS | SKIPPED_NO_SANDBOX |
    # SKIPPED_UNTESTABLE | NOT_ATTEMPTED.
    fix_verified: bool = False
    fix_verification_status: str = "NOT_ATTEMPTED"
    fix_verification_attempts: int = 0
    fix_test_command: str | None = None
    fix_test_output_tail: str | None = None
    created_at: datetime
