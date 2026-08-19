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


class TicketResponse(BaseModel):
    """Structured triage result returned to the calling system."""

    # from_attributes lets this model be built directly from the SQLAlchemy
    # `Ticket` ORM instance (response = TicketResponse.model_validate(orm_obj))
    # in addition to being built from plain kwargs.
    model_config = ConfigDict(from_attributes=True)

    ticket_id: uuid.UUID
    title: str
    environment: EnvironmentEnum
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
    created_at: datetime
