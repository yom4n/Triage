"""
SQLAlchemy ORM models.

Split out of `app/database.py` in Phase 2: the engine/session machinery and
the table definitions now change for different reasons and at different
rates, and scripts (like `scripts/seed_tickets.py`) want the models without
dragging in engine configuration. `app/database.py` re-exports `Base` and
`Ticket` so existing imports keep working.
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

# Read once at import time. pgvector fixes a vector column's width in the
# DDL, so this cannot vary per-row or per-request -- changing
# EMBEDDING_DIM requires dropping/migrating the column, not just a restart.
EMBEDDING_DIM = get_settings().embedding_dim


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


class Ticket(Base):
    """
    Persisted record of a triaged support ticket.

    This table plays two roles at once, which is the whole point of keeping
    vectors in Postgres rather than a separate vector store:

    1. **System of record** for tickets the API has processed.
    2. **RAG corpus** -- rows with a non-null `embedding` and a verified
       `resolution` are the historical knowledge that `rag_lookup_node`
       retrieves to ground new triage decisions.

    Because both live in one table, a ticket becomes searchable in the same
    transaction that creates it. There is no sync job and no window where
    the vector store disagrees with the source of truth.
    """

    __tablename__ = "tickets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    stack_trace: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="human", server_default="human", index=True
    )

    # -- Written by log_inspector_node --------------------------------------
    extracted_error: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    exception_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    affected_file: Mapped[str | None] = mapped_column(String(500), nullable=True)
    affected_line: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # -- Written by triage_router_node --------------------------------------
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # JSONB (not JSON): Postgres stores it decomposed and can index into it,
    # so "which tickets recommend restarting the pool?" stays a SQL query.
    resolution_steps: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- Written by fallback_human_escalation_node (Day 4), or defaulted to
    #    "COMPLETED" for a normal, confident, LLM-grounded triage --------
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="COMPLETED", server_default="COMPLETED", index=True
    )
    escalation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- Written by the fix sub-graph: generate_fix -> verify_fix ->
    #    open_pr | fix_escalation (only reached for non-escalated tickets --
    #    see app/graph.py's build_triage_graph routing) --------------------
    fix_attempted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # Set whenever a fix attempt stopped short of opening a PR (no affected
    # file, no GITHUB_REPO/GITHUB_TOKEN configured, a GitHub/LLM failure, an
    # identical-to-original fix, or -- Phase 1 -- a fix that never passed the
    # sandbox after sandbox_max_attempts). Lets a caller distinguish "never
    # tried" from "tried and here's exactly why it stopped".
    fix_skipped_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Unified diff computed locally (difflib), not LLM-generated. Populated
    # even when no PR was opened (no token, or escalated after a failed
    # verification), so the fix is still visible/reviewable.
    fix_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    fix_pr_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    fix_branch_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # -- Phase 1: sandboxed fix verification (verify_fix_node) -------------
    # Whether the proposed fix passed the monitored repo's own test suite,
    # run against this exact change in a throwaway Docker container, before
    # any PR was opened. A PR only carries a "verified" body when this is true.
    fix_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # PASSED | FAILED_MAX_ATTEMPTS | SKIPPED_NO_SANDBOX | SKIPPED_UNTESTABLE
    # | NOT_ATTEMPTED -- see verify_fix_node / _route_after_verify in app/graph.py.
    fix_verification_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="NOT_ATTEMPTED", server_default="NOT_ATTEMPTED"
    )
    # How many generate -> verify loops ran before a green fix or escalation.
    fix_verification_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # The command whose exit code was the verdict (e.g. "python -m pytest -q").
    fix_test_command: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # Tail of the sandbox container's stdout/stderr -- the passing run's
    # output for a verified fix, or the last failure for an escalated one.
    fix_test_output_tail: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- RAG corpus fields ---------------------------------------------------
    # The verified fix. This is the payload retrieval exists to surface --
    # `extracted_error` tells the router *what* broke before, `resolution`
    # tells it what actually fixed it.
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Only human-verified rows are allowed to ground a new recommendation.
    # Without this gate the corpus feeds the model its own past guesses and
    # confidently amplifies them -- a self-reinforcing hallucination loop.
    is_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false", index=True
    )

    # The exact string that was embedded. Kept so we can re-embed the corpus
    # after a model upgrade without re-deriving the text, and so a bad
    # retrieval can be debugged by reading what actually went into the vector.
    embedding_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Ticket id={self.id} severity={self.severity!r} "
            f"error={self.extracted_error!r}>"
        )
