"""
Async vector retrieval service (RAG).

Executes cosine-similarity search directly in Postgres via pgvector's
`<=>` operator, so "find tickets like this one" is a single indexed SQL
query -- no separate vector database, no out-of-band sync job, and the
search runs in the same transaction as everything else in the request.
"""
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Ticket

logger = logging.getLogger("triage_engine.rag")


@dataclass(frozen=True)
class SimilarTicket:
    """One retrieval hit: a historical ticket plus its similarity to the query."""

    ticket_id: str
    title: str
    extracted_error: str
    resolution: str
    similarity: float  # cosine similarity, roughly in [0, 1] for embeddings of this kind


async def find_similar_tickets(
    db: AsyncSession,
    query_vector: list[float],
    limit: int = 3,
    *,
    min_similarity: float | None = None,
    require_verified: bool = True,
) -> list[SimilarTicket]:
    """
    Return up to `limit` historical tickets whose embedding is closest to
    `query_vector` by cosine distance.

    pgvector exposes its distance operators as comparator methods on a
    `Vector` column: `.l2_distance()` (`<->`), `.cosine_distance()`
    (`<=>`), and `.max_inner_product()` (`<#>`). We use cosine distance
    because embedding *magnitude* carries no meaning for these models --
    only direction (semantic content) should determine similarity.
    `cosine_distance` returns `1 - cosine_similarity`, so callers see the
    more intuitive similarity score after we convert it back below.

    `require_verified=True` (the default, and what the live triage path
    always uses) restricts the search to `Ticket.is_verified` rows: only
    tickets where a human confirmed the recorded `resolution` actually
    fixed the problem are allowed to ground a new recommendation. Without
    this gate, the corpus would eventually include the model's own
    unverified past guesses (every triaged ticket gets embedded and
    stored -- see app/main.py), and mistakes would compound across
    tickets instead of staying anchored to verified outcomes.
    """
    settings = get_settings()
    if min_similarity is None:
        min_similarity = settings.rag_min_similarity

    distance = Ticket.embedding.cosine_distance(query_vector)
    stmt = (
        select(Ticket, distance.label("distance"))
        .where(Ticket.embedding.is_not(None))
        .where(Ticket.resolution.is_not(None))
    )
    if require_verified:
        stmt = stmt.where(Ticket.is_verified.is_(True))
    # ORDER BY directly on the distance expression is what lets the query
    # planner use the HNSW index built in app/database.py instead of a
    # sequential scan + in-memory sort -- ordering by a derived/wrapped
    # expression (e.g. `-distance`) silently defeats the index. See
    # scripts/vector_index_analysis.sql for EXPLAIN ANALYZE evidence.
    stmt = stmt.order_by(distance).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    hits: list[SimilarTicket] = []
    for ticket, distance_value in rows:
        similarity = 1.0 - float(distance_value)
        if similarity < min_similarity:
            continue
        hits.append(
            SimilarTicket(
                ticket_id=str(ticket.id),
                title=ticket.title,
                extracted_error=ticket.extracted_error,
                resolution=ticket.resolution,
                similarity=similarity,
            )
        )

    logger.info(
        "RAG retrieval: %d/%d candidates passed similarity >= %.2f (require_verified=%s)",
        len(hits), len(rows), min_similarity, require_verified,
    )
    return hits
