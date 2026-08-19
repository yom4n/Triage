"""
Database layer: async SQLAlchemy engine, session factory, and bootstrap DDL.

ORM models moved to `app/models.py` in Phase 2; they are re-exported here so
`from app.database import Ticket` keeps working.

Design notes
------------
* The engine and session factory are lazy singletons (`get_engine` /
  `get_session_maker`, memoized with `lru_cache`). `create_async_engine()`
  itself never opens a TCP socket -- SQLAlchemy's connection pool only
  dials the database on the first checkout (the first query a request
  actually runs). This means importing this module -- e.g. in unit tests,
  or before Postgres is reachable -- never fails on a connection error.
* Every I/O method is async so a slow query never blocks the FastAPI event
  loop, which would otherwise stall *every other* in-flight request
  (including concurrent LLM/graph calls) on the same worker process.
"""
from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.models import EMBEDDING_DIM, Base, Ticket

__all__ = [
    "Base",
    "Ticket",
    "EMBEDDING_DIM",
    "get_engine",
    "get_session_maker",
    "get_db",
    "init_models",
    "dispose_engine",
]


@lru_cache
def get_engine() -> AsyncEngine:
    """
    Return the process-wide async engine, constructing it on first call.

    `lru_cache` turns this into a lazy singleton: the engine (and the
    connection pool it owns) is only built the first time something
    actually needs a DB connection, and every later caller reuses the same
    instance/pool rather than opening a new one per request.
    """
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=False,  # pgvector literals are huge; enable per-query if needed
        pool_pre_ping=True,  # validate pooled connections before reuse
        future=True,
    )


@lru_cache
def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Lazily build the session factory bound to the lazy engine above."""
    return async_sessionmaker(
        bind=get_engine(), expire_on_commit=False, class_=AsyncSession
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding a request-scoped AsyncSession.

    The underlying connection is only checked out of the pool once a route
    handler executes a query -- not when this dependency is constructed --
    which preserves the lazy-connection guarantee end to end.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        yield session


# HNSW over cosine distance. Two reasons this is the default over IVFFlat:
#
#   * IVFFlat must be built *after* the table holds representative data
#     (it clusters existing rows into lists). Building it on an empty table
#     produces a useless index. HNSW is a graph built incrementally, so it
#     works on an empty table and stays correct as rows arrive.
#   * HNSW gives better recall-vs-speed at our corpus size.
#
# `vector_cosine_ops` must match the operator the query uses (`<=>`).
# An index built with `vector_l2_ops` is simply ignored by a `<=>` ORDER BY --
# the planner falls back to a sequential scan and you get a silent
# performance cliff rather than an error. See scripts/vector_index_analysis.sql.
_HNSW_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS tickets_embedding_hnsw_cosine_idx
ON tickets USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""


async def init_models() -> None:
    """
    Dev/bootstrap helper: ensure the pgvector extension, tables, and vector
    index exist.

    This is the first point in the app's lifecycle where a real connection
    is opened (called from main.py's lifespan startup, never at import
    time). In a production deployment this would be replaced by versioned
    Alembic migrations run in CI/CD; it's kept here so the service is
    runnable end-to-end with zero extra setup steps.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # Must precede create_all: the `vector` type has to exist before a
        # column can be declared with it.
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.create_all)
        await conn.exec_driver_sql(_HNSW_INDEX_DDL)


async def dispose_engine() -> None:
    """Close all pooled connections cleanly on application shutdown."""
    engine = get_engine()
    await engine.dispose()
