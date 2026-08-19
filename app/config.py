"""
Centralized application configuration.

Every setting is read once from the environment / `.env` file via
pydantic-settings and exposed through a cached singleton (`get_settings`),
so the rest of the codebase never touches `os.environ` directly and every
module sees an identical, validated configuration object.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Embedding dimensions are a *hard* schema constraint: pgvector fixes the
# column width at DDL time (`Vector(768)`), so changing the embedding model
# to one with a different width requires a migration, not just a config
# edit. This table is the reference for the models we support out of the box.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,       # Ollama default; beats OpenAI ada-002 on MTEB
    "mxbai-embed-large": 1024,     # Ollama, higher quality / slower
    "all-minilm": 384,             # Ollama, smallest & fastest
    "text-embedding-3-small": 1536,  # OpenAI, for reference
}


class Settings(BaseSettings):
    """Strictly-typed application settings, sourced from env vars / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"

    # -- Postgres ---------------------------------------------------------
    postgres_user: str = "triage_user"
    postgres_password: str = "triage_password"
    postgres_db: str = "triage_engine"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # -- Redis --------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379

    # -- LLM reasoning backend ----------------------------------------------
    # "ollama"    -> fully local inference, no API key, no per-token cost.
    # "anthropic" -> Claude via the official SDK, for when the local model's
    #                reasoning quality is not enough for novel stack traces.
    llm_provider: Literal["ollama", "anthropic"] = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    # Qwen2.5-7B-Instruct is the sweet spot for structured extraction and
    # classification at ~5GB in q4. Swap for `qwen2.5:3b-instruct` on a
    # low-VRAM machine or `qwen2.5:14b-instruct` if you have the headroom.
    ollama_chat_model: str = "qwen2.5:7b-instruct"

    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None

    # Local inference on CPU can be genuinely slow -- a 7B model may take
    # 30-60s for a long trace. This ceiling is what turns a hung model into
    # a clean, catchable timeout that the node's fallback path can handle.
    llm_timeout_seconds: float = 120.0
    # One retry: constrained decoding guarantees *syntactically* valid JSON,
    # but a small model can still emit a semantically invalid value (e.g. a
    # severity outside the enum). The retry re-prompts with the error text.
    llm_max_retries: int = 1
    # Stack traces are truncated before they reach the LLM. The head of a
    # trace holds the exception; the tail is usually framework boilerplate.
    llm_max_trace_chars: int = 4_000

    # -- Embeddings ---------------------------------------------------------
    # "ollama"        -> real semantic embeddings from a local model.
    # "deterministic" -> offline lexical hashing vectorizer (see
    #                    app/services/embeddings.py). Lets the seed script,
    #                    tests, and CI run with no model server at all.
    embedding_provider: Literal["ollama", "deterministic"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_timeout_seconds: float = 60.0

    # -- RAG retrieval ------------------------------------------------------
    rag_top_k: int = 3
    # Cosine similarity floor. Below this, a "match" is noise -- feeding it
    # to the router would ground the answer in an unrelated incident, which
    # is worse than having no historical context at all.
    rag_min_similarity: float = 0.35

    # Confidence floor for LLM output. Day 4 routes anything below this to
    # the human-escalation node; Day 3 records it on the response.
    min_confidence: float = 0.70

    @property
    def database_url(self) -> str:
        """
        Async SQLAlchemy connection string.

        Uses the `asyncpg` driver (not psycopg2) so `create_async_engine`
        in app/database.py can issue non-blocking queries from FastAPI's
        event loop.
        """
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """psycopg2-style URL, for `psql` / EXPLAIN ANALYZE tooling only."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Connection string used by the LangGraph Redis checkpointer."""
        return f"redis://{self.redis_host}:{self.redis_port}/0"


@lru_cache
def get_settings() -> Settings:
    """
    Process-wide settings singleton.

    `lru_cache` (with no arguments) memoizes the single call, so `Settings()`
    -- which parses the environment -- runs exactly once per process instead
    of on every import/request.
    """
    return Settings()
