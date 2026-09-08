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

    # -- CORS (Phase 4: crash ingestion from a browser-hosted app) ----------
    # The monitored app (e.g. the todo-app test project) POSTs crash
    # reports to /api/v1/ingest/crash directly from the browser, which is
    # a different origin (different port at minimum) than this API -- the
    # browser will refuse that request without a matching CORS header.
    # Comma-separated; "*" (the default) is fine for a local/demo project
    # monitoring apps you control, but a real deployment should list exact
    # origins instead.
    cors_allow_origins: str = "*"

    @property
    def cors_allow_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    # -- GitHub auto-fix (Phase 4b: propose_fix_node) ------------------------
    # `github_token` is optional by design: propose_fix_node checks for it
    # and no-ops (skips the fix attempt, ticket still completes normally)
    # rather than failing the whole triage request when it's unset -- the
    # crash-ingestion/triage loop must keep working with zero GitHub setup;
    # code-fix proposal is additive on top of it, not a hard dependency.
    # A fine-grained PAT scoped to just this repo, with "Contents:
    # read and write" + "Pull requests: read and write", is enough --
    # broader `repo` scope is not required.
    github_token: str | None = None
    # "owner/repo", e.g. "yom4n/todotest" -- the single repo this instance
    # monitors. One-repo-per-backend-instance is a deliberate simplification
    # (see README) -- a real multi-tenant version would resolve this per
    # ticket instead of from one global setting.
    github_repo: str | None = None
    github_base_branch: str = "main"
    github_api_base_url: str = "https://api.github.com"
    # Separate from llm_timeout_seconds: GitHub API calls are typically
    # sub-second, so a much shorter ceiling turns a hung/rate-limited call
    # into a clean, catchable failure instead of tying up the request for
    # as long as an LLM call is allowed to.
    github_timeout_seconds: float = 15.0
    # Source files are truncated before reaching the LLM, same reasoning
    # as llm_max_trace_chars -- caps worst-case prompt size/cost, and a
    # file this large is a sign the fix-proposal node should skip it
    # rather than have a small local model guess at a huge unseen tail.
    code_fix_max_file_chars: int = 8_000

    # -- Sandboxed fix verification (Phase 1) -------------------------------
    # Before a proposed fix is turned into a PR, generate_fix_node's output
    # is applied inside a throwaway Docker container and the repo's own test
    # suite is run against it. Only a fix whose tests pass (exit 0) reaches
    # open_pr_node; a fix that never goes green after `sandbox_max_attempts`
    # is escalated with the diff + failing test output attached, never
    # merged blind. See app/services/sandbox.py and app/graph.py.
    #
    # sandbox_enabled=false short-circuits verify_fix_node to the pre-Phase-1
    # behavior (open a PR with an honest "unverified" body) so the pipeline
    # still runs end-to-end on a host without a Docker daemon.
    sandbox_enabled: bool = True
    # Docker daemon socket / host. Left as None, the docker SDK reads the
    # environment (DOCKER_HOST, or the default local socket) itself.
    sandbox_docker_host: str | None = None
    # Per-language default base images, overridable wholesale for a repo
    # that needs a specific toolchain. Detection (package.json vs
    # requirements.txt vs pyproject.toml) picks one of these in
    # app/services/sandbox.py.
    sandbox_image_python: str = "python:3.12-slim"
    sandbox_image_node: str = "node:20-slim"
    # Forces a specific image regardless of what the repo looks like.
    sandbox_image_override: str | None = None
    # Forces a specific test command (e.g. "pytest -q tests/unit") instead
    # of the per-language default the runner would otherwise infer.
    sandbox_test_command_override: str | None = None
    # Hard wall-clock ceiling for the clone+install+test sequence inside the
    # container. A hung test run must not tie up a triage request forever --
    # this turns it into a clean "verification timed out" -> escalation.
    sandbox_timeout_seconds: float = 240.0
    # How many times generate_fix_node -> verify_fix_node may loop before
    # giving up and escalating. Each retry re-prompts the LLM with the
    # previous attempt's actual test failure output.
    sandbox_max_attempts: int = 3
    # How much of the container's stdout/stderr tail to keep on the ticket
    # (and feed back into the retry prompt). Caps DB row size and prompt cost.
    sandbox_output_tail_chars: int = 4_000
    # Many real test suites need a database. When the monitored repo has a
    # docker-compose.yml declaring a `postgres`/`db`/`mysql` service, the
    # sandbox starts that one service as a sidecar on a private network and
    # runs the test container alongside it, injecting the connection URL as
    # an env var (SANDBOX_DB_URL_ENV, default DATABASE_URL). This keeps the
    # verification faithful to how the repo's own tests expect to run,
    # without this service needing to understand the repo's schema.
    sandbox_provision_db: bool = True
    sandbox_db_url_env: str = "DATABASE_URL"
    # Extra time allowed just for the DB sidecar to report healthy, on top
    # of sandbox_timeout_seconds for the test run itself.
    sandbox_db_startup_seconds: float = 60.0

    @property
    def github_owner_repo(self) -> tuple[str, str] | None:
        """Split `github_repo` ("owner/repo") into its two parts, or None if unset/malformed."""
        if not self.github_repo or "/" not in self.github_repo:
            return None
        owner, _, repo = self.github_repo.partition("/")
        return (owner, repo) if owner and repo else None

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
