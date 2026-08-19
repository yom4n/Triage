"""
Embedding generation service.

Converts ticket text (title + stack trace + description) into a dense
vector so semantically similar bugs can be found even when the wording is
completely different (e.g. "checkout hangs under load" vs. "payment API
times out at peak traffic" -- no word overlap, same root cause).

Two backends, selected by `settings.embedding_provider`:

* **ollama** (default) -- calls a local Ollama server's `/api/embed`
  endpoint. Fully offline once the model is pulled, no API key, no
  per-call cost, no rate limit. This is what makes `docker compose up` +
  `ollama pull nomic-embed-text` a complete local RAG stack.
* **deterministic** -- a dependency-free hashing vectorizer (a bag-of-
  words hashed into a fixed-width vector, L2-normalized). Not semantically
  meaningful, but it is byte-for-byte reproducible and requires no running
  service, which is exactly what CI and the seed script's dry-run mode
  need. It is never selected implicitly for the live API path -- only via
  explicit config -- so a misconfigured deployment can't silently serve
  fake similarity scores.

Both implementations return a plain `list[float]` of length
`settings.embedding_dim`, matching the `Vector(EMBEDDING_DIM)` column on
`Ticket` (app/models.py). Changing embedding model/dimension is a schema
change, not a config change -- see the comment on EMBEDDING_DIM there.
"""
import hashlib
import logging
import math
import re
from functools import lru_cache

import httpx

from app.config import KNOWN_EMBEDDING_DIMS, get_settings

logger = logging.getLogger("triage_engine.embeddings")


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails or returns a malformed vector."""


def build_embedding_text(*, title: str, extracted_error: str, stack_trace: str, description: str = "") -> str:
    """
    Compose the canonical string that gets embedded for a ticket.

    Centralizing this (rather than letting callers hand-assemble strings)
    guarantees the seed script, the live triage path, and any future
    re-embedding job all vectorize text the same way -- a query embedded
    with a different field order/weighting than the corpus silently
    degrades cosine similarity for reasons that are very hard to debug
    after the fact.

    The extracted error is repeated up front (with the raw trace truncated
    to its head) so the exception signature dominates the embedding over
    boilerplate framework noise, which usually makes up most of a trace's
    tokens.
    """
    settings = get_settings()
    trace_excerpt = stack_trace[: settings.llm_max_trace_chars]
    parts = [extracted_error, title, description, trace_excerpt]
    return "\n".join(p.strip() for p in parts if p and p.strip())


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


async def _embed_ollama(text: str) -> list[float]:
    """
    Call Ollama's native embedding endpoint.

    POST /api/embed accepts either a single string or a list under `input`
    and returns `embeddings`: a list of vectors (one per input). We always
    send a single-element batch and unwrap it, which keeps this function's
    signature symmetric with `_embed_deterministic` below.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"
    payload = {"model": settings.ollama_embedding_model, "input": text}

    try:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise EmbeddingError(
            f"Ollama embedding request timed out after {settings.embedding_timeout_seconds}s "
            f"(model={settings.ollama_embedding_model}). Is `ollama serve` running and is the "
            "model pulled? Try: `ollama pull nomic-embed-text`."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise EmbeddingError(
            f"Ollama embedding request failed with {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except httpx.ConnectError as exc:
        raise EmbeddingError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            "Start it with `ollama serve` (or the Ollama desktop app)."
        ) from exc

    body = response.json()
    try:
        vector = body["embeddings"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingError(f"Unexpected Ollama embedding response shape: {body!r}") from exc

    if len(vector) != settings.embedding_dim:
        raise EmbeddingError(
            f"Ollama model '{settings.ollama_embedding_model}' returned a "
            f"{len(vector)}-dim vector, but settings.embedding_dim="
            f"{settings.embedding_dim} (and the DB column is fixed at that "
            f"width). Known dims: {KNOWN_EMBEDDING_DIMS}. Update "
            "EMBEDDING_DIM/settings.embedding_dim to match your model, "
            "then re-run migrations."
        )
    return [float(x) for x in vector]


# ---------------------------------------------------------------------------
# Deterministic offline backend
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def _embed_deterministic(text: str, dim: int) -> list[float]:
    """
    Dependency-free hashing vectorizer: tokenize, hash each token into a
    bucket in [0, dim), accumulate a signed count per bucket (the sign
    itself is also hash-derived, which is the standard "hashing trick" and
    keeps the expected inner product of unrelated vectors near zero), then
    L2-normalize so cosine similarity behaves the way callers expect.

    This is not semantically meaningful -- "timeout" and "deadline
    exceeded" hash to unrelated buckets despite meaning the same thing --
    but it is exactly reproducible with zero external services, which is
    the property tests and offline seeding need.
    """
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        # Degenerate input (e.g. a stack trace that's pure symbols):
        # deterministically hash the raw text instead of returning all-zeros,
        # since a zero vector has undefined cosine similarity.
        tokens = [text or "empty"]

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def generate_embedding(text: str) -> list[float]:
    """
    Produce an embedding for `text` using the configured backend.

    Async even in the deterministic-backend branch: callers (the RAG node,
    the seed script) never need to know which backend is active, and the
    signature stays stable if the deterministic path ever grows real I/O
    (e.g. a local model server) later.
    """
    settings = get_settings()
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    if settings.embedding_provider == "deterministic":
        return _embed_deterministic(text, settings.embedding_dim)

    return await _embed_ollama(text)


@lru_cache
def _warned_once() -> bool:
    """Sentinel so the deterministic-backend warning logs exactly once per process."""
    logger.warning(
        "embedding_provider='deterministic' is active -- similarity search "
        "results are lexical-hash based, not semantic. Set "
        "EMBEDDING_PROVIDER=ollama for real RAG behavior."
    )
    return True


def warn_if_deterministic() -> None:
    """Call once at startup so a misconfigured deployment is loud, not silent."""
    settings = get_settings()
    if settings.embedding_provider == "deterministic":
        _warned_once()
