"""
LLM reasoning backend: structured-output calls used by the graph's agent
nodes (app/graph.py).

Two backends behind one interface (`call_structured`), selected by
`settings.llm_provider`:

* **ollama** (default) -- a local model via Ollama's `/api/chat`, using its
  `format` parameter to pass a JSON Schema. Ollama compiles that schema
  into a grammar that constrains token-level decoding: the model is
  structurally incapable of emitting a token that would violate the
  schema (wrong type, a value outside an `enum`, an extra key under
  `additionalProperties: false`). That is a stronger structural guarantee
  than prompting a larger hosted model and hoping it complies.
* **anthropic** -- Claude via the official `anthropic` SDK's
  `messages.parse()`, which validates the response against a Pydantic
  model server-side and returns `response.parsed_output` already typed.

Both paths return a validated instance of the caller's Pydantic schema, so
`app/graph.py`'s nodes never branch on which backend answered them.
"""
import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings

logger = logging.getLogger("triage_engine.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for all LLM-call failures in the triage pipeline."""


class LLMTimeoutError(LLMError):
    """The configured LLM backend did not respond within llm_timeout_seconds."""


class LLMMalformedOutputError(LLMError):
    """The backend responded, but its output could not be validated against the schema."""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def _pydantic_to_ollama_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to the plain JSON Schema dict Ollama's
    `format` parameter expects.

    Pydantic v2 externalizes nested submodels (and, in general, Enum
    fields) into `$defs` + `$ref`. Ollama's grammar compiler wants a
    self-contained schema. The node-output schemas this is called with
    (`LogInspectorOutput` / `TriageOutput` in app/graph.py) are flat --
    str/int/float/list[str]/Literal fields only, using `Literal[...]`
    instead of an Enum class for exactly this reason -- so this is a
    no-op today. The guard below turns a future nested field into a loud
    error at call time instead of a silent 400 from Ollama.
    """
    schema = model.model_json_schema()
    if "$defs" in schema:
        raise LLMError(
            f"{model.__name__} has nested submodels/enums ($defs present); "
            "inline them (e.g. use typing.Literal instead of an Enum class) "
            "before using this schema with the Ollama structured-output path."
        )
    schema["additionalProperties"] = False
    return schema


async def _call_ollama_structured(*, system: str, user: str, schema_model: type[T]) -> T:
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": _pydantic_to_ollama_schema(schema_model),
        "stream": False,
        # Low temperature: this is extraction/classification, not creative
        # writing -- the same trace should route the same way every time.
        "options": {"temperature": 0.1},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError(
            f"Ollama chat request timed out after {settings.llm_timeout_seconds}s "
            f"(model={settings.ollama_chat_model}). A 7B model can be slow on CPU-only "
            "hardware; raise LLM_TIMEOUT_SECONDS or use a smaller model."
        ) from exc
    except httpx.ConnectError as exc:
        raise LLMTimeoutError(
            f"Could not reach Ollama at {settings.ollama_base_url}. "
            "Start it with `ollama serve`, and ensure the model is pulled: "
            f"`ollama pull {settings.ollama_chat_model}`."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise LLMError(
            f"Ollama chat request failed with {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc

    body = response.json()
    try:
        content = body["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LLMMalformedOutputError(f"Unexpected Ollama response shape: {body!r}") from exc

    try:
        return schema_model.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise LLMMalformedOutputError(
            f"Ollama returned output that failed {schema_model.__name__} validation: "
            f"{content[:500]!r} ({exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

_anthropic_client = None  # lazily constructed; module-level so it's reused (pooled connections)


def _get_anthropic_client():
    """
    Lazily construct and cache the AsyncAnthropic client.

    Lazy for two reasons: (1) the `anthropic` package is only imported
    here, not at module load, so an ollama-only deployment never pays the
    import cost or needs the package present; (2) client construction
    reads credentials (env var / `ant auth login` profile), which -- like
    the DB engine in app/database.py -- should happen on first real use,
    not at import time.
    """
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    try:
        import anthropic
    except ImportError as exc:
        raise LLMError(
            "llm_provider='anthropic' requires the `anthropic` package. Install it with "
            "`pip install anthropic` (it is already listed in requirements.txt)."
        ) from exc

    settings = get_settings()
    # Explicit api_key only if one was configured; otherwise let the SDK
    # resolve credentials itself (ANTHROPIC_API_KEY env var, or an
    # `ant auth login` profile) -- see the Anthropic API skill's
    # Authentication section for the full resolution order.
    _anthropic_client = (
        anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        if settings.anthropic_api_key
        else anthropic.AsyncAnthropic()
    )
    return _anthropic_client


async def _call_anthropic_structured(*, system: str, user: str, schema_model: type[T]) -> T:
    import anthropic  # local import mirrors _get_anthropic_client's laziness

    settings = get_settings()
    client = _get_anthropic_client()

    try:
        # with_options(timeout=...) uses the SDK's own request-timeout
        # machinery rather than wrapping the call in asyncio.wait_for,
        # so retry/connection-pool behavior stays exactly what the SDK
        # authors intended.
        response = await client.with_options(timeout=settings.llm_timeout_seconds).messages.parse(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema_model,
        )
    # Most-specific-first exception chain (see the Anthropic API skill's
    # error-handling guidance) so timeouts, rate limits, and generic API
    # errors are distinguishable to the caller/retry logic below.
    except anthropic.APITimeoutError as exc:
        raise LLMTimeoutError(
            f"Anthropic request timed out after {settings.llm_timeout_seconds}s"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError(f"Anthropic rate limit exceeded: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMTimeoutError(f"Could not reach the Anthropic API: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc

    if response.stop_reason == "refusal":
        raise LLMMalformedOutputError(
            f"Anthropic declined the request (stop_details={response.stop_details})"
        )
    if response.parsed_output is None:
        raise LLMMalformedOutputError(
            f"Anthropic response did not include a valid {schema_model.__name__} "
            f"(stop_reason={response.stop_reason})"
        )
    return response.parsed_output


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

_BACKENDS = {
    "ollama": _call_ollama_structured,
    "anthropic": _call_anthropic_structured,
}


async def call_structured(*, system: str, user: str, schema_model: type[T], node_name: str) -> T:
    """
    Call the configured LLM backend and return a validated `schema_model`
    instance.

    Retries up to `settings.llm_max_retries` additional times on a
    malformed-output failure only -- a decode/validation glitch is often
    transient (a smaller model occasionally drops a required field even
    under grammar-constrained decoding). A *timeout* is not retried here:
    retrying an already-slow or unreachable backend just compounds the
    latency the caller is waiting on; the node that called this function
    is expected to catch `LLMError` and fall back to its deterministic
    rule-based path (see log_inspector_node / triage_router_node in
    app/graph.py) rather than hang the request.
    """
    settings = get_settings()
    backend = _BACKENDS[settings.llm_provider]

    last_error: Exception | None = None
    attempts = settings.llm_max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            result = await backend(system=system, user=user, schema_model=schema_model)
            if attempt > 1:
                logger.info("%s: LLM call succeeded on attempt %d/%d", node_name, attempt, attempts)
            return result
        except LLMTimeoutError:
            raise
        except LLMError as exc:
            last_error = exc
            logger.warning(
                "%s: LLM call attempt %d/%d failed (%s): %s",
                node_name, attempt, attempts, settings.llm_provider, exc,
            )

    raise LLMMalformedOutputError(
        f"{node_name}: exhausted {attempts} attempt(s) against '{settings.llm_provider}'"
    ) from last_error
