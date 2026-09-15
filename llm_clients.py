import json
import os
import time
import httpx
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

from trace import current_run

load_dotenv()

_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
_EMBED_DIM = 768

# Mirrors the enrichment/ingester.py `UNMAPPED_LOG_PATH` / enrichment/dedup.py
# `REVIEW_LOG_PATH` pattern: append-only JSONL of every failed attempt, not
# just the final outcome, since `_retry_validate()` below only ever persists
# the last attempt's `retries` count to the trace DB.
SCHEMA_FAILURE_LOG_PATH = Path(__file__).parent / ".local" / "schema_validation_failures.jsonl"


def _log_schema_failure(*, kind: str, provider: str, model: str | None,
                         schema_name: str, raw_text: str, error: str):
    SCHEMA_FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCHEMA_FAILURE_LOG_PATH.open("a") as f:
        f.write(json.dumps({
            "kind": kind,
            "provider": provider,
            "model": model,
            "schema": schema_name,
            "raw_text": raw_text,
            "error": error,
        }) + "\n")


@dataclass
class Usage:
    """Token counts for one LLM call. Fields are `None` when the provider/
    endpoint doesn't report that count (e.g. Gemini's embedding endpoint
    doesn't return usage at all). `cached_tokens`/`cache_write_tokens` are
    OpenRouter-specific (from `usage.prompt_tokens_details` in its response) —
    None on every other provider, not just when a call happens to miss cache."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None


class _TransientError(Exception):
    def __init__(self, status, body):
        self.status = status
        super().__init__(f"{status}: {body[:80]}")


def _retry(call, is_transient, max_retries: int = 5):
    """Runs `call()` with exponential backoff, retrying only on errors
    `is_transient` accepts. Returns `(call()'s result, attempt)` where
    `attempt` is the number of retries actually consumed (0 = succeeded on
    the first try). On final failure, stamps the exception with
    `_retry_attempts` before re-raising so the caller can still log how many
    retries were burned."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return call(), attempt
        except Exception as e:
            if is_transient(e) and attempt < max_retries - 1:
                print(f"Transient error ({str(e)[:60]}). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                e._retry_attempts = attempt
                raise


def _http_is_transient(e) -> bool:
    # ConnectError included for the ollama embedder: unlike the hosted
    # providers, "connection refused" is a real, recoverable case here (the
    # local server briefly restarting), not just a config error.
    return isinstance(e, (httpx.TimeoutException, httpx.RemoteProtocolError,
                          httpx.ConnectError, _TransientError))


def _usage_from_gemini(response) -> Usage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", None),
        completion_tokens=getattr(meta, "candidates_token_count", None),
        total_tokens=getattr(meta, "total_token_count", None),
    )


def _usage_from_json(data: dict) -> Usage:
    """Parses a `usage` block that may be shaped either OpenAI Chat-Completions-style
    (prompt_tokens/completion_tokens/total_tokens) or OpenAI Responses-API-style
    (input_tokens/output_tokens/total_tokens) — Concentrate's proxy shape hasn't
    been confirmed against a live response, so both are checked defensively.
    `prompt_tokens_details.cached_tokens`/`cache_write_tokens` is OpenRouter-only;
    every other provider's `usage` block simply lacks that key, so `.get()`
    leaves both `None` rather than misreporting a cache miss."""
    usage = data.get("usage")
    if not usage:
        return Usage()
    details = usage.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=usage.get("prompt_tokens", usage.get("input_tokens")),
        completion_tokens=usage.get("completion_tokens", usage.get("output_tokens")),
        total_tokens=usage.get("total_tokens"),
        cached_tokens=details.get("cached_tokens"),
        cache_write_tokens=details.get("cache_write_tokens"),
    )


class _GeminiProvider:
    """Talks to Gemini directly via GEMINI_API_KEY. No per-token billing surprises tied to a third-party proxy."""

    name = "gemini"

    def __init__(self):
        from google import genai
        from google.genai import types, errors
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        self._types = types
        self._errors = errors
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        config_kwargs: dict = {"system_instruction": system}
        if schema_type is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema_type
            # Large source documents can produce large structured extractions;
            # the default output cap truncates mid-JSON on those, which then fails
            # to parse. Gemini 2.5 Flash supports up to 65536 output tokens.
            config_kwargs["max_output_tokens"] = 65536
        config = self._types.GenerateContentConfig(**config_kwargs)

        def call():
            response = self.client.models.generate_content(
                model=self.model, contents=user, config=config
            )
            if response.text is None:
                raise ValueError("LLM response does not contain text.")
            return response.text, _usage_from_gemini(response)

        def is_transient(e):
            return isinstance(e, (self._errors.ClientError, self._errors.ServerError)) and (
                "429" in str(e) or "503" in str(e)
            )

        (text, usage), retries = _retry(call, is_transient, max_retries)
        return text, usage, retries


class _ConcentrateProvider:
    """Talks to Concentrate AI's OpenAI-responses-style proxy via CONCENTRATE_AI_API_KEY."""

    name = "concentrate"
    _BASE_URL = "https://api.concentrate.ai/v1/responses"

    def __init__(self):
        self.api_key = os.getenv("CONCENTRATE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("CONCENTRATE_AI_API_KEY not set in environment.")
        self.model = os.getenv("CONCENTRATE_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {"model": self.model, "instructions": system, "input": user}
        if schema_type is not None:
            schema = schema_type.model_json_schema()
            _patch_schema(schema)  # Concentrate requires additionalProperties: false at every object level
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_type.__name__.lower(),
                    "schema": schema,
                }
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def call():
            resp = httpx.post(self._BASE_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            return data["output"][0]["content"][0]["text"], _usage_from_json(data)

        (text, usage), retries = _retry(call, _http_is_transient, max_retries)
        return text, usage, retries


class _OpenAIProvider:
    """Talks to OpenAI's Chat Completions API directly via OPENAI_API_KEY."""

    name = "openai"
    _BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema_type is not None:
            schema = schema_type.model_json_schema()
            _patch_schema(schema)  # OpenAI structured outputs require additionalProperties: false at every object level
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_type.__name__.lower(),
                    "schema": schema,
                    "strict": True,
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def call():
            resp = httpx.post(self._BASE_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], _usage_from_json(data)

        (text, usage), retries = _retry(call, _http_is_transient, max_retries)
        return text, usage, retries


class _AnthropicProvider:
    """Talks to Claude directly via the official `anthropic` SDK and
    ANTHROPIC_API_KEY. `output_config.format` constrains decoding the same
    way Gemini's `response_schema` does (guaranteed-valid JSON, not just
    instructed), so this provider belongs in _NATIVE_SCHEMA_PROVIDERS below
    alongside Gemini, not in the validate-and-retry group with
    OpenAI/Groq/Concentrate."""

    name = "anthropic"

    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment.")
        # Extraction is high call volume relative to its per-call accuracy
        # bar -> Haiku 4.5 by default (see the cost estimate in this
        # session's IMPROVEMENTS.md-adjacent discussion). Override to
        # claude-sonnet-5 via ANTHROPIC_MODEL for lower-volume,
        # accuracy-sensitive tasks (consolidation/query) without touching
        # this file.
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._anthropic = anthropic
        # This module already owns retry/backoff via _retry() below, the
        # same as every other provider here — disable the SDK's own retries
        # so a transient error isn't retried twice on two different
        # schedules.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        kwargs = {
            "model": self.model,
            "system": system,
            # Same failure mode _GeminiProvider guards against with
            # max_output_tokens=65536: a large structured extraction that runs
            # past the cap is cut off mid-JSON. Traced extraction completions
            # already reach ~2.9k tokens on a single ~4k-token episode, and the
            # ratio of completion to prompt runs 0.35-0.46, so a long source
            # (Google Doc, Claude session transcript) would blow through 8192.
            # Not higher than 16384: the SDK refuses any non-streaming request
            # it estimates could exceed 10 minutes, and 32768 raises
            # ValueError("Streaming is required...") before the call is even
            # sent (verified against anthropic 0.120.0). Going beyond this
            # means switching to client.messages.stream() +
            # get_final_message(), which isn't worth it at current sizes.
            "max_tokens": 16384,
            "messages": [{"role": "user", "content": user}],
        }
        if schema_type is not None:
            schema = schema_type.model_json_schema()
            _patch_schema(schema)  # same additionalProperties:false requirement as OpenAI/Groq/Concentrate
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        def call():
            response = self.client.messages.create(**kwargs)
            # Fail loudly on a truncated response instead of letting the caller
            # hit it as a parse error. This provider is in
            # _NATIVE_SCHEMA_PROVIDERS, so generate_gemini() skips the
            # _retry_validate() wrapper -- truncated JSON goes straight to
            # model_validate_json() and surfaces as a ValidationError that says
            # nothing about the output cap. _retry()'s is_transient below
            # rejects ValueError, so this aborts immediately rather than
            # burning five identical retries on a deterministic failure.
            if response.stop_reason == "max_tokens":
                raise ValueError(
                    f"Response hit the {kwargs['max_tokens']}-token output cap "
                    f"(model={self.model}, prompt_tokens={response.usage.input_tokens}); "
                    "output is truncated. Reduce input size (see "
                    "enrichment/chunking.py) or raise max_tokens via streaming."
                )
            text = next(b.text for b in response.content if b.type == "text")
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            return text, usage

        def is_transient(e):
            anthropic = self._anthropic
            if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError)):
                return True
            return isinstance(e, anthropic.APIStatusError) and e.status_code >= 500

        (text, usage), retries = _retry(call, is_transient, max_retries)
        return text, usage, retries


class _GroqProvider:
    """Talks to Groq's OpenAI-compatible Chat Completions API via GROQ_API_KEY. Free tier,
    used to escape Gemini's 20 generateContent-calls/day free-tier wall."""

    name = "groq"
    _BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment.")
        # Not llama-3.3-70b-versatile: Groq only supports strict json_schema
        # structured outputs (what generate() sends below) on its gpt-oss
        # models — https://console.groq.com/docs/structured-outputs#supported-models.
        # 120b (not the smaller/cheaper 20b) to stay close to the accuracy
        # bar the dense 70b model this replaces was originally picked for.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema_type is not None:
            schema = schema_type.model_json_schema()
            _patch_schema(schema)  # same OpenAI-style strict json_schema requirements
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_type.__name__.lower(),
                    "schema": schema,
                    "strict": True,
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def call():
            resp = httpx.post(self._BASE_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], _usage_from_json(data)

        (text, usage), retries = _retry(call, _http_is_transient, max_retries)
        return text, usage, retries


class _OllamaProvider:
    """Talks to a local Ollama server's OpenAI-compatible endpoint. No API key,
    no per-token cost, no rate limit — used for the highest-call-volume,
    lowest-individual-stakes calls (entity-match/dedup confirmation), where a
    wrong local call just falls back to "no match" and is cheap to recover
    from downstream. Uses JSON-object mode rather than strict json_schema
    mode: Ollama's OpenAI-compat layer doesn't enforce schemas the way
    Gemini/Groq do, so correctness relies on the prompt already spelling out
    the expected keys (true for every prompt this provider is used with, e.g.
    `ENTITY_MATCH_SYSTEM_PROMPT`/`ENTITY_MATCH_USER_PROMPT`) plus
    `LLMClient.generate_gemini`'s validate-and-retry wrapper for
    non-native-schema providers."""

    name = "ollama"
    _BASE_URL = "http://localhost:11434/v1/chat/completions"

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if schema_type is not None:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}

        def call():
            resp = httpx.post(self._BASE_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], _usage_from_json(data)

        (text, usage), retries = _retry(call, _http_is_transient, max_retries)
        return text, usage, retries


class _OpenRouterProvider:
    """Talks to OpenRouter's OpenAI-compatible Chat Completions API via
    OPENROUTER_API_KEY. OpenRouter is an aggregator, not a model host — every
    call is proxied to whichever upstream provider it routes to, so two
    things this class does matter beyond the usual payload shape:

    - `provider.sort=latency` + `provider.require_parameters=True`: picks the
      upstream with the lowest end-to-end response time that actually
      supports every parameter in this request (structured outputs
      included), rather than one that would silently drop `response_format`
      and fail schema compliance downstream. NOT `sort=price` (the first
      live ingest run, 2026-09-11, hit a single `extract_entities` call that
      ran past 120s and into `_retry()`'s backoff loop, still unresolved
      after 5+ minutes -- the cheapest upstream for a given model is not
      necessarily a fast one) and NOT `sort=throughput` either: throughput
      is tokens/sec once a response is already streaming, which doesn't
      target the failure actually observed -- the call never started
      responding at all. `latency` is the OpenRouter-reported end-to-end
      response time, the closer match. Still only a ranking preference among
      providers that pass `require_parameters`, not a hard exclusion filter
      -- `preferred_max_latency` (a percentile cutoff) is the stronger tool
      if `sort=latency` alone doesn't hold once volume ramps up.
    - `session_id` (from the active `Run`, see trace.py): OpenRouter's sticky
      routing key. Passing it keeps repeated calls within one run pinned to
      the same upstream provider so its prompt cache (keyed on this class's
      byte-identical `system` message) actually gets reused instead of
      landing on a different provider each time. Deliberately NOT setting
      `provider.order` — an explicit order list disables load balancing
      (and, per OpenRouter's docs, is a distinct knob from sticky routing;
      the two aren't meant to be combined here). Omitted entirely when no
      `Run` is active (ad-hoc scripts, notebooks) rather than sent as `None`,
      since OpenRouter treats presence of the field as the activation signal.
    """

    name = "openrouter"
    _BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment.")
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

    def generate(self, system: str, user: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "provider": {"sort": "latency", "require_parameters": True},
        }
        run = current_run()
        if run is not None:
            payload["session_id"] = run.run_id
        if schema_type is not None:
            schema = schema_type.model_json_schema()
            _patch_schema(schema)  # same OpenAI-style strict json_schema requirements
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_type.__name__.lower(),
                    "schema": schema,
                    "strict": True,
                },
            }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        def call():
            resp = httpx.post(self._BASE_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], _usage_from_json(data)

        (text, usage), retries = _retry(call, _http_is_transient, max_retries)
        return text, usage, retries


_PROVIDERS = {
    "gemini": _GeminiProvider,
    "concentrate": _ConcentrateProvider,
    "openai": _OpenAIProvider,
    "groq": _GroqProvider,
    "ollama": _OllamaProvider,
    "anthropic": _AnthropicProvider,
    "openrouter": _OpenRouterProvider,
}

# Providers whose API natively enforces the JSON schema at decode time
# (Gemini's response_schema and Anthropic's output_config.format are both
# constrained decoding — the model literally cannot emit invalid-shape
# JSON). Providers not in this set are only "instructed to comply" via
# prompt + json_schema/json_object mode, so their output is validated and
# retried by generate_gemini() below.
_NATIVE_SCHEMA_PROVIDERS = {"gemini", "anthropic"}


def _retry_validate(call, schema_type: type[BaseModel], max_retries: int = 5,
                     *, kind: str = "generate_gemini", provider: str = None, model: str = None):
    """Runs `call()` (expected to return `(text, Usage, retries)`) and
    validates the returned text against `schema_type`, re-invoking `call()`
    entirely (a fresh generation, not just a re-parse) if validation fails.
    Used for providers whose schema enforcement isn't native, since a
    malformed-JSON response from those is a real, expected failure mode
    rather than an edge case."""
    last_exc = None
    for attempt in range(max_retries):
        text, usage, retries = call()
        try:
            schema_type.model_validate_json(text)
            return text, usage, retries
        except (ValidationError, json.JSONDecodeError) as e:
            last_exc = e
            _log_schema_failure(
                kind=kind, provider=provider, model=model,
                schema_name=schema_type.__name__, raw_text=text, error=str(e),
            )
            print(
                f"Schema validation failed ({str(e)[:80]}). "
                f"Retrying generation... (attempt {attempt + 1}/{max_retries})"
            )
    raise last_exc


class _GeminiEmbedder:
    """The original, only-ever embedder. Kept selectable via
    EMBEDDING_PROVIDER=gemini for A/B comparison against the ollama path, and
    as the default so scripts that construct LLMClient without touching the
    new env var keep working unchanged."""

    name = "gemini"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        self.api_key = api_key
        self.model = "gemini-embedding-001"

    def embed(self, text: str, max_retries: int = 5) -> tuple[list[float], Usage, int]:
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "task_type": "SEMANTIC_SIMILARITY",
            "output_dimensionality": _EMBED_DIM,
        }
        params = {"key": self.api_key}

        def call():
            resp = httpx.post(_EMBED_URL, json=payload, params=params, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            # The embedding endpoint doesn't return a usage block.
            return data["embedding"]["values"], Usage()

        (values, usage), retries = _retry(call, _http_is_transient, max_retries)
        return values, usage, retries


class _OllamaEmbedder:
    """Local embeddings via Ollama's /api/embed. No API key, no rate limit,
    no per-token cost -- the point of this class existing is that Gemini's
    free-tier embed quota (1000 requests/day, see llm_clients module history)
    made a full ingest take days once resolve_entity started embedding most
    extracted entities. nomic-embed-text is the default: it outputs 768
    dimensions, matching core.graph's vector index exactly (no index
    migration needed), and benchmarked ~40x faster per text than the other
    locally-pulled option (a 0.6B Qwen embedding model) at batch size 60 --
    roughly one Notion page's worth of entities."""

    name = "ollama"
    _URL = "http://localhost:11434/api/embed"

    def __init__(self):
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def embed(self, text: str, max_retries: int = 5) -> tuple[list[float], Usage, int]:
        payload = {"model": self.model, "input": text}

        def call():
            resp = httpx.post(self._URL, json=payload, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            # /api/embed's response shape is {"embeddings": [[...]]} even for
            # a single string input -- always take the first (only) vector.
            return data["embeddings"][0], Usage()

        (values, usage), retries = _retry(call, _http_is_transient, max_retries)
        return values, usage, retries


_EMBED_PROVIDERS = {"gemini": _GeminiEmbedder, "ollama": _OllamaEmbedder}


class LLMClient:
    """
    Generation provider is chosen by the LLM_PROVIDER env var ("gemini" | "concentrate" |
    "openai" | "groq" | "ollama" | "anthropic" | "openrouter"), defaulting to "gemini". Swap
    providers (e.g. when one runs out of credit) by changing that env var — no code changes needed.

    Embedding provider is chosen independently by EMBEDDING_PROVIDER ("gemini" | "ollama"),
    defaulting to "gemini" for backward compatibility. Set it separately from LLM_PROVIDER --
    generation and embedding are unrelated axes; e.g. EXTRACTION_LLM_PROVIDER=anthropic +
    EMBEDDING_PROVIDER=ollama runs extraction on Haiku and embeddings on a local model in the
    same process.
    """

    def __init__(self, provider: str | None = None):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        if provider_name not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider_name}'. Choose from: {', '.join(_PROVIDERS)}"
            )
        self._provider = _PROVIDERS[provider_name]()

        embed_provider_name = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
        if embed_provider_name not in _EMBED_PROVIDERS:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{embed_provider_name}'. "
                f"Choose from: {', '.join(_EMBED_PROVIDERS)}"
            )
        self._embed_provider = _EMBED_PROVIDERS[embed_provider_name]()

        # Full-string cache, keyed on the exact text passed to embed() (i.e.
        # embedding_text()'s "name: context" output, not just the bare name --
        # see resolver.py for why context matters for disambiguation). Scoped
        # to this LLMClient instance, so it lives exactly as long as one
        # ingest process and never serves a stale entry across runs. Measured
        # against a real trace: ~26-32% hit rate depending on run size, mostly
        # from resolve_entity's embed() and ingester.py's node-creation
        # embed() re-embedding the identical string for every newly-created
        # node within the same run.
        self._embed_cache: dict[str, list[float]] = {}

    def generate_gemini(self, system: str, user: str, schema_type: type[BaseModel], max_retries: int = 5,
                         *, kind: str = "generate_gemini") -> str:
        def call():
            return self._provider.generate(system, user, schema_type=schema_type, max_retries=max_retries)

        if self._provider.name in _NATIVE_SCHEMA_PROVIDERS:
            wrapped_call = call
        else:
            wrapped_call = lambda: _retry_validate(
                call, schema_type, max_retries, kind=kind,
                provider=self._provider.name, model=getattr(self._provider, "model", None),
            )

        return self._traced_call(kind=kind, system=system, user=user, call=wrapped_call)

    def generate_text(self, system: str, user: str, max_retries: int = 5) -> str:
        return self._traced_call(
            kind="generate_text", system=system, user=user,
            call=lambda: self._provider.generate(system, user, max_retries=max_retries),
        )

    def _traced_call(self, *, kind: str, user: str, call, system: str | None = None,
                      provider: str = None, model: str = None, response_summary=None):
        """Runs `call()` (expected to return `(text, Usage, retries)`), logging
        it to the current `Run` (see trace.py) if one is active. A no-op
        wrapper — same return value, same exceptions — when called outside a
        `Run` block, so untraced use (notebooks, ad-hoc scripts) is unaffected.
        `provider`/`model` default to the configured generation provider;
        `embed()` overrides them with the configured embedding provider,
        which is chosen independently via `EMBEDDING_PROVIDER`. `system` is
        `None` for `embed()` (there is no system/user split for embeddings)
        and is logged separately from `user` so a trace can show whether the
        static half of a prompt was actually byte-identical across calls —
        see trace.py's `system_prompt` column."""
        run = current_run()
        provider = provider or self._provider.name
        model = model or getattr(self._provider, "model", None)
        call_id = (
            run.start_llm_call(kind=kind, provider=provider, model=model, prompt=user, system_prompt=system)
            if run else None
        )
        start = time.monotonic()
        try:
            text, usage, retries = call()
        except Exception as e:
            if run is not None:
                run.finish_llm_call(
                    call_id, response=None, error=str(e), usage=None,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    retries=getattr(e, "_retry_attempts", None),
                )
            raise
        if run is not None:
            run.finish_llm_call(
                call_id,
                response=response_summary(text) if response_summary else text,
                error=None, usage=usage,
                latency_ms=int((time.monotonic() - start) * 1000),
                retries=retries,
            )
        return text

    def embed(self, text: str, max_retries: int = 5) -> list[float]:
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached

        values = self._traced_call(
            kind="embed", user=text,
            call=lambda: self._embed_provider.embed(text, max_retries=max_retries),
            provider=self._embed_provider.name, model=self._embed_provider.model,
            response_summary=lambda v: f"<embedding dim={len(v)}>",
        )
        self._embed_cache[text] = values
        return values


def _patch_schema(schema: dict):
    # OpenAI-style strict json_schema mode (used by both the openai and
    # concentrate providers) requires every property to be listed in
    # "required" — optionality is expressed via an "anyOf"-with-null type,
    # not by omitting the key — and rejects "default" entirely.
    schema.pop("default", None)
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        properties = schema.get("properties", {})
        if properties:
            schema["required"] = list(properties.keys())
        for prop in properties.values():
            _patch_schema(prop)
    if "items" in schema:
        _patch_schema(schema["items"])
    for sub in schema.get("anyOf", []):
        _patch_schema(sub)
    for sub in schema.get("$defs", {}).values():
        _patch_schema(sub)
