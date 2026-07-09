import os
import time
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
_EMBED_DIM = 768


class _TransientError(Exception):
    def __init__(self, status, body):
        self.status = status
        super().__init__(f"{status}: {body[:80]}")


def _retry(call, is_transient, max_retries: int = 5):
    """Runs `call()` with exponential backoff, retrying only on errors `is_transient` accepts."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as e:
            if is_transient(e) and attempt < max_retries - 1:
                print(f"Transient error ({str(e)[:60]}). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def _http_is_transient(e) -> bool:
    return isinstance(e, (httpx.TimeoutException, httpx.RemoteProtocolError, _TransientError))


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

    def generate(self, prompt: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        config = None
        if schema_type is not None:
            config = self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_type,
                # Large source documents can produce large structured extractions;
                # the default output cap truncates mid-JSON on those, which then fails
                # to parse. Gemini 2.5 Flash supports up to 65536 output tokens.
                max_output_tokens=65536,
            )

        def call():
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            if response.text is None:
                raise ValueError("LLM response does not contain text.")
            return response.text

        def is_transient(e):
            return isinstance(e, (self._errors.ClientError, self._errors.ServerError)) and (
                "429" in str(e) or "503" in str(e)
            )

        return _retry(call, is_transient, max_retries)


class _ConcentrateProvider:
    """Talks to Concentrate AI's OpenAI-responses-style proxy via CONCENTRATE_AI_API_KEY."""

    name = "concentrate"
    _BASE_URL = "https://api.concentrate.ai/v1/responses"

    def __init__(self):
        self.api_key = os.getenv("CONCENTRATE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("CONCENTRATE_AI_API_KEY not set in environment.")
        self.model = os.getenv("CONCENTRATE_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, prompt: str, schema_type: type[BaseModel] | None = None, max_retries: int = 5) -> str:
        payload = {"model": self.model, "input": prompt}
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
            return data["output"][0]["content"][0]["text"]

        return _retry(call, _http_is_transient, max_retries)


_PROVIDERS = {"gemini": _GeminiProvider, "concentrate": _ConcentrateProvider}


class LLMClient:
    """
    Generation provider is chosen by the LLM_PROVIDER env var ("gemini" | "concentrate"),
    defaulting to "gemini". Swap providers (e.g. when one runs out of credit) by changing
    that env var — no code changes needed. Embeddings always go straight to Gemini
    regardless of provider, since that's the only embedding source wired up.
    """

    def __init__(self, provider: str | None = None):
        provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        if provider_name not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider_name}'. Choose from: {', '.join(_PROVIDERS)}"
            )
        self._provider = _PROVIDERS[provider_name]()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    def generate_gemini(self, prompt: str, schema_type: type[BaseModel], max_retries: int = 5) -> str:
        return self._provider.generate(prompt, schema_type=schema_type, max_retries=max_retries)

    def generate_text(self, prompt: str, max_retries: int = 5) -> str:
        return self._provider.generate(prompt, max_retries=max_retries)

    def embed(self, text: str, max_retries: int = 5) -> list[float]:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")

        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "task_type": "SEMANTIC_SIMILARITY",
            "output_dimensionality": _EMBED_DIM,
        }
        params = {"key": self.gemini_api_key}

        def call():
            resp = httpx.post(_EMBED_URL, json=payload, params=params, timeout=120)
            if resp.status_code in (429, 503):
                raise _TransientError(resp.status_code, resp.text)
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]

        return _retry(call, _http_is_transient, max_retries)


def _patch_schema(schema: dict):
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            _patch_schema(prop)
    if "items" in schema:
        _patch_schema(schema["items"])
    for sub in schema.get("$defs", {}).values():
        _patch_schema(sub)
