import os
import time
import json
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://api.concentrate.ai/v1/responses"
_MODEL = "claude-haiku-4-5-20251001"


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("CONCENTRATE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("CONCENTRATE_AI_API_KEY not set in environment.")

    def generate_gemini(self, prompt: str, schema_type: type[BaseModel], max_retries: int = 5):
        schema = schema_type.model_json_schema()
        # Concentrate requires additionalProperties: false at every object level
        _patch_schema(schema)

        payload = {
            "model": _MODEL,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_type.__name__.lower(),
                    "schema": schema,
                }
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        delay = 5
        for attempt in range(max_retries):
            try:
                resp = httpx.post(_BASE_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code in (429, 503):
                    raise _TransientError(resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
                # Extract text content from normalized response
                content = data["output"][0]["content"][0]["text"]
                return content
            except _TransientError as e:
                if attempt < max_retries - 1:
                    print(f"Transient error ({e.status}). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise


class _TransientError(Exception):
    def __init__(self, status, body):
        self.status = status
        super().__init__(f"{status}: {body[:80]}")


def _patch_schema(schema: dict):
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            _patch_schema(prop)
    if "items" in schema:
        _patch_schema(schema["items"])
    for sub in schema.get("$defs", {}).values():
        _patch_schema(sub)
