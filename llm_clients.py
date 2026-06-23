import time
from google import genai
from google.genai import types, errors
from enrichment.extraction_schema import *
from pydantic import BaseModel

class LLMClient:
    def __init__(self):
        self.client = genai.Client()

    def generate_gemini(self, prompt: str, schema_type: type[BaseModel], max_retries: int = 5):
        delay = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model = "gemini-2.5-flash",
                    contents = prompt,
                    config = types.GenerateContentConfig(
                        response_mime_type = "application/json",
                        response_schema = schema_type
                    )
                )
                if response.text is None:
                    raise ValueError("LLM response does not contain text.")
                return response.text
            except (errors.ClientError, errors.ServerError) as e:
                if ("429" in str(e) or "503" in str(e)) and attempt < max_retries - 1:
                    print(f"Transient error ({str(e)[:40]}...). Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise