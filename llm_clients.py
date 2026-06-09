from google import genai
from google.genai import types
from enrichment.extraction_schema import *

class LLMClient:
    def __init__(self):
        self.client = genai.Client()

    def generate_gemini(self, prompt: str):
        response = self.client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt,
            config = types.GenerateContentConfig(
                response_mime_type = "application/json",
                response_schema = ExtractionResult
            )
        )
        if response.text is None:
            raise ValueError("LLM response does not contain text.")
        return response.text