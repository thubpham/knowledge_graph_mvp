from llm_clients import LLMClient
from prompts import *
from .extraction_schema import *

def extract_entities_and_relations(raw_text: str, client: LLMClient):
    prompt = EXTRACTION_PROMPT.replace("{text}", raw_text)
    response = client.generate_gemini(prompt)  
    return ExtractionResult.model_validate_json(response)