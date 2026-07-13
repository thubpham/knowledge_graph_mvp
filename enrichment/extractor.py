from llm_clients import LLMClient
from prompts import *
from .extraction_schema import *

def extract_entities_and_relations(raw_text: str, client: LLMClient, known_entities: list[str] | None = None):
    known_entities_text = "\n".join(known_entities) if known_entities else "(none)"
    prompt = (
        EXTRACTION_PROMPT
        .replace("{known_entities}", known_entities_text)
        .replace("{text}", raw_text)
    )
    response = client.generate_gemini(prompt, schema_type = ExtractionResult)
    return ExtractionResult.model_validate_json(response)