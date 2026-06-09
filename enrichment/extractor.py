from llm_clients import LLMClient
from prompts import *

def extract_entities_and_relations(text: str, client: LLMClient):
    prompt = EXTRACTION_PROMPT.format(text = text)