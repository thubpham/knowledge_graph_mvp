from llm_clients import LLMClient
from prompts import *
from .extraction_schema import *


def extract_entities(raw_text: str, client: LLMClient, known_entities: list[str] | None = None) -> EntityExtractionResult:
    known_entities_text = "\n".join(known_entities) if known_entities else "(none)"
    user_prompt = (
        ENTITY_EXTRACTION_USER_PROMPT
        .replace("{known_entities}", known_entities_text)
        .replace("{text}", raw_text)
    )
    response = client.generate_gemini(
        ENTITY_EXTRACTION_SYSTEM_PROMPT, user_prompt,
        schema_type=EntityExtractionResult, kind="extract_entities",
    )
    return EntityExtractionResult.model_validate_json(response)


def extract_relations(raw_text: str, entities: list[ExtractedNode], client: LLMClient) -> RelationExtractionResult:
    entities_text = "\n".join(f"{n.name} ({n.type})" for n in entities) if entities else "(none)"
    user_prompt = (
        RELATION_EXTRACTION_USER_PROMPT
        .replace("{entities}", entities_text)
        .replace("{text}", raw_text)
    )
    response = client.generate_gemini(
        RELATION_EXTRACTION_SYSTEM_PROMPT, user_prompt,
        schema_type=RelationExtractionResult, kind="extract_relations",
    )
    return RelationExtractionResult.model_validate_json(response)


def extract_entities_and_relations(raw_text: str, client: LLMClient, known_entities: list[str] | None = None) -> ExtractionResult:
    """Two-pass extraction: entities first, then relations constrained to just those
    entities. Splitting the task keeps each call simpler (and each schema smaller),
    which is the actual accuracy lever — relation extraction never has to simultaneously
    invent an entity and connect it. Returns the same ExtractionResult shape the single-call
    version used to, so callers (enrichment/ingester.py) don't need to change."""
    entity_result = extract_entities(raw_text, client, known_entities=known_entities)
    relation_result = extract_relations(raw_text, entity_result.nodes, client)
    return ExtractionResult(
        nodes=entity_result.nodes,
        edges=relation_result.edges,
        unmapped_entities=entity_result.unmapped_entities,
        unmapped_relations=relation_result.unmapped_relations,
    )