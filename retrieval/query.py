from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from .query_schema import QueryIntent
from prompts import QUERY_INTENT_PROMPT
from enrichment.resolver import resolve_entity
from .traversal import *

def query(kg: KnowledgeGarden, question: str, client: LLMClient):
    prompt = QUERY_INTENT_PROMPT.replace("{question}", question)
    response = client.generate_gemini(prompt, schema_type = QueryIntent)  
    query_intent = QueryIntent.model_validate_json(response)
    # anchor_id = resolve_entity(query_intent.anchor_entity, kg)
    anchor_id = query_intent.anchor_entity
    if anchor_id is None:
        return {
            "error": "entity not found", 
            "anchor_entity": query_intent.anchor_entity
        }
    target_id = None
    if query_intent.pattern == "path":
        if query_intent.target_entity is None:
            return {"error": "path query missing target_entity"}
        # target_id = resolve_entity(query_intent.target_entity, kg)
        target_id = query_intent.target_entity
        if target_id is None:
            return {"error": "entity not found", "anchor_entity": query_intent.target_entity}
    anchor_node = kg.nodes[anchor_id]
    if query_intent.pattern == "direct_lookup":
        if query_intent.relation is None:
            return {"error": "direct_lookup query missing relation"}
        direction = query_intent.direction or "out"
        return direct_lookup(kg, anchor_node, query_intent.relation, direction)
    elif query_intent.pattern == "neighborhood":
        return neighbor_expansion(kg, anchor_node)
    elif query_intent.pattern == "path":
        target_node = kg.nodes[target_id]
        return path_finding(kg, anchor_node, target_node)
    elif query_intent.pattern == "impact":
        if query_intent.relation is None:
            return {"error": "impact query missing relation"}
        direction = query_intent.direction or "in"
        return impact_traversal(kg, anchor_node, query_intent.relation, direction)
    elif query_intent.pattern == "history":
        if query_intent.relation is None:
            return {"error": "history query missing relation"}
        direction = query_intent.direction or "out"
        return history_traversal(kg, anchor_node, query_intent.relation, direction)
    else:
        return {"error": "unknown pattern", "pattern": query_intent.pattern}
    