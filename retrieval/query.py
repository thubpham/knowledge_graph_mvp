from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from .query_schema import QueryIntent
from prompts import QUERY_INTENT_PROMPT
from .traversal import *
from .scoring import score_edge

def query(kg: KnowledgeGarden, question: str, client: LLMClient, now: datetime):
    prompt = QUERY_INTENT_PROMPT.replace("{question}", question)
    response = client.generate_gemini(prompt, schema_type = QueryIntent)  
    query_intent = QueryIntent.model_validate_json(response)
    anchor_id = query_intent.anchor_entity
    returned_edges = []
    if anchor_id not in kg.nodes:
        return {
            "error": "entity not found", 
            "anchor_entity": query_intent.anchor_entity
        }
    target_id = None
    if query_intent.pattern == "path":
        if query_intent.target_entity is None:
            return {"error": "path query missing target_entity"}
        if query_intent.target_entity not in kg.nodes:
            return {
                "error": "target entity not found", 
                "target_entity": query_intent.target_entity
            }
        target_id = query_intent.target_entity
    anchor_node = kg.nodes[anchor_id]
    if query_intent.pattern == "direct_lookup":
        if query_intent.relation is None:
            return {"error": "direct_lookup query missing relation"}
        direction = query_intent.direction or "out"
        returned_edges = direct_lookup(kg, anchor_node, query_intent.relation, direction)
    elif query_intent.pattern == "neighborhood":
        returned_edges = neighbor_expansion(kg, anchor_node)
    elif query_intent.pattern == "path":
        target_node = kg.nodes[target_id]
        returned_edges = path_finding(kg, anchor_node, target_node)
    elif query_intent.pattern == "impact":
        if query_intent.relation is None:
            return {"error": "impact query missing relation"}
        direction = query_intent.direction or "in"
        returned_edges = impact_traversal(kg, anchor_node, query_intent.relation, direction)
    elif query_intent.pattern == "history":
        if query_intent.relation is None:
            return {"error": "history query missing relation"}
        direction = query_intent.direction or "out"
        returned_edges = history_traversal(kg, anchor_node, query_intent.relation, direction)
        return returned_edges
    else:
        return {"error": "unknown pattern", "pattern": query_intent.pattern}
    
    valid_edges = []
    for edge in returned_edges:
        if (score_edge(edge, now) == 0.0):
            continue    
        valid_edges.append(edge)
    
    valid_edges.sort(key = lambda e: score_edge(e, now), reverse = True)
    return valid_edges 