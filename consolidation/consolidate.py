import json
import uuid
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from prompts import CONSOLIDATION_PROMPT
from enrichment.resolver import resolve_entity
from .episodic import get_episode_for_entity
from .consolidation_schema import ConsolidationResult


def consolidate(entity_id: str, kg: KnowledgeGarden, client: LLMClient):
    if not kg.node_exists(entity_id):
        raise ValueError(f"Node with id {entity_id} does not exist.")

    episodes = get_episode_for_entity(entity_id, kg)
    if not episodes:
        return None

    node = kg.get_node(entity_id)

    episodes_text = "\n".join(
        f'{i+1}. "{ep.text}"' for i, ep in enumerate(episodes)
    )

    prompt = (
        CONSOLIDATION_PROMPT
        .replace("{entity_name}", node.name)
        .replace("{episodes}", episodes_text)
    )

    raw = client.generate_gemini(prompt, schema_type=ConsolidationResult)
    result = ConsolidationResult(**json.loads(raw))

    run_id = str(uuid.uuid4())
    kg.update_node(entity_id,
        summary=result.summary,
        consolidated=True,
        consolidation_run_id=run_id,
    )

    edges_added = 0
    unresolved_edges = []

    for se in result.semantic_edges:
        target_id = resolve_entity(se.target, kg)
        if target_id is None:
            unresolved_edges.append({
                "source_id": entity_id,
                "relation": se.relation,
                "target": se.target,
                "fact": se.fact,
            })
            continue
        try:
            kg.add_edge(entity_id, target_id, se.relation, se.fact, datetime.now())
            edges_added += 1
        except ValueError:
            pass

    return {
        "entity_id": entity_id,
        "consolidation_run_id": run_id,
        "summary": result.summary,
        "edges_added": edges_added,
        "unresolved_edges": unresolved_edges,
        "episodic_only": result.episodic_only,
    }
