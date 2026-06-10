from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from core.schema import *
from .extractor import extract_entities_and_relations
from .resolver import resolve_entity, normalize

def ingest_episode(raw_text: str, reference_time: datetime, client: LLMClient, kg: KnowledgeGarden):
    episode = Episode(raw_text, reference_time = reference_time)
    response = extract_entities_and_relations(raw_text, client)
    node_id_mapping = {}
    for node in response.nodes:
        existing_node_id = resolve_entity(node, kg)
        if existing_node_id is None:
            new_id = normalize(node.name).replace(" ", "_")
            kg.add_node(new_id, node.type, node.name)
            node_id_mapping[node.name] = new_id
        else:
            node_id_mapping[node.name] = existing_node_id
    for edge in response.edges:
        source_id = node_id_mapping.get(edge.source)
        target_id = node_id_mapping.get(edge.target)
        if source_id is None or target_id is None:
            print(f"Skipping edge with unmapped source or target: {edge}")
            continue
        kg.add_edge(source_id, target_id, edge.relation, edge.fact, reference_time)
    episode_id = kg.add_episode(episode)
    return episode_id
