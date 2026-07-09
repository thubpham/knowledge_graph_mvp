import json
from pathlib import Path

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from core.schema import *
from .extractor import extract_entities_and_relations
from .resolver import resolve_entity, normalize
from .chunking import chunk_text
from .extraction_schema import ExtractionResult

UNMAPPED_LOG_PATH = Path(__file__).parent.parent / ".local" / "unmapped_log.jsonl"


def _merge_extractions(results: list[ExtractionResult]) -> ExtractionResult:
    nodes_by_key = {}
    edges_by_key = {}
    unmapped_entities = []
    unmapped_relations = []

    for result in results:
        for node in result.nodes:
            nodes_by_key.setdefault((node.name, node.type), node)
        for edge in result.edges:
            edges_by_key.setdefault((edge.source, edge.target, edge.relation), edge)
        unmapped_entities.extend(result.unmapped_entities)
        unmapped_relations.extend(result.unmapped_relations)

    return ExtractionResult(
        nodes=list(nodes_by_key.values()),
        edges=list(edges_by_key.values()),
        unmapped_entities=unmapped_entities,
        unmapped_relations=unmapped_relations,
    )


def _log_unmapped(response: ExtractionResult, episode_id: str, reference_time: datetime):
    if not response.unmapped_entities and not response.unmapped_relations:
        return
    with UNMAPPED_LOG_PATH.open("a") as f:
        for entity in response.unmapped_entities:
            f.write(json.dumps({
                "kind": "entity",
                "episode_id": episode_id,
                "reference_time": reference_time.isoformat(),
                **entity.model_dump(),
            }) + "\n")
        for relation in response.unmapped_relations:
            f.write(json.dumps({
                "kind": "relation",
                "episode_id": episode_id,
                "reference_time": reference_time.isoformat(),
                **relation.model_dump(),
            }) + "\n")


def ingest_episode(raw_text: str, reference_time: datetime, client: LLMClient, kg: KnowledgeGarden):
    episode = Episode(raw_text, reference_time = reference_time)
    chunk_results = [extract_entities_and_relations(chunk, client) for chunk in chunk_text(raw_text)]
    response = _merge_extractions(chunk_results)
    node_id_mapping = {}
    for node in response.nodes:
        existing_node_id = resolve_entity(node.name, node.type, kg, client)
        if existing_node_id is None:
            new_id = normalize(node.name).replace(" ", "_")
            embedding = client.embed(node.name)
            try:
                kg.add_node(new_id, node.type, node.name, embedding=embedding)
            except ValueError:
                pass  # ID collision with a near-duplicate name — reuse existing node
            node_id_mapping[node.name] = new_id
        else:
            node_id_mapping[node.name] = existing_node_id
    for edge in response.edges:
        source_id = node_id_mapping.get(edge.source)
        target_id = node_id_mapping.get(edge.target)
        if source_id is None or target_id is None:
            print(f"Skipping edge with unmapped source or target: {edge}")
            continue
        try:
            kg.add_edge(source_id, target_id, edge.relation, edge.fact, reference_time)
        except ValueError:
            pass
    episode_id = kg.add_episode(episode)
    _log_unmapped(response, episode_id, reference_time)
    for node in set(node_id_mapping.values()):
        try:
            kg.add_edge(node, episode_id, "MENTIONED_IN", "entity mentioned in this episode", reference_time)
            # Stamp the latest episode time so consolidation can pick up just the delta
            kg.update_node(node, last_episode_at=reference_time.isoformat())
        except ValueError:
            pass
    return episode_id
