import json
from pathlib import Path

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from core.schema import *
from .extractor import extract_entities_and_relations
from .resolver import resolve_entity, slugify, embedding_text
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


def ingest_episode(
    raw_text: str,
    reference_time: datetime,
    client: LLMClient,
    kg: KnowledgeGarden,
    resolution_client: LLMClient | None = None,
    source_type: str | None = None,
    source_category: str | None = None,
):
    """`client` handles extraction (accuracy-sensitive, low call volume relative
    to resolution). `resolution_client` handles `resolve_entity`'s LLM
    confirmation tier and `.embed()` calls; defaults to `client` when not given
    (e.g. ad-hoc/notebook use) so this stays backward-compatible. Callers that
    want the Groq-extraction/Ollama-resolution split (see llm_clients.py) pass
    both explicitly — see scripts/run_ingest.py.

    `source_type`/`source_category` are stamped onto every edge this episode
    creates (semantic + MENTIONED_IN), for retrieval/scoring.py's source-based
    weighting. Optional and additive: omitting them reproduces the exact prior
    behavior (edges get source_type=None, scored at the neutral default
    weight), so existing callers that don't pass them are unaffected. Must be
    passed here rather than added later via kg.update_episode() the way
    source_type already is on the Episode node itself -- semantic edges are
    created (via kg.add_edge below) BEFORE this episode is persisted, so
    there's no edge to retroactively stamp afterward."""
    resolution_client = resolution_client or client
    episode = Episode(raw_text, reference_time = reference_time)
    known_entities = [
        f"{node.name} ({node.type})"
        for node in kg.get_recently_active_nodes(before=reference_time)
    ]
    chunk_results = [
        extract_entities_and_relations(chunk, client, known_entities=known_entities)
        for chunk in chunk_text(raw_text)
    ]
    response = _merge_extractions(chunk_results)
    node_id_mapping = {}
    for node in response.nodes:
        existing_node_id = resolve_entity(node.name, node.type, kg, resolution_client, context=node.context)
        if existing_node_id is None:
            base_id = slugify(node.name)
            embedding = resolution_client.embed(embedding_text(node.name, node.context))
            new_id = base_id
            try:
                kg.add_node(new_id, node.type, node.name, embedding=embedding)
            except ValueError:
                # Slug collision: another node already owns this id. This
                # previously fell through with a bare `pass`, silently
                # attaching this episode's edges to whatever node was already
                # there -- correct if it's the same entity re-encountered
                # (resolve_entity should have caught that above, but a race or
                # a normalize edge case can still land here), wrong if it's a
                # same-name DIFFERENT entity (e.g. a "service" and a "concept"
                # both named "chat-api" -- resolve_entity's exact-match tier is
                # type-scoped, so it can't see across that boundary and
                # correctly returns None for the second one). Disambiguate by
                # type instead of blindly reusing: if the existing node is a
                # different type, this is very likely a different real-world
                # entity, so give it its own node rather than merging their
                # edges. Loop in case the disambiguated id ALSO collides
                # (e.g. two different "tool"-typed near-duplicates in the same
                # episode) rather than assuming one retry is enough.
                existing = kg.get_node(base_id)
                if existing.type == node.type:
                    new_id = base_id  # same type -> treat as the same entity, reuse
                else:
                    suffix = 2
                    new_id = f"{base_id}__{node.type}"
                    while kg.node_exists(new_id):
                        existing_at_candidate = kg.get_node(new_id)
                        if existing_at_candidate.type == node.type:
                            break  # a same-type disambiguated node already exists -- reuse it
                        new_id = f"{base_id}__{node.type}_{suffix}"
                        suffix += 1
                    if not kg.node_exists(new_id):
                        kg.add_node(new_id, node.type, node.name, embedding=embedding)
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
            kg.add_edge(source_id, target_id, edge.relation, edge.fact, reference_time,
                       source_type=source_type, source_id=episode.id, source_category=source_category)
        except ValueError:
            pass
    episode_id = kg.add_episode(episode)
    _log_unmapped(response, episode_id, reference_time)
    for node in set(node_id_mapping.values()):
        try:
            kg.add_edge(node, episode_id, "MENTIONED_IN", "entity mentioned in this episode", reference_time,
                       source_type=source_type, source_id=episode.id, source_category=source_category)
            # Stamp the latest episode time so consolidation can pick up just the delta
            kg.update_node(node, last_episode_at=reference_time.isoformat())
        except ValueError:
            pass
    return episode_id
