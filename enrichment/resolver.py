from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from prompts import ENTITY_MATCH_PROMPT
from .extraction_schema import EntityMatchResult
import re

# FalkorDB's vector index 'score' for similarity_function='cosine' is a
# distance (lower = more similar), not a similarity — confirmed empirically
# against a live instance. Thresholds below are calibrated directly on that
# distance using real gemini-embedding-001 output on short entity-name-style
# strings:
#   obvious synonyms (Postgres/PostgreSQL, infra team/infrastructure team,
#     auth service/authentication service)         -> distance 0.11-0.16
#   same-category but distinct (Alice/Bob)          -> distance 0.335
#   abbreviation (KG MVP/knowledge graph project)   -> distance 0.350
#   genuinely unrelated pairs                       -> distance 0.38-0.44
# Note distinct-but-related and merge-worthy pairs overlap heavily in the
# 0.2-0.4 band (e.g. Alice/Bob at 0.335 sits *closer* than KG MVP's true
# synonym at 0.350) — a pure distance cutoff cannot safely resolve that band,
# which is exactly why it's routed to the LLM confirmation step rather than
# auto-decided.
AUTO_MATCH_DISTANCE = 0.20
NO_MATCH_DISTANCE = 0.42
CANDIDATE_K = 5


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def resolve_entity(node_name: str, node_type: str | None, kg: KnowledgeGarden, client: LLMClient):
    # 1. Exact-match fast path (cheap, avoids embedding calls for repeats).
    normalized_name = normalize(node_name)
    for node in kg.get_all_nodes():
        if normalize(node.name) == normalized_name:
            return node.id

    # 2. Embed the name and search for nearby candidates.
    embedding = client.embed(node_name)
    candidates = kg.find_similar_nodes(embedding, entity_type=node_type, k=CANDIDATE_K)
    if not candidates:
        return None

    top_node, top_distance = candidates[0]

    # 3. Threshold tiers (distance: lower = more similar).
    if top_distance > NO_MATCH_DISTANCE:
        return None
    if top_distance <= AUTO_MATCH_DISTANCE:
        return top_node.id

    # 4. Ambiguous zone — ask the LLM to confirm against the candidate set.
    candidates_text = "\n".join(
        f'{i+1}. name="{c.name}", type={c.type}, distance={dist:.2f}'
        for i, (c, dist) in enumerate(candidates)
    )
    prompt = (
        ENTITY_MATCH_PROMPT
        .replace("{new_name}", node_name)
        .replace("{new_type}", node_type or "unknown")
        .replace("{candidates}", candidates_text)
    )
    raw = client.generate_gemini(prompt, schema_type=EntityMatchResult)
    result = EntityMatchResult.model_validate_json(raw)

    if result.match_name is None:
        return None
    for c, _ in candidates:
        if c.name == result.match_name:
            return c.id
    return None
