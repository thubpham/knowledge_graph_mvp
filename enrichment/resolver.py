from core.graph import KnowledgeGarden
from core.schema import Node
from llm_clients import LLMClient
from prompts import ENTITY_MATCH_PROMPT
from .extraction_schema import EntityMatchResult
from .aliases import load_aliases
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


def embedding_text(name: str, context: str | None) -> str:
    """The string actually embedded for a node — bare name alone carries too
    little semantic signal to disambiguate short, similarly-named-but-different
    entities (see IMPROVEMENTS.md's "lack of context" finding: "Bob"/"vern",
    "PostgreSQL"/"crm" landing suspiciously close in embedding space). Callers
    with a source-text snippet available (entity extraction) should pass it as
    `context`; callers without one (consolidation targets, unmapped-log
    replay) pass `None` and fall back to name-only, matching legacy behavior."""
    return f"{name}: {context}" if context else name


def confirm_match(
    new_name: str, new_type: str | None, candidates: list[tuple[Node, float]], client: LLMClient
) -> str | None:
    """Ask the LLM to confirm whether `new_name` is the same real-world
    entity as exactly one of `candidates`. Shared by `resolve_entity`'s
    ambiguous-band tier and the offline dedup pass in `enrichment/dedup.py`
    so there's exactly one implementation of this prompt-filling logic.
    Returns the matching candidate's node id, or None if no confident match."""
    candidates_text = "\n".join(
        f'{i+1}. name="{c.name}", type={c.type}, distance={dist:.2f}'
        for i, (c, dist) in enumerate(candidates)
    )
    prompt = (
        ENTITY_MATCH_PROMPT
        .replace("{new_name}", new_name)
        .replace("{new_type}", new_type or "unknown")
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


def resolve_entity(
    node_name: str,
    node_type: str | None,
    kg: KnowledgeGarden,
    client: LLMClient,
    context: str | None = None,
):
    # 0. Alias fast path — cheap dict lookup, no embedding/LLM cost. Catches
    # structural identity aliasing (a name vs. a full name vs. an email
    # address) that the embedding-based tiers below cannot: those strings
    # just don't sit close together in embedding space, so the pair never
    # even reaches the LLM confirmation tier. Aliases are seeded by hand or
    # confirmed via the offline dedup pass (`enrichment/dedup.py`,
    # `scripts/run_dedup_review.py --apply`).
    normalized_name = normalize(node_name)
    aliases = load_aliases()
    if normalized_name in aliases:
        return aliases[normalized_name]

    # 1. Exact-match fast path (cheap, avoids embedding calls for repeats).
    # Type-scoped when a type is known, so identical names of different
    # types (e.g. a "person" and a "tool" both called "Postgres") don't
    # cross-match, and the candidate pool stays small.
    candidates_pool = kg.get_nodes_by_type(node_type) if node_type else kg.get_all_nodes()
    for node in candidates_pool:
        if normalize(node.name) == normalized_name:
            return node.id

    # 2. Embed the name (+ source-text context, when available) and search
    # for nearby candidates. See embedding_text()'s docstring for why bare
    # names alone are an insufficient signal here.
    embedding = client.embed(embedding_text(node_name, context))
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
    return confirm_match(node_name, node_type, candidates, client)
