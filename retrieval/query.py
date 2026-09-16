import itertools
import time

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from .query_schema import QueryIntent
from prompts import QUERY_INTENT_SYSTEM_PROMPT, QUERY_INTENT_USER_PROMPT
from .traversal import *
from .scoring import score_edge
from trace import current_run
from enrichment.resolver import resolve_entity, resolve_all_matching

# Same-normalized-name collisions across types are rare on this graph (one
# observed instance: "chat-api"/service vs "chat_api"/concept) but not
# impossible, and resolve_all_matching has no upper bound of its own. These
# caps exist so a pathological case (many nodes sharing a generic normalized
# name) can't turn one query into dozens of graph traversals -- normal
# queries (0 or 1 match) never hit either limit.
MAX_MATCHES_PER_ENTITY = 5
MAX_PATH_PAIRS = 5


def _union_edges(edge_lists) -> list:
    """Flattens multiple edge lists into one, deduped by edge.id, preserving
    first-seen order. Used to merge results across every node a name resolved
    to (see resolve_all_matching) -- a plain concat would double-count any
    edge reachable from more than one anchor."""
    seen = set()
    merged = []
    for edges in edge_lists:
        for e in edges:
            if e.id not in seen:
                seen.add(e.id)
                merged.append(e)
    return merged


def _resolve_query_anchors(name: str, kg: KnowledgeGarden, client: LLMClient) -> list[str]:
    """Every node id an anchor/target name resolves to, for the query path
    specifically. resolve_all_matching only covers exact (normalized) name
    matches and is type-agnostic by design -- a query doesn't know the
    entity's type going in, and unlike a write, returning multiple ids here
    can only add facts, never cause a wrong merge (see resolve_all_matching's
    docstring). Falls back to resolve_entity's embed+LLM-confirm tier for a
    single best guess when there's no exact match at all, so a misspelled or
    paraphrased name still resolves."""
    ids = resolve_all_matching(name, kg)
    if not ids:
        fuzzy_id = resolve_entity(name, None, kg, client)
        ids = [fuzzy_id] if fuzzy_id is not None else []
    return ids[:MAX_MATCHES_PER_ENTITY]


def _step(run, start: float, step: str, **fields):
    """Marks one query-flow phase complete. Logged as a `trace.py` event
    (`Run.event()` was previously wired up nowhere in the codebase — this is
    the first caller) so `scripts/query_trace.py` can render a per-query
    timing waterfall alongside the LLM-call latencies already captured by
    `_traced_call()`."""
    if run is not None:
        run.event(step, duration_ms=int((time.monotonic() - start) * 1000), **fields)


def query(kg: KnowledgeGarden, question: str, client: LLMClient, now: datetime):
    run = current_run()

    t = time.monotonic()
    user_prompt = QUERY_INTENT_USER_PROMPT.replace("{question}", question)
    response = client.generate_gemini(
        QUERY_INTENT_SYSTEM_PROMPT, user_prompt,
        schema_type=QueryIntent, kind="query_intent",
    )
    query_intent = QueryIntent.model_validate_json(response)
    _step(run, t, "intent_parse", pattern=query_intent.pattern, anchor=query_intent.anchor_entity)

    if query_intent.pattern == "unsupported":
        # No anchor to resolve, nothing to traverse -- an honest decline
        # instead of forcing a fit and hallucinating an anchor_entity (see
        # query_schema.py's QueryIntent.pattern docstring). Logged as its own
        # event (not just intent_parse) so scripts/trace_dashboard.py can
        # surface these separately -- this is the signal for deciding
        # whether/what new traversal pattern is actually worth building.
        if run:
            run.event("query_unsupported", question=question, reason=query_intent.reason)
        return {"error": "unsupported query type", "question": question, "reason": query_intent.reason}

    # Resolve the LLM's free-text anchor/target to EVERY matching node, not
    # just one -- resolve_entity (the ingest-side resolver) returns a single,
    # type-scoped id, which is correct for writes but means a query silently
    # only ever sees whichever same-named node the graph scan hits first.
    # Proven live on this graph: "chat-api" (service, 16 edges) and
    # "chat_api" (concept, 2 edges) both normalize to "chat api"; a
    # single-anchor lookup returned only one of them. See
    # _resolve_query_anchors / resolve_all_matching for the full reasoning.
    t = time.monotonic()
    anchor_ids = _resolve_query_anchors(query_intent.anchor_entity, kg, client)
    _step(run, t, "entity_lookup", entity=query_intent.anchor_entity,
          found=bool(anchor_ids), match_count=len(anchor_ids))
    if not anchor_ids:
        return {
            "error": "entity not found",
            "anchor_entity": query_intent.anchor_entity
        }

    target_ids = []
    if query_intent.pattern == "path":
        if query_intent.target_entity is None:
            return {"error": "path query missing target_entity"}
        t = time.monotonic()
        target_ids = _resolve_query_anchors(query_intent.target_entity, kg, client)
        _step(run, t, "entity_lookup", entity=query_intent.target_entity,
              found=bool(target_ids), match_count=len(target_ids))
        if not target_ids:
            return {
                "error": "target entity not found",
                "target_entity": query_intent.target_entity
            }

    anchor_nodes = [kg.get_node(i) for i in anchor_ids]
    returned_edges = []

    t = time.monotonic()
    if query_intent.pattern == "direct_lookup":
        if query_intent.relation is None:
            return {"error": "direct_lookup query missing relation"}
        direction = query_intent.direction or "out"
        returned_edges = _union_edges(
            direct_lookup(kg, n, query_intent.relation, direction) for n in anchor_nodes
        )

    elif query_intent.pattern == "neighborhood":
        returned_edges = _union_edges(neighbor_expansion(kg, n) for n in anchor_nodes)

    elif query_intent.pattern == "path":
        target_nodes = [kg.get_node(i) for i in target_ids]
        # Capped rather than a full cross product: each pair is its own graph
        # traversal, so N anchors x M targets is quadratic in query cost, not
        # just result size. Ordinary queries have 1 anchor x 1 target = 1 pair
        # and never notice this cap.
        pairs = list(itertools.product(anchor_nodes, target_nodes))[:MAX_PATH_PAIRS]
        returned_edges = _union_edges(path_finding(kg, a, b) for a, b in pairs)

    elif query_intent.pattern == "impact":
        if query_intent.relation is None:
            return {"error": "impact query missing relation"}
        direction = query_intent.direction or "in"
        returned_edges = _union_edges(
            impact_traversal(kg, n, query_intent.relation, direction) for n in anchor_nodes
        )

    elif query_intent.pattern == "history":
        if query_intent.relation is None:
            return {"error": "history query missing relation"}
        direction = query_intent.direction or "out"
        result = _union_edges(
            history_traversal(kg, n, query_intent.relation, direction) for n in anchor_nodes
        )
        result.sort(key=lambda e: e.valid_from)
        _step(run, t, "traversal", pattern="history")
        return result

    else:
        return {"error": "unknown pattern", "pattern": query_intent.pattern}

    _step(run, t, "traversal", pattern=query_intent.pattern, edge_count=len(returned_edges))

    t = time.monotonic()
    valid_edges = [
        e for e in returned_edges
        if score_edge(e, now) != 0.0 and e.relation != "MENTIONED_IN"
    ]
    valid_edges.sort(key=lambda e: score_edge(e, now), reverse=True)
    _step(run, t, "scoring", edge_count=len(valid_edges))
    return valid_edges
