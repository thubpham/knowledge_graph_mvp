from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.resolver import resolve_entity
from .consolidate import consolidate


def consolidate_all(kg: KnowledgeGarden, client: LLMClient, pending_log_path: str = "pending_edges.txt"):
    all_unresolved = []
    consolidated_count = 0

    for entity_id in list(kg.nodes.keys()):
        node = kg.nodes[entity_id]
        if node.consolidated:
            continue
        result = consolidate(entity_id, kg, client)
        if result is None:
            continue
        consolidated_count += 1
        all_unresolved.extend(result["unresolved_edges"])

    # Second pass: retry edges that may resolve now that all nodes are consolidated
    still_unresolved = []
    second_pass_count = 0

    for pending in all_unresolved:
        target_id = resolve_entity(pending["target"], kg)
        if target_id is None:
            still_unresolved.append(pending)
            continue
        try:
            kg.add_edge(pending["source_id"], target_id, pending["relation"], pending["fact"], datetime.now())
            second_pass_count += 1
        except ValueError:
            pass

    if still_unresolved:
        with open(pending_log_path, "a") as f:
            for edge in still_unresolved:
                f.write(f"{edge['source_id']}\t{edge['relation']}\t{edge['target']}\t{edge['fact']}\n")

    return {
        "consolidated": consolidated_count,
        "edges_resolved_second_pass": second_pass_count,
        "still_unresolved": len(still_unresolved),
    }
