from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.resolver import resolve_entity
from .consolidate import consolidate


def consolidate_all(kg: KnowledgeGarden, client: LLMClient, pending_log_path: str = "pending_edges.txt"):
    all_unresolved = []
    consolidated_count = 0

    pending = [n for n in kg.get_all_nodes() if not n.consolidated]
    total = len(pending)
    print(f"Consolidating {total} entities...")
    for i, node in enumerate(pending, 1):
        print(f"[{i}/{total}] {node.name} ({node.id})")
        result = consolidate(node.id, kg, client)
        if result is None:
            print(f"  → no episodes, skipped")
            continue
        consolidated_count += 1
        print(f"  → {result['edges_added']} edges added, {len(result['unresolved_edges'])} unresolved")
        all_unresolved.extend(result["unresolved_edges"])

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
