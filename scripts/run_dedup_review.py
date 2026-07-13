import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.dedup import find_merge_candidates, review_cluster, log_rejected
from enrichment.resolver import normalize
from enrichment.aliases import save_alias

# Keep in sync with the allowed node types in prompts.py's EXTRACTION_PROMPT.
NODE_TYPES = ["person", "service", "team", "tool", "concept", "event", "document"]

parser = argparse.ArgumentParser(
    description="Offline entity-dedup pass: clusters same-type nodes by embedding "
    "distance and proposes merges (tombstone-based, never deletes). Dry-run by "
    "default — pass --apply to actually write."
)
parser.add_argument("--type", choices=NODE_TYPES, help="restrict to one node type (default: all)")
parser.add_argument("--apply", action="store_true", help="write merges + aliases (default: dry-run)")
args = parser.parse_args()

kg = KnowledgeGarden()
client = LLMClient()

types_to_scan = [args.type] if args.type else NODE_TYPES

totals = {"auto": 0, "llm": 0, "rejected": 0}

for node_type in types_to_scan:
    clusters = find_merge_candidates(kg, node_type)
    if not clusters:
        continue

    print(f"── {node_type} " + "─" * max(1, 40 - len(node_type)))
    for cluster in clusters:
        proposals = review_cluster(kg, client, cluster)
        for p in proposals:
            totals[p["method"]] += 1
            print(
                f'  [{p["method"]:>8}] "{p["loser_name"]}" ({p["loser_id"]}) -> '
                f'"{p["winner_name"]}" ({p["winner_id"]}) — distance={p["distance"]:.3f}'
            )
            if not args.apply:
                continue
            if p["method"] == "rejected":
                log_rejected(p)
                continue
            kg.merge_nodes(p["loser_id"], p["winner_id"])
            save_alias(normalize(p["loser_name"]), p["winner_id"])
    print()

print("── Summary " + "─" * 30)
print(f"  Auto-merged:    {totals['auto']}")
print(f"  LLM-confirmed:  {totals['llm']}")
print(f"  Rejected:       {totals['rejected']}")
if not args.apply and (totals["auto"] or totals["llm"] or totals["rejected"]):
    print("\n  Dry-run only — no writes made. Re-run with --apply to merge and log.")
