import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from trace import Run
from consolidation.consolidate_all import consolidate_all, get_pending_nodes

parser = argparse.ArgumentParser(
    description="Consolidate entities with new episodes since their last consolidation "
    "into an updated summary + resolved semantic edges."
)
# Default 4, calibrated against the real post-backfill graph (2026-07-26:
# 2165 active entities, 262 episodes) rather than guessed. The measured
# episodes-per-entity distribution was p50=1, p90=4, max=17, with 1325 of 2165
# entities (61%) sitting at exactly ONE episode -- consolidating those is
# near-pointless, since "summarize this single mention" adds nothing over the
# episode itself. Thresholds measured against that graph:
#     1 -> 2165 entities (100%), ~$8.66/run   [the old implicit behavior]
#     2 ->  840 entities  (38%), ~$3.36/run
#     3 ->  410 entities  (18%), ~$1.64/run
#     4 ->  246 entities  (11%), ~$0.98/run   <- here
#     5 ->  132 entities   (6%), ~$0.53/run
# 4 captures every entity with genuinely accumulating history (True Ventures,
# OpenAI, Slack, FalkorDB, knowledge garden -- all 10+ episodes) at ~1/3 the
# cost of 2. Nothing is permanently lost by excluding the 2-3 episode entities:
# they cross the threshold on their own as episodes accumulate, so this defers
# rather than drops. Re-measure if the source mix changes substantially.
parser.add_argument("--min-episodes", type=int, default=4,
                     help="skip an entity's consolidation unless it has at least this many "
                          "new episodes since it was last consolidated (default: 4)")
args = parser.parse_args()

kg = KnowledgeGarden()
# Consolidation is multi-step reasoning (change-over-time across episodes) ->
# Groq's 70B model by convention, same rationale as extraction. No hardcoded
# fallback here — .env is the single source of truth; if unset, LLMClient
# falls through to LLM_PROVIDER, then "gemini". See llm_clients.py and
# IMPROVEMENTS.md's Provider Routing section.
client = LLMClient(provider=os.getenv("CONSOLIDATION_LLM_PROVIDER"))

nodes = kg.get_all_nodes()
pending = get_pending_nodes(kg, args.min_episodes)
print(f"Nodes total: {len(nodes)} | Pending consolidation (>= {args.min_episodes} new episode(s)): {len(pending)}\n")

with Run(flow="consolidation", meta={"pending": len(pending), "min_new_episodes": args.min_episodes}):
    result = consolidate_all(kg, client, min_new_episodes=args.min_episodes)

print(f"\n── Consolidation complete ──")
print(f"  Consolidated:          {result['consolidated']}")
print(f"  Errors:                {result['errors']}")
print(f"  Edges resolved (pass2): {result['edges_resolved_second_pass']}")
print(f"  Still unresolved:      {result['still_unresolved']}")
