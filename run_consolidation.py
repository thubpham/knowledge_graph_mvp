import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from consolidation.consolidate_all import consolidate_all

kg = KnowledgeGarden()
client = LLMClient()

nodes = kg.get_all_nodes()
pending = [n for n in nodes if not n.consolidated]
print(f"Nodes total: {len(nodes)} | Pending consolidation: {len(pending)}\n")

result = consolidate_all(kg, client)

print(f"\n── Consolidation complete ──")
print(f"  Consolidated:          {result['consolidated']}")
print(f"  Edges resolved (pass2): {result['edges_resolved_second_pass']}")
print(f"  Still unresolved:      {result['still_unresolved']}")
