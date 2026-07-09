from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from consolidation.consolidate_all import consolidate_all

episodes = [
    "Alice joined the infra team last Monday.",
    "Alice mentioned she's grabbing coffee with Bob before the standup.",
    "Alice left the infra team and joined the platform team this week. She's now leading the payments migration.",
    "The platform team owns the payments service, which depends on Postgres.",
    "Alice reported a bug in the payments service related to timeout handling.",
]

kg = KnowledgeGarden()
client = LLMClient()

print("=== Ingesting episodes ===")
for i, text in enumerate(episodes):
    episode_id = ingest_episode(text, datetime.now(), client, kg)
    print(f"[{i+1}] ingested → episode {episode_id}")

print(f"\nNodes: {[n.id for n in kg.get_all_nodes()]}")
print(f"Edges: {[(e.source, e.relation, e.target) for e in kg.get_all_edges()]}")

print("\n=== Running consolidate_all ===")
result = consolidate_all(kg, client)
print(f"Result: {result}")

print("\n=== Node summaries after consolidation ===")
for node in kg.get_all_nodes():
    if node.summary:
        print(f"\n[{node.name}]\n  summary: {node.summary}\n  last_consolidated_at: {node.last_consolidated_at}")

print("\n=== All edges after consolidation ===")
for edge in kg.get_all_edges():
    src = kg.get_node(edge.source)
    tgt = kg.get_node(edge.target)
    src_name = src.name if src else edge.source
    tgt_name = tgt.name if tgt else edge.target
    if edge.relation != "MENTIONED_IN":
        print(f"  ({src_name}) --[{edge.relation}]--> ({tgt_name})")
