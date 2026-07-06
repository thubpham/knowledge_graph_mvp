import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient

kg = KnowledgeGarden()
client = LLMClient()

nodes = kg.get_all_nodes()
total = len(nodes)
print(f"Embedding {total} entities...")

start = time.monotonic()
errors = 0
for i, node in enumerate(nodes, 1):
    elapsed = time.monotonic() - start
    avg = elapsed / (i - 1) if i > 1 else 0
    remaining = (total - i + 1) * avg
    print(f"[{i}/{total}] {node.name} ({node.id})  "
          f"[elapsed {elapsed/60:.1f}m, ETA {remaining/60:.1f}m]")
    try:
        embedding = client.embed(node.name)
        kg.update_node_embedding(node.id, embedding)
    except Exception as e:
        errors += 1
        print(f"  -> error, skipped: {e}")

print(f"\n-- Backfill complete --")
print(f"  Embedded: {total - errors}")
print(f"  Errors:   {errors}")
