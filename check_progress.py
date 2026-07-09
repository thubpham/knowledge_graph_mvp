import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import KnowledgeGarden

kg = KnowledgeGarden()
nodes = kg.get_all_nodes()

total = len(nodes)
pending_nodes = [
    n for n in nodes
    if n.last_episode_at and (n.last_consolidated_at is None or n.last_episode_at > n.last_consolidated_at)
]
pending = len(pending_nodes)
done = total - pending
pct = (done / total * 100) if total else 0

bar_width = 30
filled = int(bar_width * done / total) if total else 0
bar = "█" * filled + "░" * (bar_width - filled)

print(f"Consolidation progress: [{bar}] {pct:.0f}%")
print(f"  Consolidated: {done} / {total}")
print(f"  Pending:      {pending}")
