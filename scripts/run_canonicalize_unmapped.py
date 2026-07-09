import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.canonicalize_unmapped import canonicalize_unmapped_log

kg = KnowledgeGarden()
client = LLMClient()

result = canonicalize_unmapped_log(kg, client)

print(f"Entities resolved:  {result['entities_resolved']}")
print(f"Relations resolved: {result['relations_resolved']}")
print(f"Left unmapped:      {result['left_unmapped']}")
