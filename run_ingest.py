import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from data.sources.notion_ingester import ingest_notion_pages
from data.sources.gcal_ingester import ingest_gcal_events

kg = KnowledgeGarden()
client = LLMClient()

print("── Notion ──────────────────────────────")
notion_result = ingest_notion_pages(kg, client)
print(f"\n  ✓ Notion done — Fetched: {notion_result['total_fetched']} | Ingested: {notion_result['ingested']} | Skipped (dedup): {notion_result['skipped_dedup']}")

print("\n── Google Calendar ─────────────────────")
gcal_result = ingest_gcal_events(kg, client)
print(f"\n  ✓ GCal done — Fetched: {gcal_result['total_fetched']} | Ingested: {gcal_result['ingested']} | Skipped (dedup): {gcal_result['skipped_dedup']}")

print("\nDone. Run `python run_consolidation.py` next.")
