import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from data.sources.notion_ingester import ingest_notion_pages
from data.sources.gcal_ingester import ingest_gcal_events
from data.sources.gmail_ingester import ingest_gmail_messages
from data.sources.gdocs_ingester import ingest_gdocs_documents
from data.sources.claude_sessions_ingester import ingest_claude_sessions

kg = KnowledgeGarden()
client = LLMClient()

print("── Notion ──────────────────────────────")
notion_result = ingest_notion_pages(kg, client)
print(f"\n  ✓ Notion done — Fetched: {notion_result['total_fetched']} | Ingested: {notion_result['ingested']} | Skipped (dedup): {notion_result['skipped_dedup']}")

print("\n── Google Calendar ─────────────────────")
gcal_result = ingest_gcal_events(kg, client)
print(f"\n  ✓ GCal done — Fetched: {gcal_result['total_fetched']} | Ingested: {gcal_result['ingested']} | Skipped (dedup): {gcal_result['skipped_dedup']}")

print("\n── Gmail ────────────────────────────────")
gmail_result = ingest_gmail_messages(kg, client)
print(f"\n  ✓ Gmail done — Fetched: {gmail_result['total_fetched']} | Ingested: {gmail_result['ingested']} | Skipped (dedup): {gmail_result['skipped_dedup']}")

print("\n── Google Docs ──────────────────────────")
gdocs_result = ingest_gdocs_documents(kg, client)
print(f"\n  ✓ Google Docs done — Fetched: {gdocs_result['total_fetched']} | Ingested: {gdocs_result['ingested']} | Skipped (dedup): {gdocs_result['skipped_dedup']}")

print("\n── Claude Code Sessions ─────────────────")
claude_result = ingest_claude_sessions(kg, client)
print(f"\n  ✓ Claude Code done — Fetched: {claude_result['total_fetched']} | Ingested: {claude_result['ingested']} | Skipped (dedup): {claude_result['skipped_dedup']}")

print("\nDone. Run `python run_consolidation.py` next.")
