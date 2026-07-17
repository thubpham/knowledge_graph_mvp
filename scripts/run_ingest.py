import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from trace import Run
from data.sources.notion_ingester import ingest_notion_pages
from data.sources.gcal_ingester import ingest_gcal_events
from data.sources.gmail_ingester import ingest_gmail_messages
from data.sources.gdocs_ingester import ingest_gdocs_documents
from data.sources.claude_sessions_ingester import ingest_claude_sessions

kg = KnowledgeGarden()
client = LLMClient()

def _run_source(name: str, fn):
    print(f"── {name} " + "─" * max(1, 40 - len(name)))
    try:
        with Run(flow="ingest", meta={"source": name}):
            result = fn(kg, client)
    except Exception as e:
        print(f"\n  ✗ {name} aborted: {e}")
        return {"total_fetched": 0, "ingested": 0, "skipped_dedup": 0, "errors": 0}
    print(
        f"\n  ✓ {name} done — Fetched: {result['total_fetched']} | "
        f"Ingested: {result['ingested']} | Skipped (dedup): {result['skipped_dedup']} | "
        f"Errors: {result.get('errors', 0)}"
    )
    return result


_run_source("Notion", ingest_notion_pages)
print()
_run_source("Google Calendar", ingest_gcal_events)
print()
_run_source("Gmail", ingest_gmail_messages)
print()
_run_source("Google Docs", ingest_gdocs_documents)
print()
_run_source("Claude Code Sessions", ingest_claude_sessions)

print("\nDone. Run `python scripts/run_consolidation.py` next.")
