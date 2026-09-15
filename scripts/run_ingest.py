import os
import signal
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph import KnowledgeGarden
from core.failure_log import log_ingest_failure
from llm_clients import LLMClient
from trace import Run
from data.sources.notion_ingester import ingest_notion_pages
from data.sources.gcal_ingester import ingest_gcal_events
from data.sources.gmail_ingester import ingest_gmail_messages
from data.sources.gdocs_ingester import ingest_gdocs_documents
# Claude Code Sessions is deliberately excluded here — see the comment at the
# top of data/sources/claude_sessions_ingester.py.

def _terminate(signum, _frame):
    """Turn SIGTERM/SIGINT into SystemExit so _run_source's handler can record
    the abort before the process goes away. Without this, a kill (timeout
    wrapper, OOM, laptop sleep) unwinds with nothing written -- which is
    exactly how a partially-ingested Gmail run went unlogged on 2026-07-25."""
    raise SystemExit(f"received signal {signal.Signals(signum).name}")


signal.signal(signal.SIGTERM, _terminate)
signal.signal(signal.SIGINT, _terminate)

kg = KnowledgeGarden()
# Extraction is accuracy-sensitive and relatively low call volume (once per
# chunk) -> Groq's 70B model by convention. Entity-resolution confirmation is
# the highest call-volume, lowest-individual-stakes step (a wrong local call
# just falls back to "no match") -> local Ollama by convention. No hardcoded
# fallback here — .env is the single source of truth; if a var is unset,
# LLMClient falls through to LLM_PROVIDER, then "gemini". See llm_clients.py
# and IMPROVEMENTS.md's Provider Routing section.
client = LLMClient(provider=os.getenv("EXTRACTION_LLM_PROVIDER"))
resolution_client = LLMClient(provider=os.getenv("RESOLVER_LLM_PROVIDER"))

def _run_source(name: str, fn):
    print(f"── {name} " + "─" * max(1, 40 - len(name)))
    try:
        with Run(flow="ingest", meta={"source": name}):
            result = fn(kg, client, resolution_client)
    except (Exception, SystemExit) as e:
        # A whole-source abort used to print and vanish -- ingest_failures.jsonl
        # only ever recorded per-item failures raised *inside* an ingester, so
        # the single most significant failure mode (a source dying outright)
        # left no trace for the next morning. SystemExit is included because
        # the SIGTERM handler below raises it, and it doesn't inherit from
        # Exception: without it, a kill/OOM/sleep would still go unlogged.
        print(f"\n  ✗ {name} aborted: {e}")
        log_ingest_failure(
            source=name,
            item_id="<source-level-abort>",
            title=f"{name} aborted before completion",
            error=e,
        )
        if isinstance(e, SystemExit):
            raise  # a kill means stop the run, not move to the next source
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

print("\nDone. Run `python scripts/run_consolidation.py` next.")
