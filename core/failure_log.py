import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / ".local" / "ingest_failures.jsonl"


def log_ingest_failure(source: str, item_id: str, title: str, error: Exception):
    """Appends a failed ingest/consolidation item so an overnight run doesn't
    lose the record to a print() nobody's watching. Gmail/Calendar/Docs
    advance their fetch cursor unconditionally at fetch-time (unlike Notion,
    which withholds it on any error), so a document that errors here is never
    re-fetched — this file is the only remaining trace of it."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "item_id": item_id,
            "title": title,
            "error": str(error),
        }) + "\n")
