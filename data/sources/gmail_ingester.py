import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gmail_fetcher import fetch_gmail_messages, save_last_fetched
from core.failure_log import log_ingest_failure

# Maps a Gmail category label to the base of retrieval/scoring.py's
# GMAIL_CATEGORY_WEIGHT key. CATEGORY_PROMOTIONS deliberately has no entry --
# the fetcher excludes it at the query level (`-category:promotions`), so it
# should never reach here; if one somehow does (e.g. a stale query), it falls
# through to "none" via .get() below rather than crashing.
_GMAIL_CATEGORY_MAP = {
    "CATEGORY_PERSONAL": "personal",
    "CATEGORY_FORUMS": "forums",
    "CATEGORY_UPDATES": "updates",
}


def _gmail_source_category(labels: list[str]) -> str:
    """First matching category in priority order (a message can carry more
    than one CATEGORY_* label; PERSONAL is the strongest relevance signal so
    it wins), then an "_important" suffix if Gmail's own IMPORTANT flag is
    set. "none" (no CATEGORY_* label at all) covers real human `Re:`/`Fwd:`
    threads Gmail didn't bucket -- see retrieval/scoring.py, weighted the
    same as PERSONAL rather than as a fallback default."""
    base = "none"
    for label, mapped in _GMAIL_CATEGORY_MAP.items():
        if label in labels:
            base = mapped
            break
    if "IMPORTANT" in labels:
        base += "_important"
    return base


def ingest_gmail_messages(kg: KnowledgeGarden, client: LLMClient, resolution_client: LLMClient | None = None) -> dict:
    messages, fetch_started_at = fetch_gmail_messages()
    ingested = 0
    skipped_dedup = 0
    errors = 0

    total = len(messages)
    for i, message in enumerate(messages, 1):
        message_id = message["message_id"]

        if kg.get_episode_by_source(message_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{message['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{message['title']}'...")
        reference_time = datetime.fromisoformat(message["message_time"])

        try:
            episode_id = ingest_episode(
                raw_text=message["plain_text_content"],
                reference_time=reference_time,
                client=client,
                kg=kg,
                resolution_client=resolution_client,
                source_type="gmail_message",
                source_category=_gmail_source_category(message.get("labels", [])),
            )

            kg.update_episode(
                episode_id,
                source_type="gmail_message",
                source_id=message_id,
                metadata=json.dumps({
                    "url": message["url"],
                    "title": message["title"],
                    "sender": message["sender"],
                }),
            )
        except Exception as e:
            errors += 1
            print(f"  → error, skipped: {e}")
            log_ingest_failure("gmail", message_id, message["title"], e)
            continue

        ingested += 1

    # Advance the fetch cursor only once everything is actually in the graph.
    # Anything still unsaved gets re-fetched next run rather than silently
    # orphaned behind an advanced cursor. Mirrors notion_ingester.
    if errors == 0:
        save_last_fetched(fetch_started_at)
    else:
        print(f"  ⚠ {errors} message(s) failed to ingest — not advancing the fetch "
              f"cursor, so they'll be retried next run.")

    return {
        "total_fetched": len(messages),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
        "errors": errors,
    }
