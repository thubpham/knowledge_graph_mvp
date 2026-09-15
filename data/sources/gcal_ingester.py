import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gcal_fetcher import fetch_gcal_events, save_last_fetched
from core.failure_log import log_ingest_failure


def ingest_gcal_events(kg: KnowledgeGarden, client: LLMClient, resolution_client: LLMClient | None = None) -> dict:
    events, fetch_started_at = fetch_gcal_events()
    ingested = 0
    skipped_dedup = 0
    errors = 0

    total = len(events)
    for i, event in enumerate(events, 1):
        event_id = event["event_id"]

        if kg.get_episode_by_source(event_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{event['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{event['title']}'...")
        reference_time = datetime.fromisoformat(event["event_time"])

        try:
            episode_id = ingest_episode(
                raw_text=event["plain_text_content"],
                reference_time=reference_time,
                client=client,
                kg=kg,
                resolution_client=resolution_client,
                source_type="gcal_event",
            )

            kg.update_episode(
                episode_id,
                source_type="gcal_event",
                source_id=event_id,
                metadata=json.dumps({
                    "url": event["url"],
                    "title": event["title"],
                    "organizer": event["organizer"],
                    "attendees": event["attendees"],
                }),
            )
        except Exception as e:
            errors += 1
            print(f"  → error, skipped: {e}")
            log_ingest_failure("gcal", event_id, event["title"], e)
            continue

        ingested += 1

    # Advance the fetch cursor only once everything is actually in the graph.
    # Anything still unsaved gets re-fetched next run rather than silently
    # orphaned behind an advanced cursor. Mirrors notion_ingester.
    if errors == 0:
        save_last_fetched(fetch_started_at)
    else:
        print(f"  ⚠ {errors} event(s) failed to ingest — not advancing the fetch "
              f"cursor, so they'll be retried next run.")

    return {
        "total_fetched": len(events),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
        "errors": errors,
    }
