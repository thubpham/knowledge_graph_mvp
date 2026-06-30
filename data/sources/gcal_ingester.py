import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gcal_fetcher import fetch_gcal_events


def ingest_gcal_events(kg: KnowledgeGarden, client: LLMClient) -> dict:
    events = fetch_gcal_events()
    ingested = 0
    skipped_dedup = 0

    total = len(events)
    for i, event in enumerate(events, 1):
        event_id = event["event_id"]

        if kg.get_episode_by_source(event_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{event['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{event['title']}'...")
        reference_time = datetime.fromisoformat(event["event_time"])

        episode_id = ingest_episode(
            raw_text=event["plain_text_content"],
            reference_time=reference_time,
            client=client,
            kg=kg,
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

        ingested += 1

    return {
        "total_fetched": len(events),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
    }
