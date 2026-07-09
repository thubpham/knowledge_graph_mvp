import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gmail_fetcher import fetch_gmail_messages


def ingest_gmail_messages(kg: KnowledgeGarden, client: LLMClient) -> dict:
    messages = fetch_gmail_messages()
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
            continue

        ingested += 1

    return {
        "total_fetched": len(messages),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
        "errors": errors,
    }
