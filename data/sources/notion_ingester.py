import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .notion_fetcher import fetch_notion_pages


def ingest_notion_pages(kg: KnowledgeGarden, client: LLMClient) -> dict:
    pages = fetch_notion_pages()
    ingested = 0
    skipped_dedup = 0

    total = len(pages)
    for i, page in enumerate(pages, 1):
        page_id = page["page_id"]

        if kg.get_episode_by_source(page_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{page['page_title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{page['page_title']}'...")
        reference_time = datetime.fromisoformat(page["last_edited_time"])

        episode_id = ingest_episode(
            raw_text=page["plain_text_content"],
            reference_time=reference_time,
            client=client,
            kg=kg,
        )

        kg.update_episode(
            episode_id,
            source_type="notion_page",
            source_id=page_id,
            metadata=json.dumps({"url": page["url"], "title": page["page_title"]}),
        )

        ingested += 1

    return {
        "total_fetched": len(pages),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
    }
