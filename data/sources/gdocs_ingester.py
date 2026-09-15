import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gdocs_fetcher import fetch_gdocs_documents, save_last_fetched
from core.failure_log import log_ingest_failure


def ingest_gdocs_documents(kg: KnowledgeGarden, client: LLMClient, resolution_client: LLMClient | None = None) -> dict:
    documents, fetch_started_at = fetch_gdocs_documents()
    ingested = 0
    skipped_dedup = 0
    errors = 0

    total = len(documents)
    for i, doc in enumerate(documents, 1):
        document_id = doc["document_id"]

        if kg.get_episode_by_source(document_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{doc['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{doc['title']}'...")
        reference_time = datetime.fromisoformat(doc["modified_time"])

        try:
            episode_id = ingest_episode(
                raw_text=doc["plain_text_content"],
                reference_time=reference_time,
                client=client,
                kg=kg,
                resolution_client=resolution_client,
                source_type="gdocs_document",
            )

            kg.update_episode(
                episode_id,
                source_type="gdocs_document",
                source_id=document_id,
                metadata=json.dumps({
                    "url": doc["url"],
                    "title": doc["title"],
                    "owner": doc["owner"],
                }),
            )
        except Exception as e:
            errors += 1
            print(f"  → error, skipped: {e}")
            log_ingest_failure("gdocs", document_id, doc["title"], e)
            continue

        ingested += 1

    # Advance the fetch cursor only once everything is actually in the graph.
    # Anything still unsaved gets re-fetched next run rather than silently
    # orphaned behind an advanced cursor. Mirrors notion_ingester.
    if errors == 0:
        save_last_fetched(fetch_started_at)
    else:
        print(f"  ⚠ {errors} document(s) failed to ingest — not advancing the fetch "
              f"cursor, so they'll be retried next run.")

    return {
        "total_fetched": len(documents),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
        "errors": errors,
    }
