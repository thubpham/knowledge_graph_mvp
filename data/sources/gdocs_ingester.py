import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .gdocs_fetcher import fetch_gdocs_document, fetch_gdocs_documents, save_last_fetched
from core.failure_log import log_ingest_failure


def _ingest_doc(doc: dict, kg: KnowledgeGarden, client: LLMClient, resolution_client: LLMClient | None):
    reference_time = datetime.fromisoformat(doc["modified_time"])
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
        source_id=doc["document_id"],
        metadata=json.dumps({
            "url": doc["url"],
            "title": doc["title"],
            "owner": doc["owner"],
        }),
    )


def ingest_one_gdocs_document(
    kg: KnowledgeGarden,
    client: LLMClient,
    resolution_client: LLMClient | None,
    url_or_id: str,
) -> dict:
    """Ingest a single Google Doc by URL or ID. Same result shape as the bulk
    ingester. Never touches the fetch cursor: a one-off ingest says nothing
    about what the bulk run has or hasn't covered. An already-ingested doc is
    skipped, not re-ingested."""
    doc = fetch_gdocs_document(url_or_id)
    if kg.get_episode_by_source(doc["document_id"]) is not None:
        print(f"Skipping '{doc['title']}' (already ingested)")
        return {"total_fetched": 1, "ingested": 0, "skipped_dedup": 1, "errors": 0}

    print(f"Ingesting '{doc['title']}'...")
    try:
        _ingest_doc(doc, kg, client, resolution_client)
    except Exception as e:
        print(f"  → error: {e}")
        log_ingest_failure("gdocs", doc["document_id"], doc["title"], e)
        return {"total_fetched": 1, "ingested": 0, "skipped_dedup": 0, "errors": 1}
    return {"total_fetched": 1, "ingested": 1, "skipped_dedup": 0, "errors": 0}


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

        try:
            _ingest_doc(doc, kg, client, resolution_client)
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
