import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .claude_sessions_fetcher import fetch_claude_sessions


def ingest_claude_sessions(kg: KnowledgeGarden, client: LLMClient) -> dict:
    sessions = fetch_claude_sessions()
    ingested = 0
    skipped_dedup = 0

    total = len(sessions)
    for i, session in enumerate(sessions, 1):
        session_id = session["session_id"]

        if kg.get_episode_by_source(session_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{session['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{session['title']}'...")
        reference_time = datetime.fromisoformat(session["session_time"])

        episode_id = ingest_episode(
            raw_text=session["plain_text_content"],
            reference_time=reference_time,
            client=client,
            kg=kg,
        )

        kg.update_episode(
            episode_id,
            source_type="claude_code_session",
            source_id=session_id,
            metadata=json.dumps({
                "title": session["title"],
            }),
        )

        ingested += 1

    return {
        "total_fetched": len(sessions),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
    }
