# Deliberately NOT wired into scripts/run_ingest.py.
#
# claude_sessions_fetcher reads raw .jsonl transcripts straight from
# ~/.claude/projects/<this-repo>/, including this project's own Claude Code
# sessions. Running it as part of the normal ingest would:
#   - ingest unfiltered chat content (secrets/credentials pasted into a
#     session, other people's data) directly into the graph with no redaction
#   - be self-referential: sessions about this KG project would get folded
#     back into the KG itself, creating recursive noise
#
# Wire this in only after adding filtering/redaction, and treat it as a
# separate opt-in source rather than part of the default pipeline.

import json
from datetime import datetime

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.ingester import ingest_episode
from .claude_sessions_fetcher import fetch_claude_sessions
from core.failure_log import log_ingest_failure


def ingest_claude_sessions(kg: KnowledgeGarden, client: LLMClient, resolution_client: LLMClient | None = None) -> dict:
    sessions = fetch_claude_sessions()
    ingested = 0
    skipped_dedup = 0
    errors = 0

    total = len(sessions)
    for i, session in enumerate(sessions, 1):
        session_id = session["session_id"]

        if kg.get_episode_by_source(session_id) is not None:
            skipped_dedup += 1
            print(f"[{i}/{total}] Skipping '{session['title']}' (already ingested)")
            continue

        print(f"[{i}/{total}] Ingesting '{session['title']}'...")
        reference_time = datetime.fromisoformat(session["session_time"])

        try:
            episode_id = ingest_episode(
                raw_text=session["plain_text_content"],
                reference_time=reference_time,
                client=client,
                kg=kg,
                resolution_client=resolution_client,
                source_type="claude_code_session",
            )

            kg.update_episode(
                episode_id,
                source_type="claude_code_session",
                source_id=session_id,
                metadata=json.dumps({
                    "title": session["title"],
                }),
            )
        except Exception as e:
            errors += 1
            print(f"  → error, skipped: {e}")
            log_ingest_failure("claude_sessions", session_id, session["title"], e)
            continue

        ingested += 1

    return {
        "total_fetched": len(sessions),
        "ingested": ingested,
        "skipped_dedup": skipped_dedup,
        "errors": errors,
    }
