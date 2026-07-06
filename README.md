# Knowledge Graph MVP

A personal, bi-temporal knowledge graph that ingests your own data (Notion, Gmail,
Google Calendar, Google Docs, Claude Code sessions) and turns it into a queryable
memory: entities, relationships, and facts with a history of when they were true —
not just what's true right now.

## 1) What this is

Most personal knowledge tools store documents. This stores *facts extracted from*
those documents, as a graph:

- **Episodes** — raw ingested text (a Notion page, an email, a calendar event, a
  chunk of a Claude Code session), each stamped with when it happened.
- **Entities** (nodes) — people, tools, services, teams, and concepts mentioned
  across your episodes, deduplicated so "Postgres" and "PostgreSQL" resolve to one
  node instead of two.
- **Edges** — typed relationships between entities (`MEMBER_OF`, `OWNS`,
  `DEPENDS_ON`, `USES`, `REPORTED`, `RESOLVED_BY`), each carrying the fact that
  justified it and a `valid_from`/`valid_until` window. When something changes
  (e.g. you switch teams), the old edge is closed out rather than deleted — the
  graph remembers what *used to* be true, not just what's current.

The pipeline runs in two stages:

1. **Ingestion** — pull raw data from a source, extract entities/relations with an
   LLM, resolve them against existing nodes (dedup), write episodic edges.
2. **Consolidation** — periodically fold each entity's accumulated episodes into a
   durable summary and promote recurring facts into semantic edges, superseding
   anything that's changed.

This mirrors the architecture behind [Zep/Graphiti](https://arxiv.org/abs/2501.13956)
(temporal knowledge graphs for agent memory), scaled down to a single-user project
backed by [FalkorDB](https://falkordb.com) instead of Neo4j.

## 2) Capabilities

**Ingested sources** (see `SOURCES.md` for live status per source):

| Source | Auth | Notes |
|---|---|---|
| Notion | Notion OAuth token | Pages, incremental sync |
| Google Calendar | Google OAuth | Events, last 180 days |
| Gmail | Google OAuth | Messages, last 30 days |
| Google Docs | Google OAuth + Drive API | Skips un-exportable docs |
| Claude Code sessions | none (local files) | Reads this project's own session transcripts |

**Entity resolution / dedup** — a three-tier resolver (`enrichment/resolver.py`):
exact-name match (fast path) → embedding similarity search via a FalkorDB vector
index → LLM confirmation for ambiguous cases. Calibrated against real
`gemini-embedding-001` output so obvious synonyms auto-merge, clearly unrelated
names are auto-rejected, and only genuinely ambiguous pairs cost an LLM call.

**Bi-temporal facts** — every edge tracks `valid_from`/`valid_until` alongside
`ingested_at`, so you can ask not just "what's true now" but "what was true then."

**Consolidation** — `consolidation/consolidate_all.py` synthesizes each entity's
episodes into a semantic summary and promotes durable facts to permanent edges,
distinguishing "this happened once" from "this is now durably true."

**Natural-language retrieval** — `retrieval/query.py` classifies a question into
one of five traversal patterns (`direct_lookup`, `neighborhood`, `path`, `impact`,
`history`), executes the matching graph traversal, scores results by recency, and
synthesizes a plain-English answer from the retrieved facts.

**Web UI** — `api.py` (FastAPI) serves a graph visualization and a query box at
`http://localhost:8000` once running.

## 3) Setup

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for FalkorDB)
- API keys: Gemini (embeddings), Concentrate AI (LLM calls — extraction,
  consolidation, query synthesis), plus OAuth credentials for whichever Google/
  Notion sources you want to ingest.

### Install dependencies

There's no pinned manifest yet — install what the code imports:

```bash
pip install falkordb fastapi uvicorn python-dotenv httpx pydantic \
            google-auth google-auth-oauthlib google-api-python-client
```

### Environment variables

Create a `.env` in the project root:

```bash
GEMINI_API_KEY=<stored-in-1password>
CONCENTRATE_AI_API_KEY=<stored-in-1password>
NOTION_API_KEY=<stored-in-1password>
FALKORDB_HOST=localhost
FALKORDB_PORT=6380
```

(Google source auth uses OAuth token files dropped in `data/sources/` —
`token.json`, `gmail_token.json`, `gdocs_token.json` — generated on first run of
each fetcher; see each fetcher's docstring/comments for the OAuth flow.)

### Start FalkorDB (the part that's easy to get wrong)

FalkorDB's data directory inside the container is `/var/lib/falkordb/data` — **not**
the generic Redis `/data` convention. Mounting a volume at `/data` looks like it
persists your graph, but silently doesn't; the database quietly lives only in the
container's throwaway layer until you remove it. Mount the volume at the correct
path:

```bash
docker volume create falkordb_persistent

docker run -d --name falkordb \
  -p 6380:6379 \
  -v falkordb_persistent:/var/lib/falkordb/data \
  falkordb/falkordb:latest
```

`-p 6380:6379` matches the `FALKORDB_HOST`/`FALKORDB_PORT` above — port 6380 on
your machine forwards to FalkorDB's default port 6379 inside the container.

**Sanity check that persistence actually works** before you trust it with real
data — ingest something, then:

```bash
docker restart falkordb
python3 -c "from core.graph import KnowledgeGarden; print(len(KnowledgeGarden().get_all_nodes()))"
```

If the node count survives the restart, you're good. If Docker Desktop isn't
running yet, start it first (`open -a Docker` on macOS) and wait a few seconds for
the daemon before running `docker run`.

To stop/start it later:

```bash
docker stop falkordb    # pauses it, data stays in the volume
docker start falkordb   # resumes it
```

### Run the pipeline

```bash
python run_ingest.py          # pull + ingest from all configured sources
python run_consolidation.py   # fold new episodes into semantic summaries/edges
python api.py                 # or: uvicorn api:app --reload
```

Then open `http://localhost:8000` for the graph UI and query box.

### One-off maintenance scripts

- `data/backfill_embeddings.py` — embeds any existing entities that predate the
  embedding-based resolver, so dedup works uniformly across old and new data.
- `check_progress.py` — quick node/edge count snapshot.
