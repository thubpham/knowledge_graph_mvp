# Knowledge Graph MVP

**TL;DR:** a local, personal knowledge graph. It reads your own Notion,
Gmail, Google Calendar, Google Docs, and Claude Code session data, uses an
LLM to pull out entities and relationships, and lets you ask plain-English
questions about your own life/work — including questions about what *used
to* be true, not just what's true right now.

## 1) What this is

Most personal knowledge tools store documents. This stores *facts extracted
from* those documents, as a graph:

- **Episodes** — raw ingested text (a Notion page, an email, a calendar
  event, a chunk of a Claude Code session), each stamped with when it
  happened.
- **Entities** (nodes) — people, tools, services, teams, and concepts
  mentioned across your episodes, deduplicated so "Postgres" and
  "PostgreSQL" resolve to one node instead of two.
- **Edges** — typed relationships between entities (`MEMBER_OF`, `OWNS`,
  `ATTENDED`, `DISCUSSED`, `SENT_TO`, `AUTHORED`, etc.), each carrying the
  fact that justified it and a `valid_from`/`valid_until` window. When
  something changes (e.g. you switch teams), the old edge is closed out
  rather than deleted — the graph remembers what *used to* be true, not just
  what's current.

The pipeline runs in two stages:

1. **Ingestion** — pull raw data from a source, extract entities/relations
   with an LLM, resolve them against existing nodes (dedup), write episodic
   edges.
2. **Consolidation** — periodically fold each entity's accumulated episodes
   into a durable summary and promote recurring facts into semantic edges,
   superseding anything that's changed.

This mirrors the architecture behind [Zep/Graphiti](https://arxiv.org/abs/2501.13956)
(temporal knowledge graphs for agent memory), scaled down to a single-user
project backed by [FalkorDB](https://falkordb.com) instead of Neo4j.

## 2) How this differs from Zep

Zep is a hosted, multi-tenant memory layer built for other applications to
plug into. This project optimizes for a different point in the design
space — a single person's own data, run locally:

- **Local-first, single-user** — your graph lives in a FalkorDB container on
  your own machine. No third-party service ever holds your data.
- **Multi-source out of the box** — Notion, Gmail, Calendar, Docs, and your
  own Claude Code session transcripts all feed the *same* graph. Zep
  integrates conversational memory per-application; this ingests your
  broader digital footprint.
- **Bi-temporal by default** — every edge tracks `valid_from`/`valid_until`
  *and* `ingested_at`, so "what was true then" is a first-class query, not a
  bolt-on.
- **A resolver you can inspect and tune** — dedup is a visible three-tier
  pipeline (exact-name match → embedding similarity search → LLM
  confirmation for ambiguous cases), calibrated against real embedding
  output, not a black box.
- **Small and hackable** — a few thousand lines you can read end to end,
  with no SDK or hosting model to learn.

The trade-off: this is a single-user MVP, not a production service — see
[Known limitations](#5-known-limitations) below.

## 3) Capabilities

**Ingested sources** (see `.local/SOURCES.md` for live status per source —
local-only, gitignored, since it logs your personal ingestion history):

| Source | Auth | Notes |
|---|---|---|
| Notion | Notion OAuth token | Pages, incremental sync |
| Google Calendar | Google OAuth | Events, last 180 days |
| Gmail | Google OAuth | Messages, last 30 days |
| Google Docs | Google OAuth + Drive API | Skips un-exportable docs |
| Claude Code sessions | none (local files) | Reads this project's own session transcripts |

**Entity resolution / dedup** — a three-tier resolver (`enrichment/resolver.py`):
exact-name match (fast path) → embedding similarity search via a FalkorDB
vector index → LLM confirmation for ambiguous cases. Calibrated against real
`gemini-embedding-001` output so obvious synonyms auto-merge, clearly
unrelated names are auto-rejected, and only genuinely ambiguous pairs cost
an LLM call. The embedding step embeds `"{name}: {context}"` (a short snippet
of the source sentence the entity was mentioned in, captured during entity
extraction), not the bare name alone — short names like "Bob" or "crm" carry
too little signal on their own for the embedding model to reliably tell apart
similarly-named-but-different entities. Existing nodes created before this
change keep their name-only embeddings (not backfilled); only new nodes get
the richer signal.

**Bi-temporal facts** — every edge tracks `valid_from`/`valid_until`
alongside `ingested_at`, so you can ask not just "what's true now" but "what
was true then."

**Incremental consolidation** — `consolidation/consolidate_all.py`
synthesizes each entity's episodes into a semantic summary and promotes
durable facts to permanent edges, tracking a per-node watermark so each run
only processes episodes since the last pass (not full history every time).

**Natural-language retrieval** — `retrieval/query.py` classifies a question
into one of five traversal patterns (`direct_lookup`, `neighborhood`,
`path_finding`, `impact_traversal`, `history_traversal`), executes the
matching graph traversal, scores results by recency, and synthesizes a
plain-English answer from the retrieved facts.

**Pluggable LLM provider, per task** — `llm_clients.py` supports five
text-generation providers (`gemini` | `concentrate` | `openai` | `groq` |
`ollama`), and each pipeline stage picks its own independently via a
dedicated env var, instead of one global switch:

| Env var | Drives | Recommended provider | Why |
|---|---|---|---|
| `EXTRACTION_LLM_PROVIDER` | entity + relation extraction (`enrichment/extractor.py`) | `groq` | accuracy-sensitive, but low enough call volume (once per chunk) that Groq's free tier keeps up |
| `CONSOLIDATION_LLM_PROVIDER` | `consolidation/consolidate.py` | `groq` | multi-step reasoning (change-over-time across episodes) needs a capable model |
| `QUERY_LLM_PROVIDER` | `api.py`'s `/query` route — both intent parsing and answer synthesis | `groq` | same capability bar, low volume |
| `RESOLVER_LLM_PROVIDER` | `resolve_entity`'s ambiguous-band confirmation (`enrichment/resolver.py`) | `ollama` | highest call volume, lowest individual stakes — a wrong local call just falls back to "no match" |
| `DEDUP_LLM_PROVIDER` | offline dedup pass confirmation (`enrichment/dedup.py`, same `confirm_match()` helper as the resolver) | `ollama` | same rationale as resolver confirmation |

None of these have a hardcoded default in code — if a var is unset, that
task's client falls through to `LLM_PROVIDER` (global fallback, defaults to
`"gemini"` if that's unset too). `.env.example` documents the recommended
assignment above.

Gemini's free tier caps at 20 `generateContent` calls/day, a hard wall that
extraction (one call per chunk) and consolidation (one call per pending
entity) both hit fast — that's why Groq (`GROQ_API_KEY`, free tier, far
higher limits, model defaults to `llama-3.3-70b-versatile` via `GROQ_MODEL`)
is the recommended default for those tasks instead. Ollama (`OLLAMA_MODEL`,
defaults to `llama3.1:8b`) runs entirely locally — no API key, no rate
limit — but requires the Ollama app installed and running
(`ollama serve`, or just launch the app) with the model pulled
(`ollama pull llama3.1:8b`) before `RESOLVER_LLM_PROVIDER=ollama` /
`DEDUP_LLM_PROVIDER=ollama` will work; if you'd rather skip that local
setup, point those two at `groq`/`gemini`/`openai` instead.

**Embeddings always go through Gemini** regardless of any provider setting
above — there's only one embedding source wired up
(`gemini-embedding-001`), so `GEMINI_API_KEY` is required no matter which
generation providers you choose.

**Two-pass extraction** — `enrichment/extractor.py` splits entity
extraction and relation extraction into two separate LLM calls
(`extract_entities` then `extract_relations`) instead of asking one call to
produce both. Relation extraction is constrained to only connect entities
already confirmed in the first pass, rather than simultaneously inventing
and connecting them — a smaller, more reliable task per call. Costs twice
the calls, which is why this is only viable now that ingestion defaults to
a free/high-limit provider like Groq.

**Web UI** — `api.py` (FastAPI) serves a graph visualization and a query box
at `http://localhost:8000` once running. It's a single static HTML file with
inline JavaScript (`ui/index.html`) served by the same process that answers
its API calls — there's no separate front-end build step.

## 4) Setup

### Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for FalkorDB)
- API keys/tokens: a Gemini API key (required — embeddings always go
  through Gemini regardless of other settings), a Groq API key (recommended
  default for extraction/consolidation/query — see the Pluggable LLM
  provider section below), plus optionally a Concentrate AI or OpenAI key
  and/or OAuth credentials for whichever Google/Notion sources you want to
  ingest. See `.env.example` for the full list.
- [Ollama](https://ollama.com) installed and running locally, with
  `llama3.1:8b` pulled — only needed if you use the recommended
  `RESOLVER_LLM_PROVIDER=ollama`/`DEDUP_LLM_PROVIDER=ollama` setting;
  otherwise skippable (point those at another provider instead).

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Environment variables

```bash
cp .env.example .env
```

Then open `.env` and fill in your real values. `.env` is already listed in
`.gitignore` — never commit it.

Create the local-state directory (gitignored, holds everything
source-of-truth shouldn't):

```bash
mkdir -p .local/sources
```

Google source auth needs a downloaded OAuth client secret placed at
`.local/sources/credentials.json` — the per-source token files
(`token.json`, `gmail_token.json`, `gdocs_token.json`) are then generated
there automatically on first run of each fetcher; see each fetcher's
docstring/comments for the OAuth flow.

### Step 3 — Start FalkorDB (the part that's easy to get wrong)

FalkorDB's data directory inside the container is `/var/lib/falkordb/data` —
**not** the generic Redis `/data` convention. Mounting a volume at `/data`
looks like it persists your graph, but silently doesn't; the database
quietly lives only in the container's throwaway layer until you remove it.

Run this **once** to create a persistent volume and start the container:

```bash
docker volume create falkordb_persistent

docker run -d --name falkordb \
  -p 6380:6379 \
  -v falkordb_persistent:/var/lib/falkordb/data \
  falkordb/falkordb:latest
```

- `-p 6380:6379` maps container port 6379 (FalkorDB's default) to port 6380
  on your machine — matches `FALKORDB_HOST`/`FALKORDB_PORT` in `.env.example`.
- `-v falkordb_persistent:/var/lib/falkordb/data` mounts the named volume at
  the *correct* path, so data survives container restarts.

**Sanity-check that persistence actually works** before trusting it with
real data — ingest something (Step 5 below), then:

```bash
docker restart falkordb
python3 -c "from core.graph import KnowledgeGarden; print(len(KnowledgeGarden().get_all_nodes()))"
```

If the node count survives the restart, you're good.

After this one-time setup, your day-to-day commands are just:

```bash
docker start falkordb    # resume it — data is already in the volume
docker stop falkordb     # pause it — data stays in the volume
```

### Step 4 — Run the pipeline

```bash
python scripts/run_ingest.py          # pull + ingest from all configured sources
python scripts/run_consolidation.py   # fold new episodes into semantic summaries/edges
uvicorn api:app --reload              # start the web UI + API
```

Then open `http://localhost:8000` for the graph UI and query box.

### One-off maintenance scripts

All entry-point scripts live in `scripts/`:

- `scripts/run_canonicalize_unmapped.py` — rule-matches entities/relations
  that fell outside the extraction vocabulary (logged in
  `.local/unmapped_log.jsonl`) back into the graph.
- `scripts/backfill_embeddings.py` — embeds any existing entities that
  predate the embedding-based resolver, so dedup works uniformly across old
  and new data.
- `scripts/check_progress.py` — quick node/edge count + consolidation
  progress snapshot.
- `scripts/smoke_test_consolidation.py` — manual smoke check for the
  ingest → consolidate flow against a handful of hardcoded episodes (not an
  automated test suite).
- `notebooks/06_gliner_benchmark.ipynb` — standalone benchmark comparing the
  current LLM extraction pipeline against GLiNER+GLiREL (off-the-shelf,
  zero-training NER/relation-extraction models) on real ingested episodes,
  against hand-labeled ground truth. Read-only against the graph; doesn't
  touch any pipeline file. Not yet run to completion — see
  `IMPROVEMENTS.md`'s "Extraction & Entity-Resolution Accuracy" section for
  the research context and adoption decision criteria. Requires
  `pip install gliner glirel` (not in `requirements.txt`, since this is an
  experimental/optional benchmark, not a pipeline dependency).

## 5) Known limitations

- **Single-user only** — no multi-tenancy, no auth on the API or UI. Don't
  expose port 8000 beyond your own machine.
- **No automated tests yet.**
- **No pinned dependency versions** — `requirements.txt` lists package names
  without version pins; if something breaks after a fresh install, check for
  upstream breaking changes first.
- **No scheduled ingestion** — `scripts/run_ingest.py`/`scripts/run_consolidation.py` are run
  manually; there's no cron/nightly job wired up yet.

## 6) Troubleshooting

- **"Connection refused" talking to FalkorDB** — Docker Desktop isn't
  running. Start it first (`open -a Docker` on macOS) and wait a few seconds
  for the daemon before running `docker start falkordb`.
- **Graph is empty after a `docker restart`** — the volume was mounted at
  the wrong path. Re-check the `-v` flag in Step 3 points at
  `/var/lib/falkordb/data`, not `/data`.
- **OAuth errors on Gmail/Calendar/Docs** — delete the relevant token file in
  `data/sources/` (`token.json`, `gmail_token.json`, `gdocs_token.json`) to
  force re-authentication on the next fetch.
- **`uvicorn: command not found`** — make sure you ran
  `pip install -r requirements.txt` in the same environment you're running
  commands from.
