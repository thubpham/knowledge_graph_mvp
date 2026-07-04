# Data Sources Tracker

Status of ingestion sources feeding the knowledge graph. Run `python run_ingest.py` to pull from all integrated sources, then `python run_consolidation.py` to fold new episodes into the graph.

## Integrated

| Source | Fetcher / Ingester | Auth | Notes |
|---|---|---|---|
| Notion | `data/sources/notion_fetcher.py` / `notion_ingester.py` | Notion OAuth token | Pages, incremental sync |
| Google Calendar | `data/sources/gcal_fetcher.py` / `gcal_ingester.py` | Google OAuth (`token.json`) | Events, last 180 days |
| Gmail | `data/sources/gmail_fetcher.py` / `gmail_ingester.py` | Google OAuth (`gmail_token.json`) | Messages, last 30 days |
| Google Docs | `data/sources/gdocs_fetcher.py` / `gdocs_ingester.py` | Google OAuth (`gdocs_token.json`), Drive API enabled | Requires Drive API enabled in GCP project; skips un-exportable docs |
| Claude Code sessions | `data/sources/claude_sessions_fetcher.py` / `claude_sessions_ingester.py` | None (local files) | Reads this project's own `~/.claude/projects/<dir>/*.jsonl` transcripts; human turns + assistant text only |

## Candidates (not yet built)

| Source | Effort | Notes |
|---|---|---|
| Sublime Text notes | Low | Local files, no auth — worth adding if notes are substantive |
| Zotero (with annotations) | Medium | Local SQLite (`~/Zotero/zotero.sqlite`) or local HTTP API; rich but narrow (reading/research habits) |
| Slack | High | Needs bot install + OAuth scopes + workspace admin approval — risky this close to a deadline |
| Claude.ai chat history | High | No accessible export/API today; skip |

## Last Run

_Update after each `run_ingest.py` pass._

| Date | Source | Fetched | Ingested | Skipped (dedup) |
|---|---|---|---|---|
| 2026-07-04 | Notion | 7 | 3 | 4 |
| 2026-07-04 | Google Calendar | 10 | 2 | 8 |
| 2026-07-04 | Gmail | 52 | 52 | 0 |
| 2026-07-04 | Google Docs | 24 | 24 | 0 |
| 2026-07-04 | Claude Code Sessions | 6 | 6 | 0 |
