---
name: ingest
description: Run a full ingest pass (Notion/Gmail/Calendar/Docs) with a complete pre-flight — FalkorDB up, source tokens valid, disk space, failure-log harness wired in — before touching any API. Use when the user asks to "ingest", "run ingest", "pull in new sources", or reingest after any kind of data loss. Does NOT run consolidation — that's a separate step/process.
user-invocable: true
allowed-tools:
  - Bash
  - Read
---

# /ingest — Preflighted Ingest Run

Runs `scripts/run_ingest.py` (Notion, Gmail, Calendar, Docs — Claude Code
Sessions is deliberately excluded, see `scripts/run_ingest.py`'s comment)
only after verifying everything it depends on is actually healthy. The
overnight-incident history in `.local/IMPROVEMENTS.md` ("Infra / Ops") is
entirely preventable checks that weren't run first — this skill exists so
those checks always happen, every time, before any API call burns time or
quota.

Do not run `scripts/run_consolidation.py` as part of this skill, even if
asked in the same breath — consolidation is a separate process with its own
failure modes (the 5-consecutive-error abort in `consolidate_all.py`) and
should be invoked on its own.

## Step 1 — No ingest already running

```bash
pgrep -fl "run_ingest.py"
```

If this returns a process, **stop** — do not start a second one. Neither
`run_ingest.py` nor the fetch-cursor files (`.local/*_last_fetched.json`)
have a lock guard, so two concurrent runs will race on the same cursor files
and double-write to FalkorDB. Tell the user a run is already in progress
(show the PID/elapsed time) and ask whether to wait or kill it first.

## Step 2 — Docker daemon + FalkorDB container

```bash
docker ps >/dev/null 2>&1 && echo "daemon up" || echo "daemon down"
```

- If down: `open -a Docker` (macOS), then poll `docker ps` every few seconds
  (up to ~30s) until it succeeds. If it never comes up, stop and tell the
  user to start Docker Desktop manually.

```bash
docker volume ls --format '{{.Name}}' | grep -qx falkordb_persistent && echo "volume exists" || echo "volume MISSING"
docker ps -a --filter name=falkordb --format '{{.Names}}: {{.Status}}'
```

- **Volume missing entirely** — do not silently recreate it. Stop and point
  the user at the README's Step 3 (`docker volume create falkordb_persistent`
  + the `docker run` command with `-v falkordb_persistent:/var/lib/falkordb/data`)
  — recreating a data volume is not something to do without them noticing.
- **Container exists but stopped** — `docker start falkordb`, then confirm
  with `docker ps --filter name=falkordb --format '{{.Status}}'`.
- **Container doesn't exist but volume does** — this is recoverable without
  losing data (the volume is what holds the graph), but still confirm with
  the user before running the full `docker run ...` command from the README,
  since getting the `-v` mount path wrong (`/data` instead of
  `/var/lib/falkordb/data`) silently produces an empty-looking graph.
- **Container already running** — nothing to do.

Sanity-check connectivity once it's up:

```bash
python3 -c "from core.graph import KnowledgeGarden; print(len(KnowledgeGarden().get_all_nodes()), 'nodes reachable')"
```

If this fails, stop — nothing downstream will work either.

## Step 3 — Provider credentials for the ingest path

Ingest only exercises `EXTRACTION_LLM_PROVIDER` and `RESOLVER_LLM_PROVIDER`
(plus embeddings, always Gemini) — not `QUERY_LLM_PROVIDER` or
`CONSOLIDATION_LLM_PROVIDER`, so only check what's actually on the ingest
path.

1. Read `.env` (not `.env.example`) and resolve the effective provider for
   extraction and resolution: the task-specific var if set, else
   `LLM_PROVIDER`, else `gemini`.
2. `GEMINI_API_KEY` is required unconditionally (embeddings always use it) —
   check it's set and not still the `<your-gemini-api-key>` placeholder.
3. For each of the two resolved providers, check the credential it actually
   needs:
   - `groq` → `GROQ_API_KEY` set, not a placeholder
   - `concentrate` → `CONCENTRATE_AI_API_KEY` set
   - `openai` → `OPENAI_API_KEY` set
   - `ollama` → confirm the server is reachable and the model is pulled:
     ```bash
     curl -s -m 3 http://localhost:11434/api/tags >/dev/null && echo "ollama up" || echo "ollama DOWN"
     ollama list | grep -F "$(grep '^OLLAMA_MODEL=' .env | cut -d= -f2)"
     ```
     If Ollama's down, `ollama serve &` is an option but ask first — it's a
     long-lived background process the user may already be managing
     themselves. If the model isn't pulled, tell the user to
     `ollama pull <model>` before continuing; don't pull multi-GB models
     without asking.
   - `gemini` → already covered by the `GEMINI_API_KEY` check above.

Any missing/placeholder credential on the active path is a stop, not a
warning — the run will fail on the first extraction call otherwise, several
minutes in.

## Step 4 — Source auth tokens (Google + Notion)

```bash
ls -la .local/sources/ 2>/dev/null
```

- `credentials.json` missing → Google sources can't run at all (no OAuth
  client to start from). Point at the README's OAuth setup.
- For each of `token.json` (Calendar), `gmail_token.json`, `gdocs_token.json`
  that's present, do a light liveness check rather than assuming a file on
  disk means a valid token:
  ```bash
  python3 -c "
  import json, sys
  from pathlib import Path
  from datetime import datetime, timezone
  p = Path('.local/sources/$TOKEN_FILE')
  if not p.exists():
      print('missing — first run will trigger an interactive browser consent flow')
      sys.exit()
  data = json.loads(p.read_text())
  has_refresh = bool(data.get('refresh_token'))
  print('refresh_token present' if has_refresh else 'NO refresh_token — will need re-auth once the access token expires')
  "
  ```
  A present `refresh_token` doesn't guarantee it hasn't been revoked
  server-side (that only surfaces as `invalid_grant` on first real use, per
  the 2026-07-23/24 incident in `.local/IMPROVEMENTS.md`) — note that as a
  residual risk rather than treating the file's mere existence as "healthy."
- Any Google source missing its token file entirely: warn that
  `run_ingest.py` will pop an interactive browser window for that source on
  first use — if this is being run non-interactively (backgrounded, no
  display), that source will hang. Ask the user whether to proceed with that
  source or skip it.
- Notion: check `NOTION_API_KEY` in `.env` is set and not the placeholder if
  Notion ingestion is expected to run.

## Step 5 — Disk space

```bash
df -h / | tail -1
docker system df 2>/dev/null | grep -A1 "Local Volumes"
```

FalkorDB's data volume and `.local/traces.db` both grow unboundedly over
time. There's no hard threshold to enforce — just flag it if available space
looks tight (a rough gut-check: under a few GB free) so a run doesn't die
mid-way from a full disk, which is a much messier failure than any of the
above.

## Step 6 — Failure-log harness sanity check

Confirm the pieces added for exactly this purpose are actually present
before relying on them:

```bash
test -f core/failure_log.py && echo "failure log module present" || echo "MISSING — failures during this run will only print, not persist"
grep -l "log_ingest_failure" data/sources/*.py | wc -l
```

Expect `5` (all five ingesters). If it's missing, that's not necessarily a
stop — ingestion still runs — but tell the user their overnight-safety net
isn't wired up and ask if they want to proceed anyway.

## Step 7 — Run it

Only after Steps 1–6 pass (or the user has explicitly accepted a flagged
risk):

```bash
python scripts/run_ingest.py
```

Stream the output rather than silently waiting — the user should see
per-source progress as it happens, same as running it directly. After it
finishes, check `.local/ingest_failures.jsonl` for any new entries (compare
line count before/after Step 7) and summarize them if present, rather than
just reporting the per-source counts `run_ingest.py` already prints.

Do not chain into `scripts/run_consolidation.py` afterward — stop here and
tell the user ingest is done.
