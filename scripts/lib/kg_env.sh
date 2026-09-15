#!/bin/bash
# Shared environment bring-up + supervision for the manual nightly/weekly
# triggers (scripts/nightly.sh, scripts/weekly.sh). Sourced, not executed.
#
# Everything in here exists because it bit us during the 2026-07-25/26
# sessions: the machine slept mid-run (1-minute sleep setting) and killed the
# process; Ollama was down and every embed failed; Docker wasn't started; a
# second ingest was launched while one was already running, racing the fetch
# cursors; a run failed and nobody found out until hours later. Each function
# below is one of those, made automatic.
#
# Written for bash 3.2 (macOS system bash) -- no associative arrays, no ${x,,}.

set -uo pipefail

KG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KG_LOCAL="$KG_ROOT/.local"
KG_RUNS="$KG_LOCAL/runs"
KG_LOCKFILE="$KG_LOCAL/kg_pipeline.lock"

# Free-space floor. Below this we stop rather than risk dying mid-run with a
# full disk, which is a far messier failure than not starting.
KG_MIN_FREE_GB="${KG_MIN_FREE_GB:-5}"

# Prune dated run logs older than this. Keeps .local/runs from growing forever
# without throwing away the recent history you'd actually want after a bad night.
KG_LOG_RETENTION_DAYS="${KG_LOG_RETENTION_DAYS:-14}"

KG_HEARTBEAT_SECONDS="${KG_HEARTBEAT_SECONDS:-1800}"

# Resolved lazily so an activated conda/venv is picked up, with an override for
# anything unusual. A bare "python3" is not enough on its own: this repo's deps
# live in a conda env, and a detached run can inherit a different PATH.
KG_PYTHON="${KG_PYTHON:-$(command -v python3 || command -v python)}"

KG_CAFFEINATE_PID=""
KG_HEARTBEAT_PID=""
KG_LOGFILE=""
KG_LABEL=""

# ── logging / notification ───────────────────────────────────────────────────

kg_log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kg_section() {
  echo
  echo "════════════════════════════════════════════════════════════"
  kg_log "$*"
  echo "════════════════════════════════════════════════════════════"
}

# macOS notification centre. This is the standalone-script equivalent of the
# push notifications we were driving from the agent session -- the point is
# that a detached overnight run can still tell you something without you
# having to go look.
kg_notify() {
  local title="$1"; shift
  local message="$*"
  osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
}

# ── single-instance lock ─────────────────────────────────────────────────────

# Neither run_ingest.py nor the fetch-cursor files have any lock of their own,
# so two overlapping runs race on .local/*_last_fetched.json and double-write
# to FalkorDB. The lock lives here rather than in the Python so it also covers
# ingest-vs-consolidation overlap, which share the graph and the resolver.
kg_acquire_lock() {
  if [ -f "$KG_LOCKFILE" ]; then
    local existing
    existing="$(cat "$KG_LOCKFILE" 2>/dev/null || echo "")"
    if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
      kg_log "ABORT: another kg pipeline run is already active (pid $existing)."
      kg_log "       Log: $(ls -t "$KG_RUNS" 2>/dev/null | head -1)"
      kg_notify "KG pipeline" "Aborted: a run is already active (pid $existing)"
      return 1
    fi
    kg_log "Clearing stale lock (pid $existing no longer running)."
    rm -f "$KG_LOCKFILE"
  fi
  echo "$$" > "$KG_LOCKFILE"
  return 0
}

kg_release_lock() {
  # Only remove a lock we actually own, so a stale-lock takeover by another run
  # doesn't get its lock deleted by our exit trap.
  if [ -f "$KG_LOCKFILE" ] && [ "$(cat "$KG_LOCKFILE" 2>/dev/null)" = "$$" ]; then
    rm -f "$KG_LOCKFILE"
  fi
}

# ── stay awake ───────────────────────────────────────────────────────────────

# System sleep on this machine is set to 1 minute, which silently killed a
# multi-hour run. -w $$ ties caffeinate's lifetime to ours, so it can never
# outlive the run and leave the machine permanently awake. Deliberately not
# touching pmset: that's a persistent system-wide change, this is scoped.
kg_start_caffeinate() {
  caffeinate -dis -w $$ &
  KG_CAFFEINATE_PID=$!
  kg_log "Holding display+system awake for the duration (caffeinate pid $KG_CAFFEINATE_PID)."
}

# ── dependencies ─────────────────────────────────────────────────────────────

kg_bring_up_docker() {
  if docker info >/dev/null 2>&1; then
    kg_log "Docker daemon already up."
    return 0
  fi
  kg_log "Docker daemon down — launching Docker Desktop..."
  open -a Docker || { kg_log "ERROR: could not launch Docker Desktop."; return 1; }
  local waited=0
  while ! docker info >/dev/null 2>&1; do
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -ge 120 ]; then
      kg_log "ERROR: Docker daemon did not come up within 120s."
      return 1
    fi
  done
  kg_log "Docker daemon up after ${waited}s."
}

kg_bring_up_falkordb() {
  # Capture-then-match instead of piping into `grep -q`: under `set -o pipefail`,
  # grep -q exits on its first match and closes the pipe, the upstream command
  # takes SIGPIPE (141), and pipefail reports the whole pipeline as failed even
  # though the match succeeded. Cost us a false "model not pulled" before this
  # was fixed; the docker check below had the same latent bug and only passed
  # because its output is small enough to finish writing before grep exits.
  local volumes
  volumes="$(docker volume ls --format '{{.Name}}' 2>/dev/null)"
  if ! printf '%s\n' "$volumes" | grep -qx falkordb_persistent; then
    # Never silently recreate the data volume -- an empty-looking graph is a
    # much more confusing failure than a hard stop here.
    kg_log "ERROR: docker volume 'falkordb_persistent' is missing."
    kg_log "       Refusing to recreate it automatically (that would silently start an empty graph)."
    kg_log "       See README Step 3."
    return 1
  fi

  local status
  status="$(docker ps -a --filter name=falkordb --format '{{.Status}}' | head -1)"
  if [ -z "$status" ]; then
    kg_log "ERROR: no 'falkordb' container exists. See README Step 3 for the docker run command"
    kg_log "       (the -v mount path must be /var/lib/falkordb/data, not /data)."
    return 1
  fi
  case "$status" in
    Up*) kg_log "FalkorDB already running ($status)." ;;
    *)   kg_log "FalkorDB stopped ($status) — starting..."
         docker start falkordb >/dev/null || { kg_log "ERROR: docker start falkordb failed."; return 1; }
         sleep 3
         kg_log "FalkorDB started: $(docker ps --filter name=falkordb --format '{{.Status}}')" ;;
  esac

  # Connectivity is the thing that actually matters; a running container that
  # can't be reached is the same outage from here.
  if ! "$KG_PYTHON" -c "
import sys
sys.path.insert(0, '$KG_ROOT')
from core.graph import KnowledgeGarden
print('  FalkorDB reachable:', len(KnowledgeGarden().get_all_nodes()), 'nodes')
" 2>&1; then
    kg_log "ERROR: FalkorDB is running but not reachable from Python."
    return 1
  fi
}

kg_bring_up_ollama() {
  if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    kg_log "Ollama not responding — starting 'ollama serve'..."
    nohup ollama serve >"$KG_LOCAL/ollama_serve.log" 2>&1 &
    local waited=0
    while ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; do
      sleep 2
      waited=$((waited + 2))
      if [ "$waited" -ge 60 ]; then
        kg_log "ERROR: Ollama did not come up within 60s (see .local/ollama_serve.log)."
        return 1
      fi
    done
    kg_log "Ollama up after ${waited}s."
  else
    kg_log "Ollama already up."
  fi

  # Embeddings are a hard dependency now that EMBEDDING_PROVIDER=ollama --
  # a missing model means every single entity resolution fails, which is how
  # a whole run becomes worthless. Check the model actually referenced by .env
  # rather than assuming the default.
  # See the pipefail/SIGPIPE note in kg_bring_up_falkordb -- same reason these
  # capture `ollama list` into a variable instead of piping it into grep -q.
  local installed embed_model gen_model
  installed="$(ollama list 2>/dev/null)"

  embed_model="$(grep -E '^OLLAMA_EMBED_MODEL=' "$KG_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")"
  [ -z "$embed_model" ] && embed_model="nomic-embed-text"
  case "$installed" in
    *"$embed_model"*) kg_log "Embedding model present: $embed_model" ;;
    *) kg_log "ERROR: embedding model '$embed_model' is not pulled. Run: ollama pull $embed_model"
       return 1 ;;
  esac

  # The generation-side local model is only required if some task is routed to
  # ollama, so a miss here is a warning rather than a stop.
  gen_model="$(grep -E '^OLLAMA_MODEL=' "$KG_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")"
  if [ -n "$gen_model" ]; then
    case "$installed" in
      *"$gen_model"*) : ;;
      *) kg_log "WARNING: generation model '$gen_model' not pulled — only matters if a task routes to ollama." ;;
    esac
  fi
}

# ── credentials ──────────────────────────────────────────────────────────────

# Checks only what the configured providers actually need. Notably does NOT
# require GEMINI_API_KEY any more: embeddings moved to local nomic, so Gemini
# is only needed if a task is explicitly routed to it.
kg_check_credentials() {
  "$KG_PYTHON" - <<PYEOF
import os, sys
sys.path.insert(0, "$KG_ROOT")
from dotenv import load_dotenv
load_dotenv("$KG_ROOT/.env")

NEEDS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "concentrate": "CONCENTRATE_AI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": None,
}

def provider(task):
    return (os.getenv(f"{task}_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "gemini").lower()

tasks = ["EXTRACTION", "RESOLVER", "CONSOLIDATION", "DEDUP"]
embed = (os.getenv("EMBEDDING_PROVIDER") or "gemini").lower()

required = set()
for t in tasks:
    key = NEEDS.get(provider(t))
    if key:
        required.add(key)
if NEEDS.get(embed):
    required.add(NEEDS[embed])

print(f"  embedding provider : {embed}")
for t in tasks:
    print(f"  {t.lower():14} provider : {provider(t)}")

missing = []
for key in sorted(required):
    val = os.getenv(key)
    ok = bool(val) and not val.startswith("<")
    print(f"  {key:26} {'set' if ok else 'MISSING/placeholder'}")
    if not ok:
        missing.append(key)

if os.getenv("NOTION_API_KEY") and not os.getenv("NOTION_API_KEY").startswith("<"):
    print("  NOTION_API_KEY             set")
else:
    print("  NOTION_API_KEY             MISSING — Notion source will fail")

if missing:
    print("\\n  ABORT: missing credentials on the active path: " + ", ".join(missing))
    sys.exit(1)
PYEOF
}

kg_check_source_tokens() {
  "$KG_PYTHON" - <<PYEOF
import json
from pathlib import Path
base = Path("$KG_ROOT/.local/sources")
if not (base / "credentials.json").exists():
    print("  WARNING: credentials.json missing — all Google sources will fail.")
for fname, label in (("token.json", "Calendar"), ("gmail_token.json", "Gmail"), ("gdocs_token.json", "Docs")):
    p = base / fname
    if not p.exists():
        print(f"  WARNING: {label} token missing — would need an interactive browser consent flow,")
        print(f"           which a detached run cannot do. That source will fail.")
        continue
    d = json.loads(p.read_text())
    if d.get("refresh_token"):
        print(f"  {label:9} token ok (refresh_token present)")
    else:
        print(f"  WARNING: {label} token has no refresh_token — will fail once the access token expires.")
PYEOF
}

# ── disk ─────────────────────────────────────────────────────────────────────

# "Prep" = prune what's safe to prune, then hard-stop if still tight. Only
# touches this project's own rotated run logs; never the graph volume, never
# traces.db, never the failure logs (those are the forensic trail).
kg_prep_disk() {
  mkdir -p "$KG_RUNS"

  local pruned
  pruned="$(find "$KG_RUNS" -name '*.log' -type f -mtime "+$KG_LOG_RETENTION_DAYS" 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$pruned" -gt 0 ]; then
    find "$KG_RUNS" -name '*.log' -type f -mtime "+$KG_LOG_RETENTION_DAYS" -delete 2>/dev/null
    kg_log "Pruned $pruned run log(s) older than ${KG_LOG_RETENTION_DAYS}d."
  fi

  local free_gb
  free_gb="$(df -g / | tail -1 | awk '{print $4}')"
  kg_log "Free space: ${free_gb}Gi (floor ${KG_MIN_FREE_GB}Gi)"

  local traces_size
  traces_size="$(du -h "$KG_LOCAL/traces.db" 2>/dev/null | cut -f1)"
  [ -n "$traces_size" ] && kg_log "traces.db: $traces_size (grows unboundedly; not auto-pruned — every LLM call is in here)"

  if [ "$free_gb" -lt "$KG_MIN_FREE_GB" ]; then
    kg_log "ABORT: only ${free_gb}Gi free, need ${KG_MIN_FREE_GB}Gi."
    kg_notify "KG pipeline" "Aborted: only ${free_gb}Gi disk free"
    return 1
  fi
}

# ── heartbeat ────────────────────────────────────────────────────────────────

# Periodic "still alive, here's where it is" notification, parsed out of the
# live log. Both run_ingest.py and consolidate_all.py print [i/N] progress
# markers, so one parser covers both.
kg_start_heartbeat() {
  local logfile="$1"
  local label="$2"
  (
    while true; do
      sleep "$KG_HEARTBEAT_SECONDS"
      [ -f "$logfile" ] || continue
      hb_progress="$(grep -oE '\[[0-9]+/[0-9]+\]' "$logfile" 2>/dev/null | tail -1)"
      hb_errors="$(grep -c '→ error, skipped' "$logfile" 2>/dev/null | tr -d ' ')"
      msg="${hb_progress:-starting}"
      [ "${hb_errors:-0}" -gt 0 ] && msg="$msg — ${hb_errors} error(s)"
      # A mass-repeat of one systemic error (billing, quota) is the failure
      # mode that already wasted hours; call it out distinctly from noise.
      if grep -q 'credit balance is too low' "$logfile" 2>/dev/null; then
        msg="STUCK: out of API credits — $msg"
      elif grep -q 'RESOURCE_EXHAUSTED' "$logfile" 2>/dev/null; then
        msg="STUCK: quota exhausted — $msg"
      fi
      kg_notify "KG $label" "$msg"
    done
  ) &
  KG_HEARTBEAT_PID=$!
  kg_log "Heartbeat every $((KG_HEARTBEAT_SECONDS / 60))m (notification centre), pid $KG_HEARTBEAT_PID."
}

kg_stop_heartbeat() {
  [ -n "$KG_HEARTBEAT_PID" ] && kill "$KG_HEARTBEAT_PID" 2>/dev/null
  KG_HEARTBEAT_PID=""
}

# ── failure-log delta ────────────────────────────────────────────────────────

kg_failure_baseline() {
  wc -l < "$KG_LOCAL/ingest_failures.jsonl" 2>/dev/null | tr -d ' ' || echo 0
}

# Sets KG_NEW_FAILURES rather than echoing it: this function's real output is
# the human-readable breakdown, which needs to reach the log. Returning the
# count via stdout too would mean callers using $(...) swallow the breakdown.
KG_NEW_FAILURES=0

kg_failure_report() {
  local baseline="$1"
  local now
  now="$(kg_failure_baseline)"
  KG_NEW_FAILURES=$((now - baseline))
  if [ "$KG_NEW_FAILURES" -gt 0 ]; then
    kg_log "Failure log grew by $KG_NEW_FAILURES entr(ies) (now $now). Breakdown:"
    tail -n "$KG_NEW_FAILURES" "$KG_LOCAL/ingest_failures.jsonl" 2>/dev/null | "$KG_PYTHON" -c "
import json, sys, collections
c = collections.Counter()
for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        continue
    err = str(d.get('error', ''))[:60]
    c[(d.get('source'), err)] += 1
for (src, err), n in c.most_common():
    print(f'    {n:4}x  {src}: {err}')
"
  else
    kg_log "Failure log unchanged ($now entries) — no new failures."
  fi
}

# ── common bring-up sequence ─────────────────────────────────────────────────

# Everything that must be true before either pipeline touches an API.
kg_bootstrap() {
  local label="$1"

  kg_section "Bring-up: hold awake, disk, Docker, FalkorDB, Ollama, credentials"
  kg_start_caffeinate
  kg_prep_disk       || return 1
  kg_bring_up_docker || return 1
  kg_bring_up_falkordb || return 1
  kg_bring_up_ollama || return 1

  kg_log "Credentials on the active provider path:"
  kg_check_credentials || return 1
  kg_log "Google/Notion source tokens:"
  kg_check_source_tokens

  kg_log "Bring-up complete — everything $label needs is verified up."
}
