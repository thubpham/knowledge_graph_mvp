#!/bin/bash
# Nightly trigger: one command, then walk away.
#
#   ./scripts/nightly.sh                 # detaches, survives closing the terminal
#   ./scripts/nightly.sh --foreground    # stay attached and watch it
#
# Does, in order: hold the machine awake -> prep/verify disk -> start Docker +
# FalkorDB + Ollama -> verify credentials and source tokens -> ingest (Notion,
# Calendar, Gmail, Docs) -> dedup/alias pass. Heartbeat notification every 30
# minutes, everything logged to .local/runs/, single-instance locked.
#
# Consolidation is deliberately NOT here -- that's scripts/weekly.sh.

set -uo pipefail

KG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$KG_DIR/scripts/lib/kg_env.sh"

FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --foreground|-f) FOREGROUND=1 ;;
    --help|-h)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

mkdir -p "$KG_RUNS"
STAMP="$(date '+%Y-%m-%d-%H%M%S')"
KG_LOGFILE="$KG_RUNS/nightly-$STAMP.log"

# Re-exec detached so the run survives the terminal closing. KG_DETACHED guards
# against looping. Done here rather than telling the user to remember nohup --
# an overnight run dying because a laptop lid closed a terminal is exactly the
# kind of thing this script exists to prevent.
if [ "$FOREGROUND" -eq 0 ] && [ "${KG_DETACHED:-0}" != "1" ]; then
  KG_DETACHED=1 nohup "${BASH_SOURCE[0]}" --foreground >"$KG_LOGFILE" 2>&1 &
  detached_pid=$!
  disown 2>/dev/null || true
  echo "Nightly run started (pid $detached_pid), detached — safe to close this terminal."
  echo "  log:    $KG_LOGFILE"
  echo "  follow: tail -f $KG_LOGFILE"
  echo "  stop:   kill $detached_pid"
  exit 0
fi

# When detached, stdout is already the logfile; when foreground, tee into it.
if [ -t 1 ]; then
  exec > >(tee -a "$KG_LOGFILE") 2>&1
fi

cleanup() {
  local rc=$?
  kg_stop_heartbeat
  kg_release_lock
  exit $rc
}
trap cleanup EXIT
trap '' HUP   # survive terminal hangup even in foreground mode

kg_section "KG NIGHTLY — $(date '+%A %Y-%m-%d %H:%M:%S')"
kg_log "Log: $KG_LOGFILE"
kg_log "Python: $KG_PYTHON"

kg_acquire_lock || exit 1

if ! kg_bootstrap "the nightly ingest"; then
  kg_log "FAILED during bring-up — nothing was ingested."
  kg_notify "KG nightly FAILED" "Bring-up failed — see $(basename "$KG_LOGFILE")"
  exit 1
fi

FAIL_BASELINE="$(kg_failure_baseline)"
kg_log "Failure-log baseline: $FAIL_BASELINE entries"
kg_notify "KG nightly" "Started — ingesting Notion, Calendar, Gmail, Docs"
kg_start_heartbeat "$KG_LOGFILE" "nightly"

cd "$KG_DIR"

kg_section "STAGE 1/2 — ingest"
INGEST_START=$(date +%s)
"$KG_PYTHON" -u scripts/run_ingest.py
INGEST_RC=$?
INGEST_MIN=$(( ($(date +%s) - INGEST_START) / 60 ))
kg_log "Ingest finished in ${INGEST_MIN}m (exit $INGEST_RC)"

# Dedup runs regardless of ingest's exit code. run_ingest.py handles per-source
# errors internally and only exits non-zero on an actual kill, so a bad exit
# here usually means "some sources worked, one didn't" -- and whatever did land
# is already committed (cursors only advance on a clean source). Dedup is safe
# on a partially-updated graph.
kg_section "STAGE 2/2 — dedup / alias pass"
DEDUP_START=$(date +%s)
"$KG_PYTHON" -u scripts/run_dedup_review.py
DEDUP_RC=$?
DEDUP_MIN=$(( ($(date +%s) - DEDUP_START) / 60 ))
kg_log "Dedup finished in ${DEDUP_MIN}m (exit $DEDUP_RC)"

kg_stop_heartbeat

kg_section "SUMMARY"
grep '✓ .* done\|✗ .* aborted' "$KG_LOGFILE" 2>/dev/null || kg_log "(no per-source summary lines found)"
kg_failure_report "$FAIL_BASELINE"

"$KG_PYTHON" -c "
import sys
sys.path.insert(0, '$KG_DIR')
from core.graph import KnowledgeGarden
kg = KnowledgeGarden()
rs = kg._graph.query('MATCH (e:Episode) RETURN e.source_type, count(*)').result_set
eps = {r[0]: r[1] for r in rs}
edges = kg._graph.query('MATCH ()-[e:EDGE]->() RETURN count(*)').result_set[0][0]
print('  episodes:', eps)
print('  total episodes:', sum(eps.values()), '| nodes:', len(kg.get_all_nodes()), '| edges:', edges)
" 2>&1

kg_log "Nightly complete — ingest exit $INGEST_RC, dedup exit $DEDUP_RC, ${KG_NEW_FAILURES:-0} new failure(s)"

if [ "$INGEST_RC" -ne 0 ] || [ "$DEDUP_RC" -ne 0 ]; then
  kg_notify "KG nightly FAILED" "ingest=$INGEST_RC dedup=$DEDUP_RC — see $(basename "$KG_LOGFILE")"
  exit 1
fi
kg_notify "KG nightly done" "Ingest ${INGEST_MIN}m + dedup ${DEDUP_MIN}m, ${KG_NEW_FAILURES:-0} new failure(s)"
