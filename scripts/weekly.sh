#!/bin/bash
# Weekly trigger: one command, then walk away.
#
#   ./scripts/weekly.sh                    # detaches, survives closing the terminal
#   ./scripts/weekly.sh --foreground       # stay attached and watch it
#   ./scripts/weekly.sh --min-episodes 3   # override the consolidation threshold
#
# Same bring-up as nightly.sh (awake, disk, Docker, FalkorDB, Ollama,
# credentials), then runs consolidation only. Consolidation folds each entity's
# new episodes into an updated summary plus resolved semantic edges, and is
# gated by --min-episodes: an entity is skipped unless it has accumulated at
# least that many new episodes since it was last consolidated.

set -uo pipefail

KG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$KG_DIR/scripts/lib/kg_env.sh"

FOREGROUND=0
MIN_EPISODES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --foreground|-f) FOREGROUND=1 ;;
    --min-episodes) shift; MIN_EPISODES="${1:-}" ;;
    --min-episodes=*) MIN_EPISODES="${1#*=}" ;;
    --help|-h)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
  shift
done

mkdir -p "$KG_RUNS"
STAMP="$(date '+%Y-%m-%d-%H%M%S')"
KG_LOGFILE="$KG_RUNS/weekly-$STAMP.log"

if [ "$FOREGROUND" -eq 0 ] && [ "${KG_DETACHED:-0}" != "1" ]; then
  if [ -n "$MIN_EPISODES" ]; then
    KG_DETACHED=1 nohup "${BASH_SOURCE[0]}" --foreground --min-episodes "$MIN_EPISODES" >"$KG_LOGFILE" 2>&1 &
  else
    KG_DETACHED=1 nohup "${BASH_SOURCE[0]}" --foreground >"$KG_LOGFILE" 2>&1 &
  fi
  detached_pid=$!
  disown 2>/dev/null || true
  echo "Weekly consolidation started (pid $detached_pid), detached — safe to close this terminal."
  echo "  log:    $KG_LOGFILE"
  echo "  follow: tail -f $KG_LOGFILE"
  echo "  stop:   kill $detached_pid"
  exit 0
fi

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
trap '' HUP

kg_section "KG WEEKLY CONSOLIDATION — $(date '+%A %Y-%m-%d %H:%M:%S')"
kg_log "Log: $KG_LOGFILE"
kg_log "Python: $KG_PYTHON"

kg_acquire_lock || exit 1

if ! kg_bootstrap "the weekly consolidation"; then
  kg_log "FAILED during bring-up — nothing was consolidated."
  kg_notify "KG weekly FAILED" "Bring-up failed — see $(basename "$KG_LOGFILE")"
  exit 1
fi

FAIL_BASELINE="$(kg_failure_baseline)"
kg_log "Failure-log baseline: $FAIL_BASELINE entries"
kg_notify "KG weekly" "Started — consolidating"
kg_start_heartbeat "$KG_LOGFILE" "weekly"

cd "$KG_DIR"

kg_section "CONSOLIDATION"
CONS_START=$(date +%s)
if [ -n "$MIN_EPISODES" ]; then
  kg_log "Threshold override: --min-episodes $MIN_EPISODES"
  "$KG_PYTHON" -u scripts/run_consolidation.py --min-episodes "$MIN_EPISODES"
else
  kg_log "Using run_consolidation.py's default --min-episodes"
  "$KG_PYTHON" -u scripts/run_consolidation.py
fi
CONS_RC=$?
CONS_MIN=$(( ($(date +%s) - CONS_START) / 60 ))
kg_log "Consolidation finished in ${CONS_MIN}m (exit $CONS_RC)"

kg_stop_heartbeat

kg_section "SUMMARY"
# consolidate_all.py aborts early after 5 consecutive errors, which means an
# API/billing problem rather than bad data -- surface that specifically, since
# it looks like a normal finish otherwise.
if grep -q 'consecutive errors' "$KG_LOGFILE" 2>/dev/null; then
  kg_log "⚠ Consolidation ABORTED EARLY on consecutive errors — likely an API/billing issue, not bad data."
fi
grep 'Consolidated:\|Errors:\|Edges resolved\|Still unresolved' "$KG_LOGFILE" 2>/dev/null || kg_log "(no summary lines found)"
kg_failure_report "$FAIL_BASELINE"

kg_log "Weekly complete — exit $CONS_RC, ${KG_NEW_FAILURES:-0} new failure(s)"

if [ "$CONS_RC" -ne 0 ]; then
  kg_notify "KG weekly FAILED" "Consolidation exit $CONS_RC — see $(basename "$KG_LOGFILE")"
  exit 1
fi
kg_notify "KG weekly done" "Consolidation ${CONS_MIN}m, ${KG_NEW_FAILURES:-0} new failure(s)"
