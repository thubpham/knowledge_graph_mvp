import subprocess
import sys
from pathlib import Path

# The nightly job: ingest, then dedup. Consolidation is NOT here -- it runs
# weekly, on its own schedule, gated by consolidate_all.get_pending_nodes()'s
# min_new_episodes threshold rather than "did anything change today."
#
# Dedup always runs after ingest, regardless of ingest's outcome. This is
# deliberate: run_ingest.py catches per-source exceptions internally and only
# exits non-zero on an actual process kill (SIGTERM/SIGINT -- see
# run_ingest.py's signal handler), so a "failure" here almost always means
# "some sources succeeded, one didn't" rather than "nothing happened." Whatever
# did get ingested is already durably in the graph (cursors only advance after
# a source finishes clean -- see the fetcher/ingester split in data/sources/),
# so there's no reason to withhold dedup just because one source had trouble.
# Dedup itself is safe to run on a partially-updated graph; it always has been.
SCRIPTS_DIR = Path(__file__).parent


def _run(script_name: str) -> int:
    print(f"\n{'=' * 60}\n{script_name}\n{'=' * 60}")
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)])
    return result.returncode


if __name__ == "__main__":
    ingest_rc = _run("run_ingest.py")
    dedup_rc = _run("run_dedup_review.py")

    print(f"\n{'=' * 60}\nnightly run complete — ingest exit {ingest_rc}, dedup exit {dedup_rc}\n{'=' * 60}")

    # Non-zero only on a real failure in either stage (a kill during ingest,
    # or dedup itself erroring out) -- surfaces to whatever's watching this
    # job's exit code (cron's mail-on-failure, launchd's StandardErrorPath,
    # etc.) without being noisy about ordinary per-source ingest errors,
    # which run_ingest.py already logs to ingest_failures.jsonl on its own.
    sys.exit(ingest_rc or dedup_rc)
