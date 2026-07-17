"""Read-only CLI over .local/traces.db (see trace.py).

Usage:
    python scripts/inspect_traces.py                # list recent runs
    python scripts/inspect_traces.py --run <run_id>  # show one run's LLM calls
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trace import DB_PATH


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"No trace db at {DB_PATH} yet — run something inside a `Run(...)` block first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _clip(s, n=80):
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def list_runs(limit: int = 20):
    conn = _connect()
    rows = conn.execute(
        """
        SELECT r.run_id, r.flow, r.status, r.started_at, r.meta_json,
               COUNT(c.id) AS calls,
               COALESCE(SUM(c.total_tokens), 0) AS total_tokens,
               COALESCE(SUM(c.retries), 0) AS total_retries,
               SUM(CASE WHEN c.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM runs r
        LEFT JOIN llm_calls c ON c.run_id = r.run_id
        GROUP BY r.run_id
        ORDER BY r.started_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("No runs recorded yet.")
        return
    header = f"{'started_at':<26} {'flow':<14} {'status':<8} {'calls':>6} {'tokens':>8} {'retries':>8} {'errs':>5}  run_id"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['started_at']:<26} {r['flow']:<14} {r['status']:<8} "
            f"{r['calls']:>6} {r['total_tokens']:>8} {r['total_retries']:>8} {r['errors'] or 0:>5}  "
            f"{r['run_id']} {_clip(r['meta_json'], 60)}"
        )


def show_run(run_id: str):
    conn = _connect()
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        print(f"No run found with id {run_id}")
        return
    print(f"run_id:   {run['run_id']}")
    print(f"flow:     {run['flow']}")
    print(f"status:   {run['status']}")
    print(f"started:  {run['started_at']}")
    print(f"ended:    {run['ended_at']}")
    print(f"meta:     {run['meta_json']}")
    if run["error"]:
        print(f"error:    {run['error']}")
    print()

    calls = conn.execute(
        "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    print(f"{len(calls)} LLM call(s):")
    for c in calls:
        print("-" * 80)
        print(
            f"[{c['seq']}] {c['kind']} via {c['provider']}/{c['model']}  "
            f"latency={c['latency_ms']}ms retries={c['retries']} "
            f"tokens(p/c/t)={c['prompt_tokens']}/{c['completion_tokens']}/{c['total_tokens']}"
        )
        if c["error"]:
            print(f"  error: {c['error']}")
        print(f"  prompt:   {_clip(c['prompt'], 200)}")
        print(f"  response: {_clip(c['response'], 200)}")

    events = conn.execute(
        "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    if events:
        print(f"\n{len(events)} event(s):")
        for e in events:
            print(f"  [{e['seq']}] {e['step']}: {_clip(e['payload_json'], 150)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Show details for a specific run_id")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent runs to list")
    args = parser.parse_args()

    if args.run:
        show_run(args.run)
    else:
        list_runs(args.limit)
