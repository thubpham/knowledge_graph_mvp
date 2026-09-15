"""Per-step timing waterfall for a single query flow (see trace.py, retrieval/query.py).

Merges the two things `Run` already records — LLM call latency (`llm_calls`,
via `_traced_call()` in llm_clients.py) and query-flow phase timing
(`events`, via `_step()` in retrieval/query.py) — into one chronological
breakdown, so it's visible end-to-end where a query actually spent its time:
intent parsing, entity lookup, graph traversal, scoring, answer synthesis.

Usage:
    python scripts/query_trace.py                # most recent query run
    python scripts/query_trace.py --run <run_id>
    python scripts/query_trace.py --list          # list recent query runs to pick a --run from
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trace import DB_PATH


def _connect():
    if not DB_PATH.exists():
        print(f"No trace db at {DB_PATH} yet — run a query through api.py first.")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_query_run(conn) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM runs WHERE flow='query' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def list_runs(limit: int = 15):
    conn = _connect()
    rows = conn.execute(
        "SELECT run_id, started_at, status, meta_json FROM runs "
        "WHERE flow='query' ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        print("No query runs recorded yet.")
        return
    for r in rows:
        question = json.loads(r["meta_json"] or "{}").get("question", "?")
        print(f"{r['started_at']}  {r['status']:<8}  {r['run_id']}  {question!r}")


def show(run_id: str):
    conn = _connect()
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        print(f"No run found with id {run_id}")
        return

    question = json.loads(run["meta_json"] or "{}").get("question", "?")
    print(f"Question: {question!r}")
    print(f"Status:   {run['status']}   started: {run['started_at']}\n")

    calls = conn.execute(
        "SELECT seq, kind, provider, model, latency_ms, error FROM llm_calls "
        "WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()
    events = conn.execute(
        "SELECT seq, step, payload_json FROM events WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()

    # Merge both sources into one chronological list by seq (each Run has its
    # own seq counter shared across llm_calls and events insert order).
    rows = []
    for c in calls:
        label = c["kind"] or "llm_call"
        if c["error"]:
            label += " (error)"
        rows.append((c["seq"], f"[llm] {label}", c["latency_ms"], f"{c['provider']}/{c['model']}"))
    for e in events:
        payload = json.loads(e["payload_json"] or "{}")
        ms = payload.pop("duration_ms", None)
        detail = ", ".join(f"{k}={v}" for k, v in payload.items())
        rows.append((e["seq"], f"[step] {e['step']}", ms, detail))
    rows.sort(key=lambda r: r[0])

    total_ms = sum(r[2] for r in rows if isinstance(r[2], (int, float)))

    header = f"{'step':<28} {'ms':>8} {'% of total':>10}   detail"
    print(header)
    print("-" * len(header))
    for _, label, ms, detail in rows:
        ms_str = str(ms) if ms is not None else "?"
        pct = f"{ms / total_ms * 100:.0f}%" if total_ms and isinstance(ms, (int, float)) else ""
        print(f"{label:<28} {ms_str:>8} {pct:>10}   {detail}")
    print("-" * len(header))
    print(f"{'TOTAL':<28} {total_ms:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Show a specific run_id (default: most recent query run)")
    parser.add_argument("--list", action="store_true", help="List recent query runs")
    parser.add_argument("--limit", type=int, default=15, help="Rows for --list")
    args = parser.parse_args()

    if args.list:
        list_runs(args.limit)
    else:
        run_id = args.run
        if run_id is None:
            conn = _connect()
            run_id = _latest_query_run(conn)
            if run_id is None:
                print("No query runs recorded yet.")
                sys.exit(0)
        show(run_id)
