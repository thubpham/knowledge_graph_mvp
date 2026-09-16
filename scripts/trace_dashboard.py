"""Live terminal dashboard over .local/traces.db (see trace.py).

Run this in a spare terminal while an ingest/query/consolidation script runs
elsewhere — like `htop` for LLM calls. Read-only; never writes to the db, so
it can run alongside any number of real processes without interfering
(trace.py already configures SQLite in WAL mode for concurrent access).

Usage:
    python scripts/trace_dashboard.py
    python scripts/trace_dashboard.py --run <run_id>   # focus the call feed on one run
    python scripts/trace_dashboard.py --interval 2 --limit 20
"""
import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from trace import DB_PATH

_STATUS_STYLE = {"ok": "green", "error": "red", "running": "yellow"}


def _clip(s, n=60):
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _connect():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _waiting_message() -> Text:
    return Text.from_markup(
        f"\n[yellow]Waiting for trace data...[/yellow]\n\n"
        f"No db found yet at [bold]{DB_PATH}[/bold].\n"
        f"Start an ingest, query, or consolidation run in another terminal — "
        f"this view will pick it up automatically.\n"
    )


def _rollup_header(conn) -> Text:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT r.run_id) AS runs,
               COUNT(c.id) AS calls,
               COALESCE(SUM(c.total_tokens), 0) AS tokens,
               SUM(CASE WHEN c.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM runs r LEFT JOIN llm_calls c ON c.run_id = r.run_id
        """
    ).fetchone()
    return Text.from_markup(
        f"[bold]Trace Dashboard[/bold]  "
        f"runs=[cyan]{row['runs']}[/cyan]  "
        f"calls=[cyan]{row['calls']}[/cyan]  "
        f"tokens=[cyan]{row['tokens']}[/cyan]  "
        f"errors=[red]{row['errors'] or 0}[/red]   "
        f"[dim](refreshing — Ctrl+C to quit)[/dim]"
    )


def _runs_table(conn, limit: int, flow: str | None) -> Table:
    table = Table(title="Recent runs", expand=True)
    for col in ("started_at", "flow", "status", "calls", "tokens", "retries", "errors", "run_id"):
        table.add_column(col)

    where = "WHERE r.flow = ?" if flow else ""
    params = (flow, limit) if flow else (limit,)
    rows = conn.execute(
        f"""
        SELECT r.run_id, r.flow, r.status, r.started_at,
               COUNT(c.id) AS calls,
               COALESCE(SUM(c.total_tokens), 0) AS tokens,
               COALESCE(SUM(c.retries), 0) AS retries,
               SUM(CASE WHEN c.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM runs r LEFT JOIN llm_calls c ON c.run_id = r.run_id
        {where}
        GROUP BY r.run_id
        ORDER BY r.started_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    for r in rows:
        style = _STATUS_STYLE.get(r["status"], "white")
        table.add_row(
            r["started_at"], r["flow"], Text(r["status"], style=style),
            str(r["calls"]), str(r["tokens"]), str(r["retries"]), str(r["errors"] or 0),
            r["run_id"],
        )
    return table


def _calls_table(conn, limit: int, run_id: str | None) -> Table:
    title = f"Live call feed (run={run_id})" if run_id else "Live call feed (all runs)"
    table = Table(title=title, expand=True)
    for col in ("time", "flow", "kind", "provider/model", "latency", "tokens (p/c/t)", "retries", "status", "prompt"):
        table.add_column(col)

    where = "WHERE c.run_id = ?" if run_id else ""
    params = (run_id, limit) if run_id else (limit,)
    rows = conn.execute(
        f"""
        SELECT c.*, r.flow FROM llm_calls c
        JOIN runs r ON r.run_id = c.run_id
        {where}
        ORDER BY c.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    for c in rows:
        if c["status"] == "running":
            style = "yellow"
            status = "RUNNING"
            started = datetime.fromisoformat(c["ts"])
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            latency = f"{elapsed:.0f}s…"
            tokens = ""
        elif c["error"] is not None:
            style = "red"
            status = "ERR"
            latency = f"{c['latency_ms']}ms"
            tokens = f"{c['prompt_tokens']}/{c['completion_tokens']}/{c['total_tokens']}"
        elif (c["retries"] or 0) > 0:
            style = "yellow"
            status = "OK"
            latency = f"{c['latency_ms']}ms"
            tokens = f"{c['prompt_tokens']}/{c['completion_tokens']}/{c['total_tokens']}"
        else:
            style = "green"
            status = "OK"
            latency = f"{c['latency_ms']}ms"
            tokens = f"{c['prompt_tokens']}/{c['completion_tokens']}/{c['total_tokens']}"
        table.add_row(
            Text(c["ts"].split("T")[-1].split(".")[0], style=style),
            c["flow"], c["kind"], f"{c['provider']}/{c['model']}",
            latency, tokens, str(c["retries"] or 0),
            Text(status, style=style),
            _clip(c["error"] or c["prompt"]),
        )
    return table


def _progress_panel(conn) -> Text | None:
    """Latest `progress` event per still-running run (see
    `consolidation/consolidate_all.py`'s `Run.event("progress", ...)` calls) —
    the current/total/ETA that used to be print-only."""
    rows = conn.execute(
        """
        SELECT e.run_id, r.flow, e.payload_json
        FROM events e
        JOIN runs r ON r.run_id = e.run_id
        WHERE r.status = 'running' AND e.step = 'progress'
          AND e.id = (SELECT MAX(id) FROM events WHERE run_id = e.run_id AND step = 'progress')
        ORDER BY e.id DESC
        """
    ).fetchall()
    if not rows:
        return None
    lines = ["[bold]In-flight progress[/bold]"]
    for r in rows:
        p = json.loads(r["payload_json"])
        pct = 100 * p["current"] / p["total"] if p.get("total") else 0
        eta_m = p.get("eta_seconds", 0) / 60
        lines.append(
            f"  [cyan]{r['flow']}[/cyan] {r['run_id'][:8]}  "
            f"{p['current']}/{p['total']} ({pct:.0f}%)  "
            f"ETA {eta_m:.1f}m  — {_clip(p.get('entity', ''), 40)}"
        )
    return Text.from_markup("\n".join(lines) + "\n")


def _unsupported_queries_panel(conn, limit: int) -> Text | None:
    """Questions the query classifier explicitly declined (see
    retrieval/query.py's pattern == "unsupported" branch, added because
    forcing every question into one of 5 traversal patterns was producing
    hallucinated anchor entities for aggregate/ranking and ungrounded
    first-person questions). This is the real-usage signal for deciding
    whether/what new traversal pattern is actually worth building, instead
    of guessing -- read this table before adding one."""
    rows = conn.execute(
        """
        SELECT e.ts, e.payload_json
        FROM events e
        WHERE e.step = 'query_unsupported'
        ORDER BY e.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return None
    lines = ["[bold]Unsupported queries[/bold] [dim](signal for new traversal patterns)[/dim]"]
    for r in rows:
        p = json.loads(r["payload_json"])
        ts = r["ts"].split("T")[-1].split(".")[0]
        lines.append(
            f"  [dim]{ts}[/dim]  \"{_clip(p.get('question', ''), 50)}\" "
            f"— {_clip(p.get('reason', ''), 70)}"
        )
    return Text.from_markup("\n".join(lines) + "\n")


def render(limit: int, run_id: str | None, flow: str | None):
    if not DB_PATH.exists():
        return _waiting_message()
    conn = _connect()
    try:
        parts = [_rollup_header(conn)]
        progress = _progress_panel(conn)
        if progress:
            parts.append(progress)
        unsupported = _unsupported_queries_panel(conn, limit)
        if unsupported:
            parts.append(unsupported)
        parts.append(_runs_table(conn, limit, flow))
        parts.append(_calls_table(conn, limit, run_id))
        return Group(*parts)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=1.5, help="Refresh interval in seconds")
    parser.add_argument("--limit", type=int, default=12, help="Rows per table")
    parser.add_argument("--run", help="Filter the call feed to a single run_id")
    parser.add_argument("--flow", help="Filter the runs table to a single flow (ingest/query/consolidation)")
    args = parser.parse_args()

    try:
        with Live(render(args.limit, args.run, args.flow), screen=True, refresh_per_second=4) as live:
            while True:
                time.sleep(args.interval)
                live.update(render(args.limit, args.run, args.flow))
    except KeyboardInterrupt:
        pass
