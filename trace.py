"""Shared LLM-call tracing primitive.

A `Run` context manager correlates every LLM call made during one unit of
work (one ingest source, one consolidation batch, one query request) into a
single SQLite-backed record, without requiring the call chain in between
(extractor.py, resolver.py, query.py, consolidate.py, ...) to know tracing
exists at all. `llm_clients.py` looks up `current_run()` right before/after
each provider call and logs to it if one is active; outside a `with Run(...)`
block, tracing is a no-op.
"""
import contextvars
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / ".local" / "traces.db"

_current_run: contextvars.ContextVar["Run | None"] = contextvars.ContextVar("current_run", default=None)
_write_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    global _initialized
    if _initialized:
        return
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                flow TEXT,
                status TEXT,
                started_at TEXT,
                ended_at TEXT,
                meta_json TEXT,
                error TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                seq INTEGER,
                ts TEXT,
                kind TEXT,
                provider TEXT,
                model TEXT,
                prompt TEXT,
                response TEXT,
                error TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms INTEGER,
                retries INTEGER
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                seq INTEGER,
                ts TEXT,
                step TEXT,
                payload_json TEXT
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id)")
    _initialized = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Max characters of prompt/response text stored per call. Full documents can
# run tens of thousands of chars; this caps a single pathological call from
# blowing up the trace db while still keeping enough to debug with.
_MAX_TEXT_CHARS = 20_000


def _clip(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    return text[:_MAX_TEXT_CHARS] + f"...[clipped, {len(text)} chars total]"


class Run:
    """Context manager scoping one unit of work (one ingest source, one
    consolidation batch, one query request) so every LLM call made inside it
    can be correlated later."""

    def __init__(self, flow: str, meta: dict | None = None):
        self.run_id = str(uuid.uuid4())
        self.flow = flow
        self.meta = meta or {}
        self._seq = 0
        self._token = None

    def __enter__(self) -> "Run":
        _init_db()
        self._safe_write(
            "INSERT INTO runs (run_id, flow, status, started_at, meta_json) VALUES (?,?,?,?,?)",
            (self.run_id, self.flow, "running", _now(), json.dumps(self.meta, default=str)),
        )
        self._token = _current_run.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "error" if exc else "ok"
        self._safe_write(
            "UPDATE runs SET status=?, ended_at=?, error=? WHERE run_id=?",
            (status, _now(), str(exc) if exc else None, self.run_id),
        )
        _current_run.reset(self._token)
        return False  # never swallow the caller's exception

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _safe_write(self, sql: str, params: tuple):
        # Tracing must never be able to break the flow it's observing.
        try:
            with _write_lock, _connect() as conn:
                conn.execute(sql, params)
        except Exception as e:
            print(f"[trace] failed to write ({sql.split()[0]} ...): {e}")

    def log_llm_call(
        self,
        *,
        kind: str,
        provider: str,
        model: str,
        prompt: str,
        response: str | None,
        error: str | None,
        usage,
        latency_ms: int,
        retries: int,
    ):
        self._safe_write(
            "INSERT INTO llm_calls (run_id, seq, ts, kind, provider, model, prompt, response, "
            "error, prompt_tokens, completion_tokens, total_tokens, latency_ms, retries) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.run_id, self._next_seq(), _now(), kind, provider, model,
                _clip(prompt), _clip(response), error,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
                getattr(usage, "total_tokens", None) if usage else None,
                latency_ms, retries,
            ),
        )

    def event(self, step: str, **fields):
        """Append a structured event under this run. Not used by Phase 0's
        llm_clients.py wiring — reserved for flow-specific instrumentation
        (chunk-level ingest events, traversal size, etc.) added later."""
        self._safe_write(
            "INSERT INTO events (run_id, seq, ts, step, payload_json) VALUES (?,?,?,?,?)",
            (self.run_id, self._next_seq(), _now(), step, json.dumps(fields, default=str)),
        )


def current_run() -> Run | None:
    return _current_run.get()
