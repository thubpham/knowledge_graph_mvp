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
                retries INTEGER,
                status TEXT,
                ended_at TEXT
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
        # `status`/`ended_at` are new columns (in-flight call tracking) — a
        # pre-existing db file's llm_calls table predates them, and `CREATE
        # TABLE IF NOT EXISTS` above is a no-op against an already-existing
        # table, so add them explicitly if missing.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)")}
        if "status" not in existing_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN status TEXT")
        if "ended_at" not in existing_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN ended_at TEXT")
        # system_prompt/cached_tokens/cache_write_tokens are new (OpenRouter
        # prompt-caching support) — same ALTER-if-missing guard as above, so
        # a pre-existing db file's llm_calls table (which predates these
        # columns) keeps working instead of failing CREATE TABLE IF NOT
        # EXISTS's no-op-against-existing-table behavior.
        if "system_prompt" not in existing_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN system_prompt TEXT")
        if "cached_tokens" not in existing_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN cached_tokens INTEGER")
        if "cache_write_tokens" not in existing_cols:
            conn.execute("ALTER TABLE llm_calls ADD COLUMN cache_write_tokens INTEGER")
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

    def __init__(self, flow: str, meta: dict | None = None, run_id: str | None = None):
        # run_id is normally auto-generated; api.py's async query endpoint
        # needs to know the id BEFORE the Run starts (to hand it to the
        # client immediately, before the background thread that will
        # eventually open this Run has even begun), so it pre-generates one
        # and passes it in here rather than reading it back afterward.
        self.run_id = run_id or str(uuid.uuid4())
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

    def _safe_write_returning_id(self, sql: str, params: tuple) -> int | None:
        try:
            with _write_lock, _connect() as conn:
                return conn.execute(sql, params).lastrowid
        except Exception as e:
            print(f"[trace] failed to write ({sql.split()[0]} ...): {e}")
            return None

    def start_llm_call(self, *, kind: str, provider: str, model: str, prompt: str,
                        system_prompt: str | None = None) -> int | None:
        """Inserts an `llm_calls` row with `status="running"` before the
        provider call is made, so a call that's still in flight is visible
        (not indistinguishable from nothing happening) instead of only
        appearing once it finishes. Pair with `finish_llm_call()`. Returns the
        row id, or None if the write itself failed (tracing must never break
        the flow it's observing). `system_prompt` is stored separately from
        `prompt` (the user half) so two calls' static prefixes can be
        diffed directly to confirm they were actually byte-identical —
        `None` for calls with no system/user split (e.g. `embed()`)."""
        return self._safe_write_returning_id(
            "INSERT INTO llm_calls (run_id, seq, ts, kind, provider, model, prompt, system_prompt, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, self._next_seq(), _now(), kind, provider, model,
             _clip(prompt), _clip(system_prompt), "running"),
        )

    def finish_llm_call(
        self,
        call_id: int | None,
        *,
        response: str | None,
        error: str | None,
        usage,
        latency_ms: int,
        retries: int,
    ):
        if call_id is None:
            return
        self._safe_write(
            "UPDATE llm_calls SET status=?, ended_at=?, response=?, error=?, "
            "prompt_tokens=?, completion_tokens=?, total_tokens=?, "
            "cached_tokens=?, cache_write_tokens=?, latency_ms=?, retries=? "
            "WHERE id=?",
            (
                "error" if error else "ok", _now(), _clip(response), error,
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
                getattr(usage, "total_tokens", None) if usage else None,
                getattr(usage, "cached_tokens", None) if usage else None,
                getattr(usage, "cache_write_tokens", None) if usage else None,
                latency_ms, retries, call_id,
            ),
        )

    def event(self, step: str, **fields):
        """Append a structured event under this run — flow-specific
        instrumentation outside the LLM-call tracing above. Used by
        `retrieval/query.py`'s per-query timing waterfall and
        `enrichment/resolver.py`'s resolve_entity() exit-tier tagging."""
        self._safe_write(
            "INSERT INTO events (run_id, seq, ts, step, payload_json) VALUES (?,?,?,?,?)",
            (self.run_id, self._next_seq(), _now(), step, json.dumps(fields, default=str)),
        )


def current_run() -> Run | None:
    return _current_run.get()
