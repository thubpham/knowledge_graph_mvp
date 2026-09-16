import json
import os
import subprocess
import sqlite3
import sys
import threading
import time
import traceback
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from retrieval.query import query
from retrieval.scoring import score_edge
from core.schema import Edge
from prompts import SYNTHESIS_SYSTEM_PROMPT, SYNTHESIS_USER_PROMPT
from trace import Run, DB_PATH

# See the comment at its use site (run_query) for the measurement behind this.
SYNTHESIS_MAX_FACTS = 20

KG_ROOT = Path(__file__).parent
RUNS_DIR = KG_ROOT / ".local" / "runs"
LOCK_PATH = KG_ROOT / ".local" / "kg_pipeline.lock"
PIPELINE_SCRIPTS = {
    "nightly": KG_ROOT / "scripts" / "nightly.sh",
    "weekly": KG_ROOT / "scripts" / "weekly.sh",
}

app = FastAPI(title="Knowledge Graph Query UI")

_kg: KnowledgeGarden | None = None
_client: LLMClient | None = None

# In-memory job store for async queries (see POST /query/async). This is a
# single-process local tool started via `uvicorn api:app` -- no separate
# worker, no need to survive a restart -- so a plain dict is sufficient; the
# job disappears if the server restarts mid-query, same as the query itself
# would (there's no result to recover either way). Guarded by a lock since
# background query threads and polling requests touch it concurrently.
_query_jobs: dict[str, dict] = {}
_query_jobs_lock = threading.Lock()


def get_kg() -> KnowledgeGarden:
    global _kg
    if _kg is None:
        _kg = KnowledgeGarden()
    return _kg


def get_client() -> LLMClient:
    global _client
    if _client is None:
        # Intent parsing (query()) and answer synthesis (SYNTHESIS_*_PROMPT below)
        # both go through this client -> Groq's 70B model by convention, same
        # rationale as extraction/consolidation. No hardcoded fallback here —
        # .env is the single source of truth; if unset, LLMClient falls
        # through to LLM_PROVIDER, then "gemini". See llm_clients.py and
        # IMPROVEMENTS.md's Provider Routing section.
        _client = LLMClient(provider=os.getenv("QUERY_LLM_PROVIDER"))
    return _client


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class EdgeResult(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    fact: str
    score: float
    valid_from: str | None
    valid_until: str | None
    confidence: float


class QueryResponse(BaseModel):
    question: str
    answer: str | None = None
    results: list[EdgeResult]
    error: str | None = None
    debug: dict | None = None


class EntityResult(BaseModel):
    id: str
    name: str
    type: str
    summary: str | None


class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    summary: str | None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    fact: str
    score: float
    valid_from: str | None
    valid_until: str | None
    confidence: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class EpisodeResult(BaseModel):
    id: str
    text: str
    source_type: str | None
    source_id: str | None
    reference_time: str | None
    metadata: dict | None = None


class NodeDetail(BaseModel):
    id: str
    name: str
    type: str
    summary: str | None
    outgoing: list[GraphEdge]
    incoming: list[GraphEdge]
    episodes: list[EpisodeResult]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "ui" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/entities", response_model=list[EntityResult])
def list_entities(q: str = ""):
    """List all entities, optionally filtered by name substring."""
    kg = get_kg()
    nodes = kg.get_all_nodes()
    if q:
        q_lower = q.lower()
        nodes = [n for n in nodes if q_lower in n.name.lower() or q_lower in n.id.lower()]
    nodes.sort(key=lambda n: n.name.lower())
    return [EntityResult(id=n.id, name=n.name, type=n.type, summary=n.summary) for n in nodes]


def _to_graph_edge(e: Edge, now: datetime) -> GraphEdge:
    return GraphEdge(
        id=e.id,
        source=e.source,
        target=e.target,
        relation=e.relation,
        fact=e.fact,
        score=round(score_edge(e, now), 4),
        valid_from=e.valid_from.isoformat() if e.valid_from else None,
        valid_until=e.valid_until.isoformat() if e.valid_until else None,
        confidence=e.confidence or 1.0,
    )


@app.get("/graph", response_model=GraphResponse)
def get_graph():
    """Full graph snapshot for rendering: all entities + active, non-mention edges."""
    kg = get_kg()
    now = datetime.now()
    nodes = kg.get_all_nodes()
    edges = [
        e for e in kg.get_all_edges()
        if e.relation != "MENTIONED_IN" and score_edge(e, now) != 0.0
    ]
    return GraphResponse(
        nodes=[GraphNode(id=n.id, name=n.name, type=n.type, summary=n.summary) for n in nodes],
        edges=[_to_graph_edge(e, now) for e in edges],
    )


@app.get("/node/{node_id}", response_model=NodeDetail)
def get_node_detail(node_id: str):
    kg = get_kg()
    now = datetime.now()
    node = kg.get_node(node_id)
    if node is None:
        return NodeDetail(id=node_id, name=node_id, type="unknown", summary=None,
                           outgoing=[], incoming=[], episodes=[])

    outgoing = [_to_graph_edge(e, now) for e in kg.get_outgoing_edges(node_id) if e.relation != "MENTIONED_IN"]
    incoming = [_to_graph_edge(e, now) for e in kg.get_incoming_edges(node_id) if e.relation != "MENTIONED_IN"]
    episodes = [
        EpisodeResult(
            id=ep.id,
            text=ep.text,
            source_type=ep.source_type,
            source_id=ep.source_id,
            reference_time=ep.reference_time.isoformat() if ep.reference_time else None,
            metadata=ep.metadata,
        )
        for ep in kg.get_episodes_for_entity(node_id)
    ]
    return NodeDetail(
        id=node.id, name=node.name, type=node.type, summary=node.summary,
        outgoing=outgoing, incoming=incoming, episodes=episodes,
    )


def _execute_query(kg: KnowledgeGarden, client: LLMClient, question: str) -> QueryResponse:
    """The actual query logic, run INSIDE a `with Run(...)` block by both
    callers below (the synchronous /query and the background thread started
    by /query/async) -- pulled out once so the two entry points can't drift.
    Never raises: any exception is caught by the caller and turned into an
    error QueryResponse, same as before this was split out."""
    now = datetime.now()
    raw = query(kg, question, client, now)

    # query() returns a dict on error/unrecognised pattern
    if isinstance(raw, dict):
        error_msg = raw.get("error", "unknown error")
        debug: dict = dict(raw)

        # If the anchor entity wasn't found, suggest close matches by name
        if raw.get("error") == "entity not found":
            anchor = raw.get("anchor_entity", "")
            suggestions = _find_similar_entities(kg, anchor)
            debug["suggestions"] = suggestions
            if suggestions:
                error_msg = (
                    f"Entity '{anchor}' not found. "
                    f"Did you mean: {', '.join(s['id'] for s in suggestions[:3])}?"
                )

        # Unsupported query type (see query_schema.py's QueryIntent.pattern):
        # there's no anchor at all, so a name-similarity "did you mean"
        # suggestion would be meaningless -- give the honest capability
        # boundary instead.
        elif raw.get("error") == "unsupported query type":
            error_msg = (
                "I can look up direct relationships, neighborhoods, paths, "
                "impact, and history for a specific named entity — not "
                "aggregate/ranking questions or unresolved self-reference "
                "yet." + (f" ({raw['reason']})" if raw.get("reason") else "")
            )

        return QueryResponse(question=question, results=[], error=error_msg, debug=debug)

    # history_traversal returns a plain dict
    if not isinstance(raw, list):
        return QueryResponse(question=question, results=[], debug=raw)

    results = []
    for e in raw:
        if not isinstance(e, Edge):
            continue
        score = score_edge(e, now)
        results.append(EdgeResult(
            id=e.id,
            source=e.source,
            target=e.target,
            relation=e.relation,
            fact=e.fact,
            score=round(score, 4),
            valid_from=e.valid_from.isoformat() if e.valid_from else None,
            valid_until=e.valid_until.isoformat() if e.valid_until else None,
            confidence=e.confidence or 1.0,
        ))

    answer = None
    if results:
        # `results` is already score-sorted descending (query.py), but was
        # passed to synthesis whole and undeduped. Measured on a real
        # query: 96 fact lines (85 unique -- the duplicates are the same
        # fact re-emitted once per triple, e.g. "2 people were active:
        # X and Y" appearing for both people's MEMBER_OF edge) cost 2543
        # prompt tokens for a 105-token, "2-4 sentences max" answer --
        # 110.8s of a 173s query, the single largest cost in the whole
        # path. Dedupe by fact text first (order-preserving), then cap:
        # SYNTHESIS_MAX_FACTS=20 ~ 600 tokens, ~4x cheaper, still well
        # above what a short-answer prompt can use.
        #
        # Note this makes the cap a COST bound, not a quality filter:
        # score_edge now discriminates by relation type and source (see
        # retrieval/scoring.py) rather than being near-flat, so "top 20 by
        # score" is closer to "the 20 best" than it used to be, but is
        # still a bound chosen for cost, not re-derived from the new
        # scoring distribution.
        seen_facts = set()
        deduped = []
        for e in results:
            if e.fact not in seen_facts:
                seen_facts.add(e.fact)
                deduped.append(e)
        facts_text = "\n".join(
            f"- {e.fact} ({e.source} {e.relation} {e.target})"
            for e in deduped[:SYNTHESIS_MAX_FACTS]
        )
        user_prompt = SYNTHESIS_USER_PROMPT.replace("{question}", question).replace("{facts}", facts_text)
        answer = client.generate_text(SYNTHESIS_SYSTEM_PROMPT, user_prompt)

    return QueryResponse(question=question, answer=answer, results=results)


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """Synchronous query -- blocks for the full duration (measured up to
    ~173s on ollama before the fixes logged in .local/IN_FLIGHT.md, ~50s
    after). Kept for backward compatibility / scripting; the UI itself uses
    POST /query/async + GET /query/async/{run_id} below, which returns
    immediately and lets the caller poll live progress instead of holding one
    HTTP connection open for minutes (a real timeout risk on top of the UX
    problem)."""
    try:
        with Run(flow="query", meta={"question": req.question}):
            return _execute_query(get_kg(), get_client(), req.question)
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)  # always log full traceback to server console
        return QueryResponse(
            question=req.question,
            results=[],
            error=str(exc),
            debug={"traceback": tb},
        )


def _run_async_query_job(run_id: str, question: str):
    """Runs in a background thread (see POST /query/async). Deliberately
    builds its OWN KnowledgeGarden/LLMClient instead of reusing the module-
    level get_kg()/get_client() singletons -- those are shared with every
    other synchronous request handler, and this thread can be mid-query while
    another request comes in on FastAPI's own thread pool. Neither class was
    written with concurrent-thread use in mind, so this sidesteps the
    question entirely rather than relying on it happening to be safe."""
    try:
        kg = KnowledgeGarden()
        client = LLMClient(provider=os.getenv("QUERY_LLM_PROVIDER"))
        with Run(flow="query", meta={"question": question}, run_id=run_id):
            result = _execute_query(kg, client, question)
        with _query_jobs_lock:
            _query_jobs[run_id] = {"status": "done", "result": result.model_dump()}
    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        with _query_jobs_lock:
            _query_jobs[run_id] = {
                "status": "done",
                "result": QueryResponse(
                    question=question, results=[], error=str(exc), debug={"traceback": tb}
                ).model_dump(),
            }


class AsyncQueryStart(BaseModel):
    run_id: str


@app.post("/query/async", response_model=AsyncQueryStart)
def start_async_query(req: QueryRequest):
    run_id = str(uuid.uuid4())
    with _query_jobs_lock:
        _query_jobs[run_id] = {"status": "running"}
    threading.Thread(target=_run_async_query_job, args=(run_id, req.question), daemon=True).start()
    return AsyncQueryStart(run_id=run_id)


@app.get("/query/async/{run_id}")
def poll_async_query(run_id: str):
    """Polled by the UI every ~1s while a query is in flight. While running,
    returns the SAME step/call waterfall data trace_dashboard.py already
    renders in the terminal (see _run_steps/_run_calls below) -- no new
    instrumentation, this just exposes what enrichment/resolver.py and
    retrieval/query.py were already writing to traces.db. Once done, returns
    the final QueryResponse under `result`."""
    with _query_jobs_lock:
        job = _query_jobs.get(run_id)
    if job is None:
        raise HTTPException(404, f"unknown or expired run_id: {run_id}")
    if job["status"] == "done":
        return {"status": "done", "result": job["result"]}
    return {
        "status": "running",
        "steps": _run_steps(run_id),
        "calls": _run_calls(run_id, limit=20),
    }


def _find_similar_entities(kg: KnowledgeGarden, anchor: str) -> list[dict]:
    """Return entities whose name or id contains the anchor string (case-insensitive)."""
    anchor_lower = _normalize(anchor)
    results = []
    for node in kg.get_all_nodes():
        node_norm = _normalize(node.name)
        if anchor_lower in node_norm or anchor_lower in node.id:
            results.append({"id": node.id, "name": node.name, "type": node.type})
    return results


# ── Observability (mirrors scripts/trace_dashboard.py's terminal view) ──────
#
# Everything below reads .local/traces.db read-only, in WAL mode (already
# configured by trace.py), so it can run alongside any number of ingest/
# consolidation/query processes that are actively writing to it -- same
# access pattern scripts/trace_dashboard.py already relies on. No new
# instrumentation: this exposes data enrichment/resolver.py, retrieval/
# query.py, and llm_clients.py were already recording.

def _trace_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _run_steps(run_id: str) -> list[dict]:
    """The per-step timing waterfall (_step() calls in retrieval/query.py,
    resolve_tier events in enrichment/resolver.py, progress events in
    consolidation/consolidate_all.py) for one run, in order."""
    if not DB_PATH.exists():
        return []
    conn = _trace_db_connect()
    try:
        rows = conn.execute(
            "SELECT step, ts, payload_json FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [
            {"step": r["step"], "ts": r["ts"], **(json.loads(r["payload_json"]) if r["payload_json"] else {})}
            for r in rows
        ]
    finally:
        conn.close()


def _run_calls(run_id: str | None, limit: int = 50) -> list[dict]:
    """LLM call feed, optionally scoped to one run. Mirrors trace_dashboard.py's
    _calls_table exactly, including the "still running" elapsed-time
    computation for in-flight calls (status="running" rows written by
    llm_clients.py before the provider call returns)."""
    if not DB_PATH.exists():
        return []
    conn = _trace_db_connect()
    try:
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
        out = []
        for c in rows:
            elapsed_s = None
            if c["status"] == "running" and c["ts"]:
                started = datetime.fromisoformat(c["ts"])
                elapsed_s = round((datetime.now(timezone.utc) - started).total_seconds())
            out.append({
                "id": c["id"], "run_id": c["run_id"], "flow": c["flow"], "ts": c["ts"],
                "kind": c["kind"], "provider": c["provider"], "model": c["model"],
                "status": c["status"], "error": c["error"],
                "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
                "total_tokens": c["total_tokens"], "latency_ms": c["latency_ms"],
                "retries": c["retries"], "elapsed_s": elapsed_s,
            })
        return out
    finally:
        conn.close()


@app.get("/observability/rollup")
def observability_rollup():
    if not DB_PATH.exists():
        return {"runs": 0, "calls": 0, "tokens": 0, "errors": 0}
    conn = _trace_db_connect()
    try:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT r.run_id) AS runs, COUNT(c.id) AS calls,
                   COALESCE(SUM(c.total_tokens), 0) AS tokens,
                   SUM(CASE WHEN c.error IS NOT NULL THEN 1 ELSE 0 END) AS errors
            FROM runs r LEFT JOIN llm_calls c ON c.run_id = r.run_id
            """
        ).fetchone()
        return {"runs": row["runs"], "calls": row["calls"], "tokens": row["tokens"], "errors": row["errors"] or 0}
    finally:
        conn.close()


@app.get("/observability/runs")
def observability_runs(flow: str | None = None, limit: int = 30):
    """Recent runs (ingest/consolidation/query), most recent first. Mirrors
    trace_dashboard.py's _runs_table."""
    if not DB_PATH.exists():
        return []
    conn = _trace_db_connect()
    try:
        where = "WHERE r.flow = ?" if flow else ""
        params = (flow, limit) if flow else (limit,)
        rows = conn.execute(
            f"""
            SELECT r.run_id, r.flow, r.status, r.started_at, r.ended_at, r.meta_json, r.error,
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
        out = []
        for r in rows:
            out.append({
                "run_id": r["run_id"], "flow": r["flow"], "status": r["status"],
                "started_at": r["started_at"], "ended_at": r["ended_at"], "error": r["error"],
                "meta": json.loads(r["meta_json"]) if r["meta_json"] else {},
                "calls": r["calls"], "tokens": r["tokens"],
                "retries": r["retries"], "errors": r["errors"] or 0,
            })
        return out
    finally:
        conn.close()


@app.get("/observability/runs/{run_id}")
def observability_run_detail(run_id: str):
    """Full drill-down for one run: the run row plus its complete step
    waterfall and call feed -- everything trace_dashboard.py's `--run`
    filter shows, in one response."""
    if not DB_PATH.exists():
        raise HTTPException(404, "no trace database yet")
    conn = _trace_db_connect()
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"unknown run_id: {run_id}")
        return {
            "run_id": row["run_id"], "flow": row["flow"], "status": row["status"],
            "started_at": row["started_at"], "ended_at": row["ended_at"], "error": row["error"],
            "meta": json.loads(row["meta_json"]) if row["meta_json"] else {},
            "steps": _run_steps(run_id),
            "calls": _run_calls(run_id, limit=500),
        }
    finally:
        conn.close()


# ── Pipeline triggers (scripts/nightly.sh, scripts/weekly.sh) ───────────────
#
# These invoke the exact same scripts a terminal would -- no separate
# scheduling or process-management logic here. scripts/lib/kg_env.sh already
# owns: the single-instance lock (kg_acquire_lock), self-detaching so the run
# outlives this HTTP request, and dated logs under .local/runs/. This layer
# only shells out and reports what the script itself printed.

def _current_pipeline_log() -> Path | None:
    """Most recent *.log under .local/runs/, by mtime. Read off disk rather
    than tracked in memory, so "what's running" reflects reality regardless
    of whether it was started from this UI, a bare `./scripts/nightly.sh` in
    a terminal, or (once wired up) cron -- one source of truth, not a second
    one this endpoint could drift from. Safe to treat the newest log as THE
    active pipeline because the lock in kg_env.sh guarantees at most one
    script is ever running at a time."""
    if not RUNS_DIR.exists():
        return None
    logs = sorted(RUNS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _lock_pid() -> int | None:
    if not LOCK_PATH.exists():
        return None
    try:
        pid = int(LOCK_PATH.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)  # signal 0: existence check only, does not kill
    except OSError:
        return None  # stale lock -- the process that held it is gone
    return pid


@app.get("/pipelines/status")
def pipelines_status(tail: int = 60):
    pid = _lock_pid()
    log_path = _current_pipeline_log()
    pipeline = log_path.stem.split("-")[0] if log_path else None
    log_lines = []
    if log_path and log_path.exists():
        log_lines = log_path.read_text(errors="replace").splitlines()[-tail:]
    return {
        "running": pid is not None,
        "pid": pid,
        "pipeline": pipeline,
        "log_path": str(log_path.relative_to(KG_ROOT)) if log_path else None,
        "log_tail": log_lines,
    }


@app.get("/pipelines/log")
def pipelines_log(tail: int = 200):
    log_path = _current_pipeline_log()
    if log_path is None or not log_path.exists():
        return {"log_path": None, "log_tail": []}
    return {
        "log_path": str(log_path.relative_to(KG_ROOT)),
        "log_tail": log_path.read_text(errors="replace").splitlines()[-tail:],
    }


@app.post("/pipelines/{name}")
def trigger_pipeline(name: str):
    if name not in PIPELINE_SCRIPTS:
        raise HTTPException(404, f"unknown pipeline '{name}'. Choose from: {', '.join(PIPELINE_SCRIPTS)}")
    script = PIPELINE_SCRIPTS[name]
    if not script.exists():
        raise HTTPException(500, f"pipeline script missing: {script}")

    # nightly.sh/weekly.sh's own detach logic prints "started" and exits 0
    # UNCONDITIONALLY, before the detached child it just spawned has even
    # checked the single-instance lock (kg_acquire_lock runs inside the
    # --foreground child, after this launcher already returned) -- confirmed
    # live: triggering a second run while one was active returned exit 0 and
    # a "started" message here, while the actual detached child immediately
    # self-aborted with "ABORT: another kg pipeline run is already active" in
    # its own log. So a 0 exit from the launcher is NOT sufficient evidence of
    # a real start; find the log file it just created and give the detached
    # child a moment to reach its lock check before trusting it.
    logs_before = set(RUNS_DIR.glob("*.log")) if RUNS_DIR.exists() else set()
    try:
        result = subprocess.run(
            ["bash", str(script)], cwd=KG_ROOT, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "pipeline launcher did not return within 30s (expected to detach almost immediately)")

    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if result.returncode != 0:
        return {"started": False, "message": output or f"exited {result.returncode}"}

    new_log = None
    for _ in range(20):  # poll up to ~2s for the detached child's log to appear
        time.sleep(0.1)
        created = (set(RUNS_DIR.glob("*.log")) if RUNS_DIR.exists() else set()) - logs_before
        if created:
            new_log = max(created, key=lambda p: p.stat().st_mtime)
            break
    if new_log is None:
        return {"started": True, "message": output}  # unexpected, but don't fail the request over it

    time.sleep(0.5)  # let the child reach kg_acquire_lock (the first real thing it does)
    lines = new_log.read_text(errors="replace").strip().splitlines()
    abort_idx = next((i for i, l in enumerate(lines) if "ABORT: another kg pipeline run is already active" in l), None)
    if abort_idx is not None:
        # Include the following "Log: ..." continuation line kg_env.sh prints
        # right after the ABORT line, not just the ABORT line alone.
        return {"started": False, "message": "\n".join(lines[abort_idx:abort_idx + 2])}
    return {"started": True, "message": output, "log_path": str(new_log.relative_to(KG_ROOT))}
