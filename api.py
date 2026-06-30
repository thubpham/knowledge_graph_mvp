import sys
import traceback
import re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from retrieval.query import query
from retrieval.scoring import score_edge
from core.schema import Edge
from prompts import SYNTHESIS_PROMPT

app = FastAPI(title="Knowledge Graph Query UI")

_kg: KnowledgeGarden | None = None
_client: LLMClient | None = None


def get_kg() -> KnowledgeGarden:
    global _kg
    if _kg is None:
        _kg = KnowledgeGarden()
    return _kg


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
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


@app.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    try:
        kg = get_kg()
        client = get_client()
        now = datetime.now()

        raw = query(kg, req.question, client, now)

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

            return QueryResponse(question=req.question, results=[], error=error_msg, debug=debug)

        # history_traversal returns a plain dict
        if not isinstance(raw, list):
            return QueryResponse(question=req.question, results=[], debug=raw)

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
            facts_text = "\n".join(f"- {e.fact} ({e.source} {e.relation} {e.target})" for e in results)
            prompt = SYNTHESIS_PROMPT.replace("{question}", req.question).replace("{facts}", facts_text)
            answer = client.generate_text(prompt)

        return QueryResponse(question=req.question, answer=answer, results=results)

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)  # always log full traceback to server console
        return QueryResponse(
            question=req.question,
            results=[],
            error=str(exc),
            debug={"traceback": tb},
        )


def _find_similar_entities(kg: KnowledgeGarden, anchor: str) -> list[dict]:
    """Return entities whose name or id contains the anchor string (case-insensitive)."""
    anchor_lower = _normalize(anchor)
    results = []
    for node in kg.get_all_nodes():
        node_norm = _normalize(node.name)
        if anchor_lower in node_norm or anchor_lower in node.id:
            results.append({"id": node.id, "name": node.name, "type": node.type})
    return results
