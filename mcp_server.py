"""MCP server exposing the knowledge graph to Claude Desktop.

Claude launches this as a subprocess over stdio, so it can't rely on the
project's cwd/shell env being set up — `.env` is loaded by absolute path
below, before `api` (and therefore `llm_clients`) is imported.

Reuses api.py's route functions directly (they're plain functions under
FastAPI decorators, callable without going through HTTP) instead of
duplicating the query/entity-lookup logic.

Requires FalkorDB running (`docker start falkordb`) — same requirement as
api.py / run_ingest.py.
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP

import api

mcp = FastMCP("knowledge-graph")


@mcp.tool()
def query_knowledge_graph(question: str) -> dict:
    """Ask a natural-language question against the personal knowledge graph
    (Notion, Google Calendar, Gmail, Google Docs, Claude Code sessions).
    Returns a synthesized answer plus the supporting facts it was based on."""
    result = api.run_query(api.QueryRequest(question=question))
    return result.model_dump()


@mcp.tool()
def search_entities(search: str = "") -> list[dict]:
    """Search entities (people, orgs, concepts, events) in the graph by name.
    Pass an empty string to list all entities. Use this to find an exact
    entity id before calling get_entity_detail."""
    return [e.model_dump() for e in api.list_entities(search)]


@mcp.tool()
def get_entity_detail(entity_id: str) -> dict:
    """Get full detail for one entity: summary, outgoing/incoming relations,
    and the source episodes (documents/messages/events) it was extracted
    from. entity_id must match an id returned by search_entities."""
    return api.get_node_detail(entity_id).model_dump()


if __name__ == "__main__":
    mcp.run(transport="stdio")
