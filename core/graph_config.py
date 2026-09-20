import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GRAPH_NAME = "knowledge_garden"
_LOCAL_DIR = Path(__file__).parent.parent / ".local"


def graph_name() -> str:
    """Which FalkorDB graph this process reads/writes. One FalkorDB server holds
    many independent graphs, so a second knowledge base only needs a second name
    (set FALKORDB_GRAPH) -- no new container or volume."""
    return os.getenv("FALKORDB_GRAPH", DEFAULT_GRAPH_NAME)


def cursor_path(source: str) -> Path:
    """Fetch-cursor file for `source`, namespaced by graph so that a new graph
    doesn't inherit another graph's "already fetched up to X" and skip
    everything older. The default graph keeps its original flat path, so
    existing cursors keep working untouched."""
    name = graph_name()
    if name == DEFAULT_GRAPH_NAME:
        return _LOCAL_DIR / f"{source}_last_fetched.json"
    directory = _LOCAL_DIR / "graphs" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{source}_last_fetched.json"


def checkpoint_dir() -> Path:
    """Where in-flight extraction results are kept until their episode is fully
    written. Namespaced by graph like the cursors, for the same reason: a
    checkpoint made against one graph's recent-entity context shouldn't be
    resumed into another."""
    name = graph_name()
    base = _LOCAL_DIR if name == DEFAULT_GRAPH_NAME else _LOCAL_DIR / "graphs" / name
    directory = base / "extraction_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
