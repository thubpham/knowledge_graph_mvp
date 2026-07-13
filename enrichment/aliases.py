"""Identity alias table: a small, hand-editable/machine-appendable map from a
normalized entity name to the id of its canonical node. Exists as its own
module (rather than living in `resolver.py` or `dedup.py`) so both can import
it without a circular dependency — `resolver.py` needs it for the alias
fast-path tier, `dedup.py` needs it to persist confirmed merges, and
`dedup.py` also imports from `resolver.py` (for the shared LLM-confirmation
helper).
"""

import json
import os
from pathlib import Path

ALIAS_PATH = Path(__file__).parent.parent / ".local" / "entity_aliases.json"

# Module-level cache, invalidated by the alias file's mtime — a long-running
# ingest process resolves many entities per run and shouldn't re-read/re-parse
# this file on every single call.
_cache: dict[str, str] | None = None
_cache_mtime: float | None = None


def load_aliases() -> dict[str, str]:
    """Keys are `resolver.normalize()`-d names, values are canonical node ids."""
    global _cache, _cache_mtime

    if not ALIAS_PATH.exists():
        _cache, _cache_mtime = {}, None
        return _cache

    mtime = os.path.getmtime(ALIAS_PATH)
    if _cache is not None and _cache_mtime == mtime:
        return _cache

    _cache = json.loads(ALIAS_PATH.read_text())
    _cache_mtime = mtime
    return _cache


def save_alias(normalized_name: str, canonical_id: str):
    """Read-modify-write — fine for this low-volume, single-user tool; not
    trying to handle concurrent writers."""
    global _cache, _cache_mtime

    ALIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    aliases = json.loads(ALIAS_PATH.read_text()) if ALIAS_PATH.exists() else {}
    aliases[normalized_name] = canonical_id
    ALIAS_PATH.write_text(json.dumps(aliases, indent=2, sort_keys=True) + "\n")

    # Invalidate the cache rather than updating it in place, so the next
    # load_aliases() call re-reads from disk and picks up the new mtime.
    _cache, _cache_mtime = None, None
