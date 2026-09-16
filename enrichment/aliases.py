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
    trying to handle concurrent writers.

    First-write-wins: both call sites (resolver.py's automatic llm_confirm
    tier and run_dedup_review.py's offline pass) are the same trust level --
    an unattended LLM confirmation, no human actually looks at either one --
    so there's no principled reason to let a later call silently clobber an
    earlier confirmed mapping. Previously this was an unconditional
    overwrite: a second, possibly-wrong confirmation for the same normalized
    name would silently replace the first with no record of the change. Now
    a conflicting write is refused and printed instead of applied -- see
    .local/IMPROVEMENTS.md's Entity Resolution/Dedup section for the
    additive (list-valued, re-disambiguate via confirm_match) alternative
    design, deferred until a real conflict is observed here."""
    global _cache, _cache_mtime

    ALIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    aliases = json.loads(ALIAS_PATH.read_text()) if ALIAS_PATH.exists() else {}

    existing = aliases.get(normalized_name)
    if existing is not None and existing != canonical_id:
        print(
            f"alias conflict: {normalized_name!r} already -> {existing}, "
            f"new confirmation -> {canonical_id} -- keeping {existing} "
            "(first-write-wins; see enrichment/aliases.py's save_alias docstring)"
        )
        return

    aliases[normalized_name] = canonical_id
    ALIAS_PATH.write_text(json.dumps(aliases, indent=2, sort_keys=True) + "\n")

    # Invalidate the cache rather than updating it in place, so the next
    # load_aliases() call re-reads from disk and picks up the new mtime.
    _cache, _cache_mtime = None, None
