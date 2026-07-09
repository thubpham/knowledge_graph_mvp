import re

# Rough chars-per-token estimate (no tokenizer dependency needed for a budget check).
_CHARS_PER_TOKEN = 4

# Below this, a single extraction call already sees the whole episode — skip chunking.
SKIP_THRESHOLD_TOKENS = 6_000

# Target size for each chunk once we do split.
TARGET_CHUNK_TOKENS = 4_000

# How many trailing sentences of chunk N are repeated as a prefix on chunk N+1,
# so an entity/relation whose sentence sits right on a boundary isn't lost.
OVERLAP_SENTENCES = 2

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s]


def _split_unit(text: str, separators: list[str]) -> list[str]:
    """Splits `text` on the first separator that actually breaks it into
    multiple non-trivial pieces, falling back to the next separator in the
    list, and finally to sentence/hard-wrap splitting."""
    if not separators:
        return _split_sentences(text) or _hard_wrap(text)

    sep, rest = separators[0], separators[1:]
    pieces = [p for p in text.split(sep) if p.strip()]
    if len(pieces) <= 1:
        return _split_unit(text, rest)
    return pieces


def _hard_wrap(text: str, max_chars: int = TARGET_CHUNK_TOKENS * _CHARS_PER_TOKEN) -> list[str]:
    words = text.split(" ")
    pieces, current = [], []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= max_chars:
            pieces.append(" ".join(current))
            current, current_len = [], 0
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str) -> list[str]:
    """Splits `text` into token-budgeted chunks with sentence-level overlap.

    Recursively prefers larger structural boundaries (paragraphs, then lines,
    then sentences) over an arbitrary character cut, so a chunk boundary never
    lands mid-sentence. Returns [text] unchanged when it already fits in one
    extraction call.
    """
    if _estimate_tokens(text) <= SKIP_THRESHOLD_TOKENS:
        return [text]

    units = _split_unit(text, ["\n\n", "\n"])

    # Any unit still too large on its own (e.g. one giant paragraph/code block)
    # gets recursively broken down to sentence/hard-wrap level.
    flattened = []
    for unit in units:
        if _estimate_tokens(unit) > TARGET_CHUNK_TOKENS:
            flattened.extend(_split_unit(unit, []))
        else:
            flattened.append(unit)

    # Greedily pack units into chunks up to the target budget.
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    for unit in flattened:
        unit_tokens = _estimate_tokens(unit)
        if current_parts and current_tokens + unit_tokens > TARGET_CHUNK_TOKENS:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_tokens = [], 0
        current_parts.append(unit)
        current_tokens += unit_tokens
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Prefix each chunk (after the first) with the trailing sentences of the
    # previous chunk, so extraction sees a bit of leading context.
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sentences = _split_sentences(chunks[i - 1])
        overlap = " ".join(prev_sentences[-OVERLAP_SENTENCES:])
        overlapped.append(f"{overlap}\n\n{chunks[i]}" if overlap else chunks[i])

    return overlapped
