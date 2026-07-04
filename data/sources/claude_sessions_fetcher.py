import json
from datetime import datetime, timezone
from pathlib import Path

MIN_WORDS = 20

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SESSION_DIR_NAME = "".join(c if c.isalnum() else "-" for c in str(_PROJECT_ROOT))
_SESSION_DIR = Path.home() / ".claude" / "projects" / _SESSION_DIR_NAME
LAST_FETCHED_PATH = Path(__file__).parent.parent / "claude_sessions_last_fetched.json"


def _load_last_fetched() -> datetime | None:
    if LAST_FETCHED_PATH.exists():
        raw = json.loads(LAST_FETCHED_PATH.read_text()).get("last_fetched")
        return datetime.fromisoformat(raw) if raw else None
    return None


def _save_last_fetched(dt: datetime):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": dt.isoformat()}))


def _extract_turns(path: Path) -> tuple[list[str], str | None, str | None]:
    turns = []
    title = None
    last_timestamp = None

    for line in path.read_text(errors="ignore").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = entry.get("type")

        if etype == "ai-title":
            title = entry.get("aiTitle") or title
            continue

        if etype not in ("user", "assistant"):
            continue

        message = entry.get("message", {})
        content = message.get("content")

        if etype == "user" and isinstance(content, str) and content.strip():
            turns.append(f"Human: {content.strip()}")
        elif etype == "assistant" and isinstance(content, list):
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            if text.strip():
                turns.append(f"Assistant: {text.strip()}")

        if entry.get("timestamp"):
            last_timestamp = entry["timestamp"]

    return turns, title, last_timestamp


def fetch_claude_sessions(max_results: int = 100) -> list[dict]:
    fetch_started_at = datetime.now(timezone.utc)
    last_fetched = _load_last_fetched()

    if not _SESSION_DIR.exists():
        print(f"No Claude Code session directory found at {_SESSION_DIR}.")
        _save_last_fetched(fetch_started_at)
        return []

    if last_fetched:
        print(f"Fetching Claude Code sessions (modified since {last_fetched.isoformat()})...")
    else:
        print("Fetching Claude Code sessions (all available)...")

    results = []
    session_files = sorted(_SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)

    for path in session_files:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if last_fetched and mtime <= last_fetched:
            continue

        session_id = path.stem
        turns, title, last_timestamp = _extract_turns(path)
        if not turns:
            continue

        title = title or turns[0][:80]
        text = "\n\n".join([title, *turns])
        word_count = len(text.split())

        if word_count < MIN_WORDS:
            print(f"  Skipping session '{title}' ({word_count} words, below threshold)")
            continue

        session_time = last_timestamp or fetch_started_at.isoformat()
        print(f"  Fetched '{title}' ({word_count} words)")
        results.append({
            "session_id": session_id,
            "title": title,
            "plain_text_content": text,
            "session_time": session_time,
        })

        if len(results) >= max_results:
            break

    _save_last_fetched(fetch_started_at)
    print(f"Done. {len(results)} qualifying sessions fetched.")
    return results
