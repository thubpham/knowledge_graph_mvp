import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
LAST_FETCHED_PATH = Path(__file__).parent.parent / "notion_last_fetched.json"

BLOCK_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3",
    "bulleted_list_item", "numbered_list_item", "to_do", "code", "quote",
}


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _request_with_retry(method: str, url: str, token: str, **kwargs) -> dict:
    delay = 1
    for attempt in range(6):
        response = httpx.request(method, url, headers=_headers(token), timeout=30, **kwargs)
        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", delay))
            print(f"Rate limited. Retrying in {wait}s...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Failed after retries: {url}")


def _load_last_fetched() -> str | None:
    if LAST_FETCHED_PATH.exists():
        data = json.loads(LAST_FETCHED_PATH.read_text())
        return data.get("last_fetched")
    return None


def _save_last_fetched(ts: str):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": ts}))


def _extract_plain_text(rich_text: list) -> str:
    return "".join(segment.get("plain_text", "") for segment in rich_text)


def _block_to_text(block: dict) -> str:
    block_type = block.get("type")
    if block_type not in BLOCK_TYPES:
        return ""
    content = block.get(block_type, {})
    rich_text = content.get("rich_text", [])
    return _extract_plain_text(rich_text)


def _fetch_block_children(page_id: str, token: str) -> str:
    lines = []
    url = f"{NOTION_API_BASE}/blocks/{page_id}/children"
    params = {"page_size": 100}

    while True:
        data = _request_with_retry("GET", url, token, params=params)
        for block in data.get("results", []):
            text = _block_to_text(block)
            if text:
                lines.append(text)
            # Recurse into nested blocks
            if block.get("has_children"):
                child_text = _fetch_block_children(block["id"], token)
                if child_text:
                    lines.append(child_text)
        if not data.get("has_more"):
            break
        params["start_cursor"] = data["next_cursor"]

    return "\n".join(lines)


def _get_page_title(page: dict) -> str:
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            rich_text = prop.get("title", [])
            return _extract_plain_text(rich_text)
    return ""


def _search_pages(token: str, last_fetched: str | None) -> list[dict]:
    url = f"{NOTION_API_BASE}/search"
    pages = []
    body = {
        "filter": {"value": "page", "property": "object"},
        "page_size": 100,
        "sort": {"direction": "descending", "timestamp": "last_edited_time"},
    }

    while True:
        data = _request_with_retry("POST", url, token, json=body)
        for result in data.get("results", []):
            if last_fetched and result.get("last_edited_time", "") <= last_fetched:
                return pages  # results are sorted desc; stop when we reach stale pages
            pages.append(result)
        if not data.get("has_more"):
            break
        body["start_cursor"] = data["next_cursor"]

    return pages


def fetch_notion_pages(token: str | None = None) -> list[dict]:
    token = token or os.getenv("NOTION_API_KEY")
    if not token:
        raise ValueError("NOTION_API_KEY not set in environment or .env")

    last_fetched = _load_last_fetched()
    fetch_started_at = datetime.now(timezone.utc).isoformat()

    print(f"Fetching Notion pages (since {last_fetched or 'beginning'})...")
    raw_pages = _search_pages(token, last_fetched)
    print(f"Found {len(raw_pages)} pages to process.")

    qualifying = []
    for page in raw_pages:
        page_id = page["id"]
        title = _get_page_title(page)
        plain_text = _fetch_block_children(page_id, token)
        word_count = len(plain_text.split())

        if word_count < 50:
            print(f"  Skipping '{title}' ({word_count} words, below threshold)")
            continue

        qualifying.append({
            "page_id": page_id,
            "page_title": title,
            "plain_text_content": plain_text,
            "last_edited_time": page.get("last_edited_time"),
            "created_time": page.get("created_time"),
            "url": page.get("url"),
        })
        print(f"  Fetched '{title}' ({word_count} words)")

    _save_last_fetched(fetch_started_at)
    print(f"Done. {len(qualifying)} qualifying pages fetched.")
    return qualifying
