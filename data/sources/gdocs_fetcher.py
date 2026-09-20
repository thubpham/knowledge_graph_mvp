import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from core.graph_config import cursor_path

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_LOCAL_DIR = Path(__file__).parent.parent.parent / ".local"
CREDENTIALS_PATH = _LOCAL_DIR / "sources" / "credentials.json"
TOKEN_PATH = _LOCAL_DIR / "sources" / "gdocs_token.json"
LAST_FETCHED_PATH = cursor_path("gdocs")

MIN_WORDS = 20
DOC_MIME_TYPE = "application/vnd.google-apps.document"


def _get_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def _load_last_fetched() -> datetime | None:
    if LAST_FETCHED_PATH.exists():
        raw = json.loads(LAST_FETCHED_PATH.read_text()).get("last_fetched")
        return datetime.fromisoformat(raw) if raw else None
    return None


def save_last_fetched(dt: datetime):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": dt.isoformat()}))


def _doc_to_text(name: str, body: str, owner: str) -> str:
    parts = [name]
    if owner:
        parts.append(f"Owner: {owner}")
    if body.strip():
        parts.append(body.strip())
    return "\n".join(parts)


def _file_to_doc(service, file: dict, fallback_time: datetime, min_words: int = MIN_WORDS) -> dict | None:
    """Export one Drive file to the doc dict the ingester consumes, or None if
    it can't be exported or is below `min_words`. Shared by the bulk fetch and
    the single-document fetch so both produce identical shapes."""
    file_id = file["id"]
    title = file.get("name", "Untitled document")

    try:
        exported = service.files().export(fileId=file_id, mimeType="text/plain").execute()
    except HttpError as e:
        print(f"  Skipping '{title}' (export failed: {e.reason})")
        return None
    body = exported.decode("utf-8", errors="ignore") if isinstance(exported, bytes) else exported

    owners = file.get("owners", [])
    owner = owners[0].get("displayName") or owners[0].get("emailAddress", "") if owners else ""

    text = _doc_to_text(title, body, owner)
    word_count = len(text.split())

    if word_count < min_words:
        print(f"  Skipping '{title}' ({word_count} words, below threshold)")
        return None

    print(f"  Fetched '{title}' ({word_count} words)")
    return {
        "document_id": file_id,
        "title": title,
        "plain_text_content": text,
        "modified_time": file.get("modifiedTime", fallback_time.isoformat()),
        "url": file.get("webViewLink", ""),
        "owner": owner,
    }


_DOC_URL_RE = re.compile(r"/document/(?:u/\d+/)?d/([A-Za-z0-9_-]+)")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,}$")


def parse_document_id(url_or_id: str) -> str:
    """Accepts a docs.google.com/document/d/<ID>/... URL or a bare document ID."""
    value = url_or_id.strip()
    match = _DOC_URL_RE.search(value)
    if match:
        return match.group(1)
    if _BARE_ID_RE.match(value):
        return value
    raise ValueError(f"Not a Google Docs URL or document ID: {url_or_id!r}")


def fetch_gdocs_document(url_or_id: str) -> dict:
    """Fetch exactly one Google Doc by URL or ID. Does not touch the fetch
    cursor and has no minimum-length threshold (an explicit request beats the
    bulk noise filter). Raises ValueError if the file isn't a Google Doc or is
    trashed, so the caller sees why nothing was ingested."""
    document_id = parse_document_id(url_or_id)
    service = _get_service()
    file = service.files().get(
        fileId=document_id,
        fields="id, name, mimeType, trashed, modifiedTime, webViewLink, owners",
    ).execute()
    if file.get("mimeType") != DOC_MIME_TYPE:
        raise ValueError(f"{file.get('name')!r} is a {file.get('mimeType')}, not a Google Doc")
    if file.get("trashed"):
        raise ValueError(f"{file.get('name')!r} is in the trash")
    doc = _file_to_doc(service, file, datetime.now(timezone.utc), min_words=1)
    if doc is None:
        raise ValueError(f"Could not export {file.get('name')!r} as text")
    return doc


def fetch_gdocs_documents(days_back: int = 30, max_results: int = 100) -> tuple[list[dict], datetime]:
    fetch_started_at = datetime.now(timezone.utc)
    last_fetched = _load_last_fetched()

    if last_fetched:
        since = last_fetched
        print(f"Fetching Google Docs (modified since {since.isoformat()})...")
    else:
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        print(f"Fetching Google Docs (last {days_back} days)...")

    query = f"mimeType='{DOC_MIME_TYPE}' and modifiedTime > '{since.isoformat()}' and trashed = false"
    service = _get_service()
    results = []
    page_token = None

    while True:
        resp = service.files().list(
            q=query,
            pageSize=min(max_results, 100),
            orderBy="modifiedTime",
            fields="nextPageToken, files(id, name, modifiedTime, webViewLink, owners)",
            pageToken=page_token,
        ).execute()

        for file in resp.get("files", []):
            doc = _file_to_doc(service, file, fetch_started_at)
            if doc is not None:
                results.append(doc)

        page_token = resp.get("nextPageToken")
        if not page_token or len(results) >= max_results:
            break

    # Cursor is deliberately NOT saved here -- see the ingester, which saves it
    # only after every item has actually been ingested. Saving at fetch time
    # orphans everything fetched if the run dies mid-ingestion. Mirrors the
    # notion_fetcher/notion_ingester split.
    print(f"Done. {len(results)} qualifying documents fetched.")
    return results, fetch_started_at
