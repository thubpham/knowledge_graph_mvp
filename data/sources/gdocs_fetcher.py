import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_LOCAL_DIR = Path(__file__).parent.parent.parent / ".local"
CREDENTIALS_PATH = _LOCAL_DIR / "sources" / "credentials.json"
TOKEN_PATH = _LOCAL_DIR / "sources" / "gdocs_token.json"
LAST_FETCHED_PATH = _LOCAL_DIR / "gdocs_last_fetched.json"

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


def _save_last_fetched(dt: datetime):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": dt.isoformat()}))


def _doc_to_text(name: str, body: str, owner: str) -> str:
    parts = [name]
    if owner:
        parts.append(f"Owner: {owner}")
    if body.strip():
        parts.append(body.strip())
    return "\n".join(parts)


def fetch_gdocs_documents(days_back: int = 180, max_results: int = 100) -> list[dict]:
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
            file_id = file["id"]
            title = file.get("name", "Untitled document")

            try:
                exported = service.files().export(fileId=file_id, mimeType="text/plain").execute()
            except HttpError as e:
                print(f"  Skipping '{title}' (export failed: {e.reason})")
                continue
            body = exported.decode("utf-8", errors="ignore") if isinstance(exported, bytes) else exported

            owners = file.get("owners", [])
            owner = owners[0].get("displayName") or owners[0].get("emailAddress", "") if owners else ""

            text = _doc_to_text(title, body, owner)
            word_count = len(text.split())

            if word_count < MIN_WORDS:
                print(f"  Skipping '{title}' ({word_count} words, below threshold)")
                continue

            print(f"  Fetched '{title}' ({word_count} words)")
            results.append({
                "document_id": file_id,
                "title": title,
                "plain_text_content": text,
                "modified_time": file.get("modifiedTime", fetch_started_at.isoformat()),
                "url": file.get("webViewLink", ""),
                "owner": owner,
            })

        page_token = resp.get("nextPageToken")
        if not page_token or len(results) >= max_results:
            break

    _save_last_fetched(fetch_started_at)
    print(f"Done. {len(results)} qualifying documents fetched.")
    return results
