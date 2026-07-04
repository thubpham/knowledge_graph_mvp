import base64
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_DIR = Path(__file__).parent
CREDENTIALS_PATH = _DIR / "credentials.json"
TOKEN_PATH = _DIR / "gmail_token.json"
LAST_FETCHED_PATH = Path(__file__).parent.parent / "gmail_last_fetched.json"

MIN_WORDS = 20


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
    return build("gmail", "v1", credentials=creds)


def _load_last_fetched() -> datetime | None:
    if LAST_FETCHED_PATH.exists():
        raw = json.loads(LAST_FETCHED_PATH.read_text()).get("last_fetched")
        return datetime.fromisoformat(raw) if raw else None
    return None


def _save_last_fetched(dt: datetime):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": dt.isoformat()}))


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_body(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text

    if payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    return ""


def _message_to_text(headers: list[dict], body: str) -> str:
    parts = []
    subject = _header(headers, "Subject") or "(no subject)"
    parts.append(subject)

    sender = _header(headers, "From")
    if sender:
        parts.append(f"From: {sender}")

    to = _header(headers, "To")
    if to:
        parts.append(f"To: {to}")

    if body.strip():
        parts.append(body.strip())

    return "\n".join(parts)


def fetch_gmail_messages(days_back: int = 30, max_results: int = 100) -> list[dict]:
    fetch_started_at = datetime.now(timezone.utc)
    last_fetched = _load_last_fetched()

    if last_fetched:
        after = last_fetched
        print(f"Fetching Gmail messages (since {after.isoformat()})...")
    else:
        after = datetime.now(timezone.utc) - timedelta(days=days_back)
        print(f"Fetching Gmail messages (last {days_back} days)...")

    query = f"after:{int(after.timestamp())}"
    service = _get_service()
    results = []
    page_token = None

    while True:
        resp = service.users().messages().list(
            userId="me",
            q=query,
            maxResults=min(max_results, 500),
            pageToken=page_token,
        ).execute()

        message_refs = resp.get("messages", [])
        for ref in message_refs:
            msg = service.users().messages().get(
                userId="me", id=ref["id"], format="full",
            ).execute()

            headers = msg.get("payload", {}).get("headers", [])
            body = _extract_body(msg.get("payload", {}))
            text = _message_to_text(headers, body)
            word_count = len(text.split())
            subject = _header(headers, "Subject") or "(no subject)"

            if word_count < MIN_WORDS:
                print(f"  Skipping '{subject}' ({word_count} words, below threshold)")
                continue

            internal_date_ms = int(msg.get("internalDate", 0))
            message_time = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc).isoformat()

            print(f"  Fetched '{subject}' ({word_count} words)")
            results.append({
                "message_id": msg["id"],
                "title": subject,
                "plain_text_content": text,
                "message_time": message_time,
                "url": f"https://mail.google.com/mail/u/0/#inbox/{msg['id']}",
                "sender": _header(headers, "From"),
            })

        page_token = resp.get("nextPageToken")
        if not page_token or len(results) >= max_results:
            break

    _save_last_fetched(fetch_started_at)
    print(f"Done. {len(results)} qualifying messages fetched.")
    return results
