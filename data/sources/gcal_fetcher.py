import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

_DIR = Path(__file__).parent
CREDENTIALS_PATH = _DIR / "credentials.json"
TOKEN_PATH = _DIR / "token.json"
LAST_FETCHED_PATH = Path(__file__).parent.parent / "gcal_last_fetched.json"

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
    return build("calendar", "v3", credentials=creds)


def _load_last_fetched() -> datetime | None:
    if LAST_FETCHED_PATH.exists():
        raw = json.loads(LAST_FETCHED_PATH.read_text()).get("last_fetched")
        return datetime.fromisoformat(raw) if raw else None
    return None


def _save_last_fetched(dt: datetime):
    LAST_FETCHED_PATH.write_text(json.dumps({"last_fetched": dt.isoformat()}))


def _event_to_text(event: dict) -> str:
    parts = []
    summary = event.get("summary", "Untitled event")
    parts.append(summary)

    start = event.get("start", {})
    start_str = start.get("dateTime") or start.get("date", "")
    end = event.get("end", {})
    end_str = end.get("dateTime") or end.get("date", "")
    if start_str:
        parts.append(f"Time: {start_str} to {end_str}")

    organizer = event.get("organizer", {}).get("displayName") or event.get("organizer", {}).get("email", "")
    if organizer:
        parts.append(f"Organizer: {organizer}")

    attendees = event.get("attendees", [])
    if attendees:
        names = [a.get("displayName") or a.get("email", "") for a in attendees if not a.get("self")]
        if names:
            parts.append(f"Attendees: {', '.join(names)}")

    description = event.get("description", "").strip()
    if description:
        parts.append(description)

    location = event.get("location", "").strip()
    if location:
        parts.append(f"Location: {location}")

    return "\n".join(parts)


def fetch_gcal_events(days_back: int = 180) -> list[dict]:
    fetch_started_at = datetime.now(timezone.utc)
    last_fetched = _load_last_fetched()

    if last_fetched:
        time_min = last_fetched.isoformat()
        print(f"Fetching calendar events (since {time_min})...")
    else:
        time_min = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        print(f"Fetching calendar events (last {days_back} days)...")

    service = _get_service()
    results = []
    page_token = None

    while True:
        resp = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            maxResults=250,
            singleEvents=True,
            orderBy="updated",
            pageToken=page_token,
        ).execute()

        for event in resp.get("items", []):
            if event.get("status") == "cancelled":
                continue
            text = _event_to_text(event)
            word_count = len(text.split())
            title = event.get("summary", "Untitled event")
            if word_count < MIN_WORDS:
                print(f"  Skipping '{title}' ({word_count} words, below threshold)")
                continue
            start = event.get("start", {})
            event_time = start.get("dateTime") or start.get("date") or fetch_started_at.isoformat()
            print(f"  Fetched '{title}' ({word_count} words)")
            results.append({
                "event_id": event["id"],
                "title": event.get("summary", "Untitled event"),
                "plain_text_content": text,
                "event_time": event_time,
                "updated": event.get("updated", fetch_started_at.isoformat()),
                "url": event.get("htmlLink", ""),
                "organizer": event.get("organizer", {}).get("email", ""),
                "attendees": [
                    a.get("displayName") or a.get("email", "")
                    for a in event.get("attendees", [])
                ],
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    _save_last_fetched(fetch_started_at)
    print(f"Done. {len(results)} qualifying events fetched.")
    return results
