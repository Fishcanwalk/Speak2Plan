"""Google Tasks / Calendar access for Speak2Plan.

First run opens a browser for OAuth consent and saves token.json;
later runs reuse (and auto-refresh) that token.

Smoke test:
    python -m src.google_api
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

# Read + write scopes requested up front so add_task / add_event later
# don't force a second consent. Changing this list = delete token.json.
SCOPES = [
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/calendar.events",
]


def get_credentials() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"ไม่พบ {CREDENTIALS_FILE.name} — ดาวน์โหลด OAuth client (Desktop app) "
                "จาก Google Cloud Console แล้ววางไว้ที่รากโปรเจกต์"
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json())
    return creds


def tasks_service():
    return build("tasks", "v1", credentials=get_credentials())


def calendar_service():
    return build("calendar", "v3", credentials=get_credentials())


def list_tasks(max_results: int = 20, tasklist: str = "@default") -> list[dict]:
    """Return incomplete tasks from one task list: [{'id', 'title', 'due'}]."""
    resp = (
        tasks_service()
        .tasks()
        .list(tasklist=tasklist, maxResults=max_results, showCompleted=False)
        .execute()
    )
    return [
        {"id": t["id"], "title": t.get("title", ""), "due": t.get("due")}
        for t in resp.get("items", [])
    ]


def add_task(title: str, due: date | None = None, tasklist: str = "@default") -> dict:
    """Create a task. Tasks API keeps only the date part of `due` (time is ignored)."""
    body = {"title": title}
    if due:
        body["due"] = f"{due.isoformat()}T00:00:00.000Z"
    return tasks_service().tasks().insert(tasklist=tasklist, body=body).execute()


def complete_task(task_id: str, tasklist: str = "@default") -> dict:
    """Mark a task done (it stays in Google Tasks under 'Completed', not deleted)."""
    return (
        tasks_service()
        .tasks()
        .patch(tasklist=tasklist, task=task_id, body={"status": "completed"})
        .execute()
    )


def delete_task(task_id: str, tasklist: str = "@default") -> None:
    tasks_service().tasks().delete(tasklist=tasklist, task=task_id).execute()


def list_events(days: int = 7, max_results: int = 20, start: datetime | None = None) -> list[dict]:
    """Return primary-calendar events from `start` (default: now) to `days` later:
    [{'id', 'summary', 'start'}]. Pass a local-midnight `start` with days=1 for one whole day."""
    now = start.astimezone(timezone.utc) if start else datetime.now(timezone.utc)
    resp = (
        calendar_service()
        .events()
        .list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=(now + timedelta(days=days)).isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return [
        {
            "id": e["id"],
            "summary": e.get("summary", "(ไม่มีชื่อ)"),
            # all-day events have 'date', timed events have 'dateTime'
            "start": e["start"].get("dateTime", e["start"].get("date")),
        }
        for e in resp.get("items", [])
    ]


def add_event(summary: str, start: datetime | date, end: datetime | date | None = None) -> dict:
    """Create a primary-calendar event. A `date` makes an all-day event; a `datetime`
    (timezone-aware or local) a timed one, 1 hour long unless `end` is given."""
    if isinstance(start, datetime):
        start = start if start.tzinfo else start.astimezone()   # naive = this machine's local time
        end = end or start + timedelta(hours=1)
        body = {"start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}
    else:  # all-day: end date is exclusive
        body = {"start": {"date": start.isoformat()},
                "end": {"date": (end or start + timedelta(days=1)).isoformat()}}
    body["summary"] = summary
    return calendar_service().events().insert(calendarId="primary", body=body).execute()


if __name__ == "__main__":
    print("== Google Tasks ==")
    tasks = list_tasks()
    if not tasks:
        print("  (ไม่มี task ค้าง)")
    for t in tasks:
        print(f"  - {t['title']}" + (f"  (due {t['due'][:10]})" if t["due"] else ""))

    print("\n== Google Calendar (7 วันข้างหน้า) ==")
    events = list_events()
    if not events:
        print("  (ไม่มีนัด)")
    for e in events:
        print(f"  - {e['start']}  {e['summary']}")
