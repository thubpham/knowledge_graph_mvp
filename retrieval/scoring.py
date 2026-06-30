from datetime import datetime, timezone
from core.schema import Edge


def _naive_utc(dt: datetime) -> datetime:
    """Strip timezone info, converting to UTC first if the datetime is aware."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def score_edge(edge: Edge, now: datetime):
    now = _naive_utc(now)
    if edge.valid_until is not None and _naive_utc(edge.valid_until) < now:
        return 0.0
    age_days = (now - _naive_utc(edge.valid_from)).days
    return 1 / (1 + age_days / 365)
