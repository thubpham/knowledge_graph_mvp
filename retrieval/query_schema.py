from pydantic import BaseModel
from typing import Optional, Literal

class QueryIntent(BaseModel):
    # "unsupported" is an explicit escape hatch, not a 6th traversal -- it
    # exists so the model can honestly decline a question none of the other
    # 5 patterns fit (aggregate/ranking questions like "who's connected to
    # the most people", or first-person questions the classifier can't
    # ground) instead of being forced to pick a pattern and invent an
    # anchor_entity to go with it. See prompts.py's QUERY_INTENT_SYSTEM_PROMPT
    # and .local/IMPROVEMENTS.md's Retrieval section for the failure this
    # was added to stop.
    pattern: Literal["direct_lookup", "neighborhood", "path", "impact", "history", "unsupported"]
    anchor_entity: Optional[str] = None
    relation: Optional[Literal["MEMBER_OF", "OWNS", "DEPENDS_ON", "USES", "REPORTED", "RESOLVED_BY", "MENTIONED_IN"]] = None
    direction: Optional[Literal["in", "out"]] = None
    target_entity: Optional[str] = None
    # Only meaningful when pattern == "unsupported": a short, free-text
    # explanation of why (e.g. "aggregate/ranking question, no such pattern"
    # or "first-person reference the classifier can't ground to a node").
    # Logged so real usage tells us what to build next, instead of guessing --
    # see scripts/trace_dashboard.py's unsupported-queries panel.
    reason: Optional[str] = None
    