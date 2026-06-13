from pydantic import BaseModel
from typing import Optional, Literal

class QueryIntent(BaseModel):
    pattern: Literal["direct_lookup", "neighborhood", "path", "impact", "history"]
    anchor_entity: str
    relation: Optional[Literal["MEMBER_OF", "OWNS", "DEPENDS_ON", "USES", "REPORTED", "RESOLVED_BY", "MENTIONED_IN"]] = None
    direction: Optional[Literal["in", "out"]] = None
    target_entity: Optional[str] = None
    