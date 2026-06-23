from pydantic import BaseModel
from typing import Literal, List

CONSOLIDATION_RELATION = Literal[
    "MEMBER_OF", "OWNS", "DEPENDS_ON", "USES", "REPORTED", "RESOLVED_BY"
]

class SemanticEdge(BaseModel):
    relation: CONSOLIDATION_RELATION
    target: str
    fact: str

class ConsolidationResult(BaseModel):
    summary: str
    semantic_edges: List[SemanticEdge]
    episodic_only: List[str]
