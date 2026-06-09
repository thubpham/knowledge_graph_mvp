from pydantic import BaseModel
from typing import List

class ExtractedNode(BaseModel):
    name: str
    type: str

class ExtractedEdge(BaseModel):
    source: str
    target: str
    relation: str
    fact: str

class UnmappedEntity(BaseModel):
    name: str
    attempted_type: str
    fact: str
    reason: str

class UnmappedRelation(BaseModel):
    source: str
    target: str
    attempted_relation: str
    fact: str
    reason: str

class ExtractionResult(BaseModel):
    nodes: List[ExtractedNode]
    edges: List[ExtractedEdge]
    unmapped_entities: List[UnmappedEntity]
    unmapped_relations: List[UnmappedRelation]