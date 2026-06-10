from datetime import datetime
from typing import Optional
import uuid

class Node: 
    def __init__(self, id: str, type: str, name: str):
        self.id = id
        self.type = type
        self.name = name
        self.summary = None
        self.embeddings = None
        self.created_at = datetime.now()
        self.consolidated = False 
        self.consolidation_run_id = None

class Edge: 
    def __init__(self, id: str, source: str, target: str, relation: str, fact: str, valid_from: datetime):
        self.id = id
        self.source = source
        self.target = target
        self.relation = relation
        self.fact = fact
        self.valid_from = valid_from
        self.valid_until = None
        self.confidence = 1.0
        self.source_type = None
        self.source_id = None
        self.ingested_at = datetime.now()
        self.extracted_by = None

class Episode: 
    def __init__(self, text: str, source_type: Optional[str] = None, source_id: Optional[str] = None, reference_time: Optional[datetime] = None):
        self.id = str(uuid.uuid4())
        self.text = text
        self.source_type = source_type
        self.source_id = source_id
        self.reference_time = reference_time or datetime.now()
        self.ingested_at = datetime.now()

class UnmappedEntity: 
    def __init__(self, name: str, attempted_type: str, fact: str, reason: str):
        self.name = name
        self.attempted_type = attempted_type
        self.fact = fact
        self.reason = reason

class UnmappedRelation: 
    def __init__(self, source: str, target: str, attempted_relation: str, fact: str, reason: str):
        self.source = source
        self.target = target
        self.attempted_relation = attempted_relation
        self.fact = fact
        self.reason = reason