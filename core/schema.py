from datetime import datetime

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