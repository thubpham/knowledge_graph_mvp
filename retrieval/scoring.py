from datetime import datetime
from core.schema import Edge

def score_edge(edge: Edge, now: datetime):
    if edge.valid_until is not None and edge.valid_until < now: 
        return 0.0
    age_days = (now - edge.valid_from).days
    return 1 / (1 + age_days / 365)
