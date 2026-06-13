
from .extraction_schema import *
from core.graph import KnowledgeGarden
import re

def normalize(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def resolve_entity(node_name: str, kg: KnowledgeGarden):
    normalized_name = normalize(node_name)
    for existing_node in kg.nodes.values():
        if normalize(existing_node.name) == normalized_name:
            return existing_node.id
    return None