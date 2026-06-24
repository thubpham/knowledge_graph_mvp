from core.graph import KnowledgeGarden
import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def resolve_entity(node_name: str, kg: KnowledgeGarden):
    normalized_name = normalize(node_name)
    for node in kg.get_all_nodes():
        if normalize(node.name) == normalized_name:
            return node.id
    return None
