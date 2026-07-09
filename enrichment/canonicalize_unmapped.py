import json
from datetime import datetime
from pathlib import Path

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from .resolver import resolve_entity, normalize

UNMAPPED_LOG_PATH = Path(__file__).parent.parent / ".local" / "unmapped_log.jsonl"

# Rule-based canonicalization only — no LLM generation calls. Each canonical
# value maps to a list of substrings matched (case-insensitive) against the
# LLM's free-text "attempted_type"/"attempted_relation" guess. Order matters:
# first match wins, so put more specific keywords before general ones.
TYPE_SYNONYMS = {
    "event": ["meeting", "event", "call", "appointment"],
    "document": ["document", "doc", "file", "spreadsheet", "slide"],
    "person": ["person", "user", "employee", "contact"],
    "team": ["team", "group", "department"],
    "service": ["service", "api", "system"],
    "tool": ["tool", "library", "software", "app"],
    "concept": ["concept", "topic", "idea", "project"],
}

RELATION_SYNONYMS = {
    "ATTENDED": ["attend", "joined the meeting", "was present"],
    "DISCUSSED": ["discuss", "talked about", "brought up", "raised"],
    "DECIDED": ["decide", "agreed", "concluded", "resolved to"],
    "SENT_TO": ["sent", "emailed", "messaged", "wrote to"],
    "SCHEDULED": ["schedule", "booked", "set up a meeting", "organized"],
    "AUTHORED": ["authored", "wrote", "created the doc", "drafted"],
    "REFERENCES": ["reference", "linked to", "mentions", "cites"],
}


def _match(value: str, synonyms: dict[str, list[str]]) -> str | None:
    normalized = value.lower()
    for canonical, keywords in synonyms.items():
        if any(keyword in normalized for keyword in keywords):
            return canonical
    return None


def canonicalize_unmapped_log(kg: KnowledgeGarden, client: LLMClient, log_path: Path = UNMAPPED_LOG_PATH) -> dict:
    """Rule-matches logged unmapped entities/relations against the canonical
    vocabulary and replays matches into the graph. No LLM generation calls —
    only the embedding calls resolve_entity already makes for entity
    resolution. Anything that doesn't match a rule stays logged as-is for a
    future pass."""
    if not log_path.exists():
        return {"entities_resolved": 0, "relations_resolved": 0, "left_unmapped": 0}

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]

    resolved_entities = 0
    resolved_relations = 0
    leftover = []

    for entry in entries:
        if entry["kind"] == "entity":
            canonical_type = _match(entry["attempted_type"], TYPE_SYNONYMS)
            if canonical_type is None:
                leftover.append(entry)
                continue
            existing_id = resolve_entity(entry["name"], canonical_type, kg, client)
            if existing_id is None:
                new_id = normalize(entry["name"]).replace(" ", "_")
                embedding = client.embed(entry["name"])
                try:
                    kg.add_node(new_id, canonical_type, entry["name"], embedding=embedding)
                except ValueError:
                    pass  # ID collision with a near-duplicate name — already covered
            resolved_entities += 1

        elif entry["kind"] == "relation":
            canonical_relation = _match(entry["attempted_relation"], RELATION_SYNONYMS)
            if canonical_relation is None:
                leftover.append(entry)
                continue
            source_id = resolve_entity(entry["source"], None, kg, client)
            target_id = resolve_entity(entry["target"], None, kg, client)
            if source_id is None or target_id is None:
                leftover.append(entry)
                continue
            reference_time = datetime.fromisoformat(entry["reference_time"])
            try:
                kg.add_edge(source_id, target_id, canonical_relation, entry["fact"], reference_time)
            except ValueError:
                pass
            resolved_relations += 1

    log_path.write_text("".join(json.dumps(e) + "\n" for e in leftover))

    return {
        "entities_resolved": resolved_entities,
        "relations_resolved": resolved_relations,
        "left_unmapped": len(leftover),
    }
