EXTRACTION_PROMPT = """
You are an expert Information Extraction system. Your task is to analyze the provided Natural Language Text and extract entities (nodes) and their relationships (edges) into a strict, valid JSON format based on the provided schema.

### Schema Definition
You must output a single JSON object containing exactly four keys: "nodes", "edges", "unmapped_entities", and "unmapped_relations".

1. "nodes": A list of objects, where each object represents an entity and MUST contain:
   - "name": The specific name or identifier of the entity as it appears or is inferred from the text.
   - "type": The category of the entity. MUST be one of: "person" | "service" | "team" | "tool" | "concept"

2. "edges": A list of objects, where each object represents a directed relationship and MUST contain:
   - "source": The "name" of the starting node. Must exactly match a "name" in the "nodes" list.
   - "target": The "name" of the ending node. Must exactly match a "name" in the "nodes" list.
   - "relation": MUST be one of: "MEMBER_OF" | "OWNS" | "DEPENDS_ON" | "USES" | "REPORTED" | "RESOLVED_BY" | "MENTIONED_IN"
   - "fact": A short, accurate snippet from the text that justifies this relationship.

3. "unmapped_entities": A list of objects for entities that cannot be classified into the allowed types. Each object MUST contain:
   - "name": The entity name as it appears in the text.
   - "attempted_type": The type you would assign if unconstrained.
   - "fact": The snippet from the text where this entity appears.
   - "reason": Why it cannot be mapped to the allowed types.

4. "unmapped_relations": A list of objects for relationships that cannot be mapped to the allowed relation types. Each object MUST contain:
   - "source": The source entity name.
   - "target": The target entity name.
   - "attempted_relation": The relation you would assign if unconstrained.
   - "fact": The snippet from the text that describes this relationship.
   - "reason": Why it cannot be mapped to the allowed relation types.

### Strict Extraction Rules
- **Type constraint:** Every node type MUST be one of the allowed types. If it cannot be mapped, do NOT include it in "nodes" — put it in "unmapped_entities" instead.
- **Relation constraint:** Every relation MUST be one of the allowed relation types. If it cannot be mapped, do NOT include it in "edges" — put it in "unmapped_relations" instead. The source and target nodes should still appear in "nodes" if they have other valid edges or can be typed.
- **Entity consistency:** The "source" and "target" strings in "edges" must exactly match "name" strings in "nodes" (case-sensitive).
- **No hallucinations:** Only extract nodes and edges explicitly stated or directly implied by the text. Do not invent facts.
- **Coreference resolution:** If the text uses pronouns (e.g., "she", "they") that clearly refer to a named entity, resolve them to the correct entity name.
- **Empty lists:** If there are no unmapped entities or relations, return empty lists for those keys. Never omit the keys.
- **Output format:** Return ONLY a valid JSON object. No conversational filler, no markdown code blocks, no explanations.

### Allowed Vocabularies
Node types: person | service | team | tool | concept
Relation types: MEMBER_OF | OWNS | DEPENDS_ON | USES | REPORTED | RESOLVED_BY | MENTIONED_IN

### Example
Input Text: "Alice joined the infra team last Monday. The infra team owns the auth service, which depends on Postgres. Alice is also the on-call engineer for auth service this week."

Output:
{
  "nodes": [
    {"name": "Alice", "type": "person"},
    {"name": "infra team", "type": "team"},
    {"name": "auth service", "type": "service"},
    {"name": "Postgres", "type": "tool"}
  ],
  "edges": [
    {"source": "Alice", "target": "infra team", "relation": "MEMBER_OF", "fact": "Alice joined the infra team last Monday"},
    {"source": "infra team", "target": "auth service", "relation": "OWNS", "fact": "The infra team owns the auth service"},
    {"source": "auth service", "target": "Postgres", "relation": "DEPENDS_ON", "fact": "which depends on Postgres"}
  ],
  "unmapped_entities": [],
  "unmapped_relations": [
    {
      "source": "Alice",
      "target": "auth service",
      "attempted_relation": "ON_CALL_FOR",
      "fact": "Alice is also the on-call engineer for auth service this week",
      "reason": "ON_CALL_FOR has no equivalent in the allowed relation vocabulary"
    }
  ]
}

### Current Task
Input Text: "{text}"
Output:
"""

QUERY_INTENT_PROMPT = """
You are a query router for a knowledge graph. Given a natural language question, classify it into exactly one traversal pattern and extract the parameters needed to execute it.

### Traversal Patterns
- "direct_lookup" — asking for a specific relationship of a named entity (e.g. "who owns X?", "what team is alice on?")
- "neighborhood" — asking what is broadly connected to or related to an entity (e.g. "what does the infra team work with?")
- "path" — asking how two specific entities are connected (e.g. "how is alice connected to postgres?")
- "impact" — asking what would be affected if an entity changed or failed (e.g. "what breaks if postgres goes down?")
- "history" — asking about past states, changes over time, or "has X ever been Y" (e.g. "what teams has alice been on?")

### Relation Vocabulary
MEMBER_OF | OWNS | DEPENDS_ON | USES | REPORTED | RESOLVED_BY | MENTIONED_IN

### Output Format
Return ONLY a valid JSON object with these keys:
- "pattern": one of the 5 pattern names above
- "anchor_entity": the primary entity the question is about, as it appears in the question (lowercase, normalized form expected to match a node name)
- "relation": the relation type from the vocabulary above that's relevant to this query, or null if not applicable
- "direction": "in" or "out" — for direct_lookup and history, indicates whether the anchor entity is the source or target of the relation. Use "out" if the anchor is doing the action (e.g. "alice MEMBER_OF X"), "in" if the anchor is receiving it (e.g. "X OWNS auth_service" — anchor is auth_service, direction is "in")
- "target_entity": for "path" queries only, the second entity. Otherwise null.

### Examples
Question: "Who owns the auth service?"
Output: {"pattern": "direct_lookup", "anchor_entity": "auth_service", "relation": "OWNS", "direction": "in", "target_entity": null}

Question: "What teams has alice been on?"
Output: {"pattern": "history", "anchor_entity": "alice", "relation": "MEMBER_OF", "direction": "out", "target_entity": null}

Question: "What breaks if postgres goes down?"
Output: {"pattern": "impact", "anchor_entity": "postgres", "relation": "DEPENDS_ON", "direction": "in", "target_entity": null}

Question: "How is alice connected to postgres?"
Output: {"pattern": "path", "anchor_entity": "alice", "relation": null, "direction": null, "target_entity": "postgres"}

Question: "What does the infra team work with?"
Output: {"pattern": "neighborhood", "anchor_entity": "infra_team", "relation": null, "direction": null, "target_entity": null}

### Current Task
Question: "{question}"
Output:
"""

CONSOLIDATION_PROMPT = """
You are consolidating episodic memory about an entity into semantic knowledge.

You will be given an entity and a chronological list of raw episodes that mention it. Your task is to synthesize across ALL episodes — not summarize each one individually — and determine what is durably, currently true about this entity.

### Entity
{entity_name}

### Raw Episodes (ordered oldest to newest)
{episodes}

### Instructions
1. Read all episodes in order. Pay attention to changes over time — if a later episode contradicts or supersedes an earlier one (e.g. a role change, a relationship ending), your summary must reflect the CURRENT state, not just concatenate everything as if it's all still true.
2. Identify facts that are persistent or recurring — these should be promoted to permanent semantic facts.
3. Identify facts that appear only once, are incidental, or are too specific/transient to be a standing fact about the entity — these stay episodic and should NOT be promoted.
4. Do not invent or infer facts beyond what the episodes state or directly imply.

### Output Format
Return ONLY a valid JSON object with these keys:

1. "summary": A 2-3 sentence semantic summary of what is persistently true about this entity, reflecting its current state.

2. "semantic_edges": A list of objects for facts that should become permanent graph edges. Each object MUST contain:
   - "relation": MUST be one of: "MEMBER_OF" | "OWNS" | "DEPENDS_ON" | "USES" | "REPORTED" | "RESOLVED_BY"
   - "target": the name of the other entity in this relationship, as it appears in the episodes
   - "fact": a short justification snippet, in your own words, for why this is a durable fact

3. "episodic_only": A list of short strings describing facts that were mentioned but should NOT be promoted to semantic edges (one-off details, transient context).

### Example

Entity: alice

Raw Episodes (ordered oldest to newest):
1. "alice joined the infra team last Monday."
2. "alice mentioned she's grabbing coffee with bob before the standup."
3. "alice left the infra team and joined the platform team this week. she's now leading the payments migration."

Output:
{
  "summary": "Alice is currently a member of the platform team, having previously been on the infra team. She is leading the payments migration effort.",
  "semantic_edges": [
    {"relation": "MEMBER_OF", "target": "platform_team", "fact": "Alice joined the platform team this week, having left the infra team"},
    {"relation": "OWNS", "target": "payments_migration", "fact": "Alice is leading the payments migration"}
  ],
  "episodic_only": [
    "Alice mentioned grabbing coffee with bob before a standup"
  ]
}

### Current Task
Entity: {entity_name}
Raw Episodes (ordered oldest to newest):
{episodes}

Output:
"""
