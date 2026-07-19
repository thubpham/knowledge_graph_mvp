ENTITY_EXTRACTION_PROMPT = """
You are an expert Information Extraction system. Your task is to analyze the provided Natural Language Text and extract only the entities (nodes) mentioned in it, into a strict, valid JSON format based on the provided schema. Do not extract relationships — that is a separate task performed later.

### Schema Definition
You must output a single JSON object containing exactly two keys: "nodes" and "unmapped_entities".

1. "nodes": A list of objects, where each object represents an entity and MUST contain:
   - "name": The specific name or identifier of the entity as it appears or is inferred from the text.
   - "type": The category of the entity. MUST be one of: "person" | "service" | "team" | "tool" | "concept" | "event" | "document"
   - "context": A short snippet (ideally the single sentence) from the source text where this entity is mentioned or introduced. Capture surrounding descriptive detail (role, category, what it's part of) rather than just repeating the name — this is used later to disambiguate the entity from similarly-named-but-different entities elsewhere in the graph.

2. "unmapped_entities": A list of objects for entities that cannot be classified into the allowed types. Each object MUST contain:
   - "name": The entity name as it appears in the text.
   - "attempted_type": The type you would assign if unconstrained.
   - "fact": The snippet from the text where this entity appears.
   - "reason": Why it cannot be mapped to the allowed types.

### Strict Extraction Rules
- **Type constraint:** Every node type MUST be one of the allowed types. If it cannot be mapped, do NOT include it in "nodes" — put it in "unmapped_entities" instead.
- **No hallucinations:** Only extract entities explicitly stated or directly implied by the text. Do not invent entities.
- **Coreference resolution:** If the text uses pronouns (e.g., "she", "they") that clearly refer to a named entity introduced earlier in this same text, resolve them to that entity's name — do not create a separate node for the pronoun. If a pronoun or vague reference (e.g. "she", "the service", "that meeting") instead clearly and unambiguously refers to one of the Known Entities listed below, use that entity's exact listed name instead of inventing a new one. Only do this when you are confident — if it's ambiguous which entity (or whether any known entity) is meant, do not force a match; treat it as you normally would (extract with the name as it plainly appears in the text, or leave unmapped).
- **Empty lists:** If there are no unmapped entities, return an empty list for that key. Never omit the key.
- **Output format:** Return ONLY a valid JSON object. No conversational filler, no markdown code blocks, no explanations.

### Allowed Vocabulary
Node types: person | service | team | tool | concept | event | document

### Known Entities (optional coreference targets)
These entities already exist in the knowledge graph from recent prior text (other emails, docs, meetings, etc). They are provided ONLY to help you resolve pronouns/vague references that clearly point to one of them — do not treat their presence here as license to invent facts about them beyond what THIS text states, and do not force a match when unsure ("when in doubt, don't match" — a missed resolution is cheaper to fix than a wrong one):
{known_entities}

### Example
Input Text: "Alice joined the infra team last Monday. The infra team owns the auth service, which depends on Postgres. Alice is also the on-call engineer for auth service this week."

Output:
{
  "nodes": [
    {"name": "Alice", "type": "person", "context": "Alice joined the infra team last Monday."},
    {"name": "infra team", "type": "team", "context": "Alice joined the infra team last Monday."},
    {"name": "auth service", "type": "service", "context": "The infra team owns the auth service, which depends on Postgres."},
    {"name": "Postgres", "type": "tool", "context": "The auth service depends on Postgres."}
  ],
  "unmapped_entities": []
}

### Current Task
Input Text: "{text}"
Output:
"""

RELATION_EXTRACTION_PROMPT = """
You are an expert Information Extraction system. You have already been given the list of entities present in the provided Natural Language Text. Your task now is ONLY to extract the directed relationships (edges) between those entities, into a strict, valid JSON format based on the provided schema.

### Schema Definition
You must output a single JSON object containing exactly two keys: "edges" and "unmapped_relations".

1. "edges": A list of objects, where each object represents a directed relationship and MUST contain:
   - "source": The "name" of the starting entity. Must exactly match one of the "name" strings in the Entities list below.
   - "target": The "name" of the ending entity. Must exactly match one of the "name" strings in the Entities list below.
   - "relation": MUST be one of: "MEMBER_OF" | "OWNS" | "DEPENDS_ON" | "USES" | "REPORTED" | "RESOLVED_BY" | "MENTIONED_IN" | "ATTENDED" | "DISCUSSED" | "DECIDED" | "SENT_TO" | "SCHEDULED" | "AUTHORED" | "REFERENCES"
   - "fact": A short, accurate snippet from the text that justifies this relationship.

2. "unmapped_relations": A list of objects for relationships that cannot be mapped to the allowed relation types. Each object MUST contain:
   - "source": The source entity name.
   - "target": The target entity name.
   - "attempted_relation": The relation you would assign if unconstrained.
   - "fact": The snippet from the text that describes this relationship.
   - "reason": Why it cannot be mapped to the allowed relation types.

### Strict Extraction Rules
- **Entity constraint:** "source" and "target" MUST exactly match one of the "name" strings in the Entities list below (case-sensitive). Never introduce a source/target name that isn't in that list.
- **Relation constraint:** Every relation MUST be one of the allowed relation types. If it cannot be mapped, do NOT include it in "edges" — put it in "unmapped_relations" instead.
- **No hallucinations:** Only extract relationships explicitly stated or directly implied by the text. Do not invent facts.
- **Empty lists:** If there are no unmapped relations, return an empty list for that key. Never omit the key.
- **Output format:** Return ONLY a valid JSON object. No conversational filler, no markdown code blocks, no explanations.

### Allowed Vocabulary
Relation types: MEMBER_OF | OWNS | DEPENDS_ON | USES | REPORTED | RESOLVED_BY | MENTIONED_IN | ATTENDED | DISCUSSED | DECIDED | SENT_TO | SCHEDULED | AUTHORED | REFERENCES

### Relation Type Guide (for calendar/email/docs content)
- ATTENDED: a person attended a meeting/event (person -> event)
- DISCUSSED: a person or group talked about a topic/concept in a meeting, email, or doc (source -> concept)
- DECIDED: a decision was made about something (person/team -> concept, or event -> concept)
- SENT_TO: a message/email was sent from one person to another (person -> person)
- SCHEDULED: a person or team scheduled an event (person/team -> event)
- AUTHORED: a person wrote/created a document (person -> document)
- REFERENCES: a document or event references another document, tool, or concept (document/event -> concept/tool/document)

### Entities (the only valid source/target names)
{entities}

### Example
Input Text: "Alice joined the infra team last Monday. The infra team owns the auth service, which depends on Postgres. Alice is also the on-call engineer for auth service this week."
Entities: Alice (person), infra team (team), auth service (service), Postgres (tool)

Output:
{
  "edges": [
    {"source": "Alice", "target": "infra team", "relation": "MEMBER_OF", "fact": "Alice joined the infra team last Monday"},
    {"source": "infra team", "target": "auth service", "relation": "OWNS", "fact": "The infra team owns the auth service"},
    {"source": "auth service", "target": "Postgres", "relation": "DEPENDS_ON", "fact": "which depends on Postgres"}
  ],
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
Entities:
{entities}

Output:
"""

ENTITY_MATCH_PROMPT = """
You are resolving whether a newly-extracted entity refers to the same real-world entity as one of several existing candidates in a knowledge graph.

### New Entity
Name: {new_name}
Type: {new_type}

### Candidates
{candidates}

### Instructions
- Decide if the New Entity is the same real-world entity as exactly one of the candidates (e.g. an abbreviation, synonym, alternate phrasing, or naming variant of the same thing).
- Only match if you are confident they refer to the same thing. When in doubt, do not match — a missed merge is cheaper to fix than a wrongful one.
- Do not match candidates that are merely related or similar in topic but are distinct entities.

### Output Format
Return ONLY a valid JSON object with these keys:
- "match_name": the exact "name" string of the matching candidate, or null if none match.
- "reason": a short justification for your decision.

### Current Task
New Entity: {new_name} (type: {new_type})
Candidates:
{candidates}

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

SYNTHESIS_PROMPT = """
You are answering a question using only the facts retrieved from a personal knowledge graph.

Question: {question}

Retrieved facts:
{facts}

Instructions:
- Answer the question directly and concisely using only the facts above.
- Write in plain prose, 2-4 sentences max.
- If the facts don't fully answer the question, say what you do know and note the gap.
- Do not invent anything beyond the retrieved facts.
"""

CONSOLIDATION_PROMPT = """
You are consolidating episodic memory about an entity into semantic knowledge.

You will be given an entity, its existing semantic summary (from prior consolidation runs, if any), and a chronological list of NEW raw episodes that mention it since the last consolidation. Your task is to UPDATE the existing summary in light of the new episodes — not re-derive it from scratch — and determine what is durably, currently true about this entity.

### Entity
{entity_name}

### Existing Summary (from prior consolidation; may be "None yet" if this is the first run)
{existing_summary}

### New Raw Episodes Since Last Consolidation (ordered oldest to newest)
{episodes}

### Instructions
1. Treat the Existing Summary as already-established durable fact. Read the New Raw Episodes in order and merge them into it.
2. Pay attention to changes over time — if a new episode contradicts or supersedes the existing summary or an earlier new episode (e.g. a role change, a relationship ending), your updated summary must reflect the CURRENT state, not just concatenate everything as if it's all still true.
3. Identify facts from the NEW episodes that are persistent or recurring — these should be promoted to permanent semantic facts. Only emit semantic_edges for facts learned from the new episodes (facts already captured in the Existing Summary do not need a new edge unless they changed).
4. Identify facts from the NEW episodes that appear only once, are incidental, or are too specific/transient to be a standing fact — these stay episodic and should NOT be promoted.
5. Do not invent or infer facts beyond what the existing summary or new episodes state or directly imply.

### Output Format
Return ONLY a valid JSON object with these keys:

1. "summary": A 2-3 sentence semantic summary of what is persistently true about this entity, reflecting its current state after merging in the new episodes.

2. "semantic_edges": A list of objects for NEW or CHANGED facts (from the new episodes) that should become permanent graph edges. Each object MUST contain:
   - "relation": MUST be one of: "MEMBER_OF" | "OWNS" | "DEPENDS_ON" | "USES" | "REPORTED" | "RESOLVED_BY"
   - "target": the name of the other entity in this relationship, as it appears in the episodes
   - "fact": a short justification snippet, in your own words, for why this is a durable fact

3. "episodic_only": A list of short strings describing facts from the new episodes that were mentioned but should NOT be promoted to semantic edges (one-off details, transient context).

### Example

Entity: alice

Existing Summary (from prior consolidation; may be "None yet" if this is the first run):
Alice is a member of the infra team.

New Raw Episodes Since Last Consolidation (ordered oldest to newest):
1. "alice mentioned she's grabbing coffee with bob before the standup."
2. "alice left the infra team and joined the platform team this week. she's now leading the payments migration."

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
Existing Summary (from prior consolidation; may be "None yet" if this is the first run):
{existing_summary}
New Raw Episodes Since Last Consolidation (ordered oldest to newest):
{episodes}

Output:
"""
