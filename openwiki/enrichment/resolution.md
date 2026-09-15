---
type: Concept
title: Entity Resolution
description: Details on the three-tier entity resolution and deduplication pipeline.
---

# Entity Resolution

Entity resolution (deduplication) is critical for maintaining a clean graph. The system uses a three-tier pipeline defined in `enrichment/resolver.py`:

## Tier 1: Exact Match (Fast Path)
This is the fastest lookup tier. If an entity with the exact name is already known, it is reused.

## Tier 2: Embedding Similarity
If no exact match exists, the system compares the new entity's embedding with existing nodes in the graph's vector index (FalkorDB).
*   **Context-rich embeddings**: To improve accuracy, the system embeds `"{name}: {context}"`, where `{context}` is a snippet from the source sentence that justified the entity.

## Tier 3: LLM Confirmation
If the embedding similarity is ambiguous, the system invokes an LLM to confirm whether the entities refer to the same concept. This tier is only invoked when necessary to minimize LLM costs and latency.
