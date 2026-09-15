---
type: Concept
title: Architecture Overview
description: High-level architectural overview of the Personal Knowledge Graph.
---

# Architecture Overview

The PKG project is a local-first, single-user system designed to ingest, consolidate, and query personal digital memory. It is heavily inspired by Bitemporal Knowledge Graph architectures like [Zep/Graphiti](https://arxiv.org/abs/2501.13956).

## Pipeline Structure
The core processing logic follows two distinct phases:

### 1. Ingestion (`enrichment/`)
The ingestion phase transforms unstructured data into graph elements.
*   **Source Fetchers**: Modules in `data/sources/` pull raw data from Notion, Gmail, GCal, Google Docs, and local Claude Code sessions.
*   **Extraction**: `enrichment/extractor.py` processes raw content to identify entities and relationships.
*   **Resolution**: `enrichment/resolver.py` performs entity deduplication using a three-tier approach:
    1.  **Exact Match**: Fast-path for known entities.
    2.  **Embedding Similarity**: Using FalkorDB vector indices.
    3.  **LLM Confirmation**: Confirms ambiguity for entities with similar names.

### 2. Consolidation (`consolidation/`)
The consolidation phase promotes temporal facts into long-term semantic knowledge.
*   **Episodic folding**: Each node's episodic history is summarized into a durable state.
*   **Semantic promotion**: Recurring patterns in episodes are converted to permanent edges.

## Key Components
*   **Graph Backend**: [FalkorDB](https://falkordb.com) is used to persist nodes and edges.
*   **LLM Orchestration**: Managed by `llm_clients.py`, supporting multiple providers (Groq, Ollama, OpenAI).
*   **Querying**: The system supports bi-temporal retrieval, allowing queries like "What was my team structure in 2024?".
