---
type: Concept
title: Operational Scripts
description: Guide for running daily operations and maintaining the graph.
---

# Operational Scripts

The project relies on several key scripts located in `scripts/` to maintain the graph's state.

## Core Workflows
*   **Ingestion**: `scripts/run_ingest.py` triggers the data fetcher and entity extraction pipeline.
*   **Consolidation**: `scripts/run_consolidation.py` runs the periodic folding of entity episodes into semantic edges.
*   **Deduplication Review**: `scripts/run_dedup_review.py` allows manual verification of entity resolution decisions.
*   **Tracing**: `scripts/trace_dashboard.py` and `scripts/inspect_traces.py` provide visibility into the system's reasoning and LLM interactions.

## Maintenance
*   **Backfilling**: `scripts/backfill_embeddings.py` is used when updating the embedding strategy for existing entities.
*   **Progress Monitoring**: `scripts/check_progress.py` helps track the current state of ingestion and consolidation tasks.
