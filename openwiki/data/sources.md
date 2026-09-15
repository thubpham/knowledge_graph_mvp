---
type: Concept
title: Ingestion Sources
description: Summary of supported data sources and their ingestion patterns.
---

# Ingestion Sources

The system supports multiple data sources, orchestrating them through a consistent ingestion interface.

## Supported Sources
*   **Notion**: Fetches pages using OAuth tokens.
*   **Google Calendar**: Fetches events (last 180 days).
*   **Gmail**: Fetches messages (last 30 days).
*   **Google Docs**: Uses the Drive API to extract content.
*   **Claude Code Sessions**: Reads local transcripts directly.

## Implementation Details
Each source is implemented as an "ingester" in `data/sources/`. These modules are responsible for fetching raw data, normalizing it, and passing it to the `enrichment/` pipeline.

Live ingestion status and metadata for these sources are logged locally (and git-ignored) in the `.local/` directory.
