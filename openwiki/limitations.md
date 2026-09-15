---
type: Concept
title: Known Limitations
description: Current limitations of the Personal Knowledge Graph MVP.
---

# Known Limitations

This project is a single-user MVP, not a production-grade service.

*   **Scalability**: While FalkorDB is efficient, the ingestion and consolidation pipelines are currently optimized for a single user's volume.
*   **Backfilling**: Older nodes created before embedding updates are not automatically backfilled with new, richer context.
*   **OAuth Management**: OAuth tokens for sources (Notion/Google) require manual setup as documented in the environment configuration samples.
*   **Ambiguity**: While the three-tier resolution pipeline is robust, it still relies on LLM confirmation for the most ambiguous cases, which carries a cost.
