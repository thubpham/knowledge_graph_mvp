---
type: Concept
title: Quickstart
description: A quick start guide for the Personal Knowledge Graph (PKG) project.
---

# Quickstart

This repository implements a **Personal Knowledge Graph (PKG)** designed to capture, link, and reason over your personal digital data.

## What this is
Unlike document-store tools, this project extracts **facts** from your data to build a **bi-temporal graph**.

*   **Episodes**: Raw source data (emails, pages, events, session transcripts).
*   **Entities (Nodes)**: People, tools, projects, and concepts extracted across episodes.
*   **Edges**: Typed relationships (`MEMBER_OF`, `AUTHORED`, etc.) that are **bi-temporal** (they record `valid_from` and `valid_until` timestamps, preserving history).

## Pipeline Architecture
1.  **Ingestion**: Sources are fetched, entities/relations are extracted using an LLM, resolved against existing nodes (deduplication), and stored as episodic edges.
2.  **Consolidation**: Entity-specific episodes are periodically folded into durable summaries; recurring facts are promoted to semantic edges.

## Getting Started
See the [Architecture Overview](/openwiki/architecture/overview.md) to understand how the components interact.

## Backlog
*   [ ] Add detailed setup guide for OAuth credentials.
*   [ ] Document `scripts/` usage for manual runs.
*   [ ] Detail the FalkorDB schema.
