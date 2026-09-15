---
type: Concept
title: API and LLM Integration
description: How the application interacts with LLM providers and its internal API.
---

# API and LLM Integration

## LLM Clients (`llm_clients.py`)
The project supports pluggable LLM providers. Routing is configured to handle:
*   **Provider Routing**: Calls can be routed to different providers (Groq, Ollama, OpenAI).
*   **Per-Task Routing**: The system supports routing specific types of tasks (like query-flow vs ingestion-flow) to specialized providers via environment configuration (e.g., `QUERY_LLM_PROVIDER`).

## Tracing (`trace.py`)
Observability is handled through end-to-end tracing of LLM calls, accessible via a live terminal dashboard (`scripts/trace_dashboard.py`).
