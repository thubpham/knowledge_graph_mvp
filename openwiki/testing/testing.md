---
type: Concept
title: Testing and Evals
description: Guidance for testing and benchmarking the PKG pipeline.
---

# Testing and Evals

## Smoke Tests
*   `test_consolidation.py` (or `scripts/smoke_test_consolidation.py`) verifies that the consolidation pipeline runs correctly without regressions.

## Benchmarks
*   `notebooks/06_gliner_benchmark.ipynb` includes benchmarking logic for entity extraction (using GLiNER).

## Best Practices
When changing entity extraction (`enrichment/extractor.py`) or resolution logic, always perform a dry run with `scripts/run_ingest.py` on a subset of data to monitor for unexpected graph growth or entity splits.
