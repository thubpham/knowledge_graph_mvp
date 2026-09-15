"""Tests for enrichment/resolver.py's confirm_match index resolution.

Run with: python -m tests.test_resolver (from repo root)
"""
from core.schema import Node
from enrichment.extraction_schema import EntityMatchResult
from enrichment.resolver import confirm_match


class _FakeClient:
    """Stands in for LLMClient: returns a fixed EntityMatchResult JSON blob
    instead of calling out to a real model."""

    def __init__(self, result: EntityMatchResult):
        self._raw = result.model_dump_json()

    def generate_gemini(self, system, user, schema_type=None, kind=None):
        return self._raw


def _candidates():
    return [
        (Node(id="postgresql", type="tool", name="PostgreSQL"), 0.25),
        (Node(id="redis", type="tool", name="Redis"), 0.35),
    ]


def test_valid_index_returns_matching_node_id():
    client = _FakeClient(EntityMatchResult(match_index=1, reason="same tool"))
    assert confirm_match("Postgres", "tool", _candidates(), client) == "postgresql"


def test_none_index_returns_none():
    client = _FakeClient(EntityMatchResult(match_index=None, reason="not confident"))
    assert confirm_match("Something Else", "tool", _candidates(), client) is None


def test_out_of_range_index_returns_none():
    client = _FakeClient(EntityMatchResult(match_index=99, reason="hallucinated index"))
    assert confirm_match("Postgres", "tool", _candidates(), client) is None


if __name__ == "__main__":
    test_valid_index_returns_matching_node_id()
    test_none_index_returns_none()
    test_out_of_range_index_returns_none()
    print("All confirm_match tests passed.")
