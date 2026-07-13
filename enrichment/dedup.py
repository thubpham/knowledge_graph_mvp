"""Repeatable, offline entity-dedup pipeline.

Unlike `resolver.py`'s online path (which only ever compares a *new* entity
against a handful of ANN-nearest candidates at ingest time), this module does
an exhaustive pass over an entire node type after the fact: embed every live
node of that type, cluster by cosine distance, and route each candidate pair
through the same auto-merge / LLM-confirm tiers `resolve_entity` uses. Meant
to be re-run any time drift has accumulated, not a one-off fix — see
`scripts/run_dedup_review.py` for the CLI wrapper.

Confirmed merges are recorded in the alias table (`enrichment/aliases.py`) so
the next ingest run's `resolve_entity` alias tier (step 0) catches the same
identity immediately, without ever re-walking the embedding/LLM tiers.
"""

import json
from datetime import datetime
from pathlib import Path

from core.graph import KnowledgeGarden
from core.schema import Node
from llm_clients import LLMClient
from .resolver import AUTO_MATCH_DISTANCE, NO_MATCH_DISTANCE, confirm_match, normalize
from .aliases import ALIAS_PATH, load_aliases, save_alias  # re-exported for callers of this module

REVIEW_LOG_PATH = Path(__file__).parent.parent / ".local" / "dedup_review_log.jsonl"


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


class _UnionFind:
    """Groups duplicate chains (A~B~C) into one cluster instead of leaving
    them as separate overlapping pairs."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_merge_candidates(kg: KnowledgeGarden, node_type: str) -> list[dict]:
    """Returns a list of cluster dicts: {"members": [(id, name), ...],
    "pairs": [(id_a, id_b, distance), ...]}. Pairwise cosine distance over
    all nodes of `node_type` — O(n^2), fine since n is expected to be a few
    hundred at most per type. No numpy: keeping the dependency footprint
    identical to the rest of the repo (see requirements.txt)."""
    embeddings = [(i, n, e) for (i, n, e) in kg.get_node_embeddings_by_type(node_type) if e]

    pairs = []
    for i in range(len(embeddings)):
        id_a, _, emb_a = embeddings[i]
        for j in range(i + 1, len(embeddings)):
            id_b, _, emb_b = embeddings[j]
            distance = _cosine_distance(emb_a, emb_b)
            if distance <= NO_MATCH_DISTANCE:
                pairs.append((id_a, id_b, distance))

    if not pairs:
        return []

    uf = _UnionFind(i for i, _, _ in embeddings)
    for id_a, id_b, _ in pairs:
        uf.union(id_a, id_b)

    names_by_id = {i: n for i, n, _ in embeddings}
    clusters_by_root: dict[str, dict] = {}
    for id_a, id_b, distance in pairs:
        root = uf.find(id_a)
        cluster = clusters_by_root.setdefault(root, {"members": set(), "pairs": []})
        cluster["members"].add(id_a)
        cluster["members"].add(id_b)
        cluster["pairs"].append((id_a, id_b, distance))

    return [
        {
            "members": [(mid, names_by_id[mid]) for mid in cluster["members"]],
            "pairs": cluster["pairs"],
        }
        for cluster in clusters_by_root.values()
    ]


def review_cluster(kg: KnowledgeGarden, client: LLMClient, cluster: dict) -> list[dict]:
    """Picks the cluster's earliest-created_at member as the canonical
    ("winner") candidate and compares every other member against it:
    - distance <= AUTO_MATCH_DISTANCE -> auto-confirm, no LLM call
    - otherwise -> ask the LLM via the shared confirm_match() helper
    Returns proposal dicts; never writes to the graph or the alias table —
    that's the caller's job (scripts/run_dedup_review.py --apply)."""
    member_ids = [mid for mid, _ in cluster["members"]]
    nodes_by_id = {mid: node for mid in member_ids if (node := kg.get_node(mid)) is not None}
    if len(nodes_by_id) < 2:
        return []

    canonical = min(nodes_by_id.values(), key=lambda n: n.created_at or datetime.max)

    distance_by_pair = {
        frozenset((id_a, id_b)): distance for id_a, id_b, distance in cluster["pairs"]
    }

    proposals = []
    for mid, node in nodes_by_id.items():
        if mid == canonical.id:
            continue

        distance = distance_by_pair.get(frozenset((mid, canonical.id)))
        if distance is None:
            # Only connected to the canonical member transitively (through a
            # third cluster member) — no direct pairwise distance was
            # computed. Treat conservatively: route to the LLM rather than
            # guessing a distance.
            distance = NO_MATCH_DISTANCE

        if distance <= AUTO_MATCH_DISTANCE:
            proposals.append({
                "loser_id": mid,
                "loser_name": node.name,
                "winner_id": canonical.id,
                "winner_name": canonical.name,
                "distance": distance,
                "method": "auto",
            })
            continue

        match_id = confirm_match(node.name, node.type, [(canonical, distance)], client)
        proposals.append({
            "loser_id": mid,
            "loser_name": node.name,
            "winner_id": canonical.id,
            "winner_name": canonical.name,
            "distance": distance,
            "method": "llm" if match_id == canonical.id else "rejected",
        })

    return proposals


def log_rejected(proposal: dict, log_path: Path = REVIEW_LOG_PATH):
    """Append-only JSONL log of hard cases the pipeline declined to merge, so
    a human can inspect them later and, if warranted, add an alias entry by
    hand — mirrors the `unmapped_log.jsonl` pattern in
    `enrichment/canonicalize_unmapped.py`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({**proposal, "logged_at": datetime.now().isoformat()}) + "\n")
