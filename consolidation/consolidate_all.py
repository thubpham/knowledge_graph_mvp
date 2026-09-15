import time
from datetime import datetime
from pathlib import Path

from core.graph import KnowledgeGarden
from llm_clients import LLMClient
from enrichment.resolver import resolve_entity
from .consolidate import consolidate
from trace import current_run
from core.failure_log import log_ingest_failure

_PENDING_EDGES_PATH = Path(__file__).parent.parent / ".local" / "pending_edges.txt"


def get_pending_nodes(kg: KnowledgeGarden, min_new_episodes: int = 1) -> list:
    """Two-stage filter, shared by consolidate_all() and
    scripts/run_consolidation.py's preview line (previously duplicated
    inline in both places, independently).

    Stage 1 is the original cheap check: a node with no episode more recent
    than its last consolidation provably has zero new episodes, so it's
    excluded without a query. Stage 2 only runs for nodes that pass stage 1,
    and only when min_new_episodes > 1 -- get_episodes_for_entity() is the
    exact same primitive consolidate() itself calls right before deciding
    whether there's anything to do, so this counts the same thing consolidate()
    would, just before spending an LLM call to find out.

    min_new_episodes=1 (the default) reproduces the pre-existing "any new
    episode" behavior exactly -- this function is a strict generalization,
    not a behavior change, until a caller passes something higher. Raising it
    means an entity that picked up e.g. one new mention this week waits until
    it has accumulated enough to be worth a consolidation pass, rather than
    being re-consolidated off a single episode every time this runs."""
    candidates = [
        n for n in kg.get_all_nodes()
        if n.last_episode_at and (n.last_consolidated_at is None or n.last_episode_at > n.last_consolidated_at)
    ]
    if min_new_episodes <= 1:
        return candidates
    return [
        n for n in candidates
        if len(kg.get_episodes_for_entity(n.id, since=n.last_consolidated_at)) >= min_new_episodes
    ]


def consolidate_all(kg: KnowledgeGarden, client: LLMClient, pending_log_path: Path = _PENDING_EDGES_PATH,
                     min_new_episodes: int = 1):
    all_unresolved = []
    consolidated_count = 0
    error_count = 0

    pending = get_pending_nodes(kg, min_new_episodes)
    total = len(pending)
    print(f"Consolidating {total} entities...")
    start = time.monotonic()
    consecutive_errors = 0
    run = current_run()
    for i, node in enumerate(pending, 1):
        elapsed = time.monotonic() - start
        avg = elapsed / (i - 1) if i > 1 else 0
        remaining = (total - i + 1) * avg
        print(f"[{i}/{total}] {node.name} ({node.id})  "
              f"[elapsed {elapsed/60:.1f}m, ETA {remaining/60:.1f}m]")
        if run:
            run.event("progress", current=i, total=total,
                       elapsed_seconds=elapsed, eta_seconds=remaining, entity=node.name)
        try:
            result = consolidate(node.id, kg, client)
        except Exception as e:
            consecutive_errors += 1
            error_count += 1
            print(f"  → error, skipped: {e}")
            log_ingest_failure("consolidation", node.id, node.name, e)
            if consecutive_errors >= 5:
                print(f"  ⚠ {consecutive_errors} consecutive errors — likely an API/billing issue, not bad data. Aborting early.")
                break
            continue
        consecutive_errors = 0
        if result is None:
            print(f"  → no episodes, skipped")
            continue
        consolidated_count += 1
        print(f"  → {result['edges_added']} edges added, {len(result['unresolved_edges'])} unresolved")
        all_unresolved.extend(result["unresolved_edges"])

    still_unresolved = []
    second_pass_count = 0

    for pending in all_unresolved:
        target_id = resolve_entity(pending["target"], None, kg, client)
        if target_id is None:
            still_unresolved.append(pending)
            continue
        try:
            kg.add_edge(pending["source_id"], target_id, pending["relation"], pending["fact"], datetime.now(),
                       source_type="consolidation")
            second_pass_count += 1
        except ValueError:
            pass

    if still_unresolved:
        with open(pending_log_path, "a") as f:
            for edge in still_unresolved:
                f.write(f"{edge['source_id']}\t{edge['relation']}\t{edge['target']}\t{edge['fact']}\n")

    return {
        "consolidated": consolidated_count,
        "errors": error_count,
        "edges_resolved_second_pass": second_pass_count,
        "still_unresolved": len(still_unresolved),
    }
