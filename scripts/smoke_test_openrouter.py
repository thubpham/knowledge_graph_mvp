import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_clients import LLMClient
from prompts import QUERY_INTENT_SYSTEM_PROMPT, QUERY_INTENT_USER_PROMPT
from retrieval.query_schema import QueryIntent
from trace import DB_PATH, Run

# A fixed question -> a byte-identical user prompt on every call, same as the
# (already byte-identical) system prompt. Real production traffic won't
# repeat a question verbatim, but this smoke test is checking the plumbing
# (sticky routing + cache token reporting), not extraction quality, so
# forcing a cache-friendly shape is the point.
N_CALLS = 10
QUESTION = "Who owns the auth service?"


def main():
    client = LLMClient(provider="openrouter")
    user_prompt = QUERY_INTENT_USER_PROMPT.replace("{question}", QUESTION)

    with Run(flow="smoke_test_openrouter") as run:
        run_id = run.run_id
        for i in range(N_CALLS):
            try:
                client.generate_gemini(
                    QUERY_INTENT_SYSTEM_PROMPT, user_prompt,
                    schema_type=QueryIntent, kind="smoke_test_openrouter",
                )
                print(f"[{i + 1}/{N_CALLS}] ok")
            except Exception as e:
                print(f"[{i + 1}/{N_CALLS}] FAILED: {e}")

    # generate_gemini() only returns the response text, not per-call
    # provider/usage -- that's logged to the trace db by _traced_call(), so
    # read it back from there rather than plumb a new return value through
    # every provider just for this script.
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT seq, provider, model, cached_tokens, cache_write_tokens, retries, status, error "
        "FROM llm_calls WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ).fetchall()
    conn.close()

    print()
    header = f"{'call':>4}  {'provider':<12} {'model':<30} {'cached':>8} {'cache_wr':>8} {'retries':>7}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        cached = r["cached_tokens"] if r["cached_tokens"] is not None else "-"
        cache_wr = r["cache_write_tokens"] if r["cache_write_tokens"] is not None else "-"
        retries = r["retries"] if r["retries"] is not None else "-"
        model = (r["model"] or "")[:30]
        status = r["status"] or ("error" if r["error"] else "?")
        print(f"{r['seq']:>4}  {r['provider'] or '':<12} {model:<30} {cached!s:>8} {cache_wr!s:>8} {retries!s:>7}  {status}")

    providers_seen = {r["provider"] for r in rows}
    retry_count = sum((r["retries"] or 0) for r in rows)
    print()
    if len(providers_seen) == 1:
        print(f"Sticky routing: OK — all {len(rows)} calls served by provider={providers_seen.pop()!r}")
    else:
        print(f"Sticky routing: WARNING — routed across {len(providers_seen)} providers: {providers_seen}")
    print(f"Total validation retries across {len(rows)} calls: {retry_count} "
          f"(high retries here means require_parameters isn't filtering out a "
          f"non-structured-output-capable upstream)")

    cached_after_first = [r["cached_tokens"] for r in rows[1:] if r["cached_tokens"]]
    if cached_after_first:
        print(f"Cache hits observed on calls 2+: {cached_after_first}")
    else:
        print("No cached_tokens > 0 observed on calls 2+ — either caching didn't "
              "land (check sticky routing above) or the upstream provider doesn't "
              "cache this model's prompts.")


if __name__ == "__main__":
    main()
