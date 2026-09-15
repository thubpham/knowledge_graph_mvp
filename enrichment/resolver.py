from core.graph import KnowledgeGarden
from core.schema import Node
from llm_clients import LLMClient
from prompts import ENTITY_MATCH_SYSTEM_PROMPT, ENTITY_MATCH_USER_PROMPT
from .extraction_schema import EntityMatchResult
from .aliases import load_aliases, save_alias
from trace import current_run
import re
import unicodedata

# FalkorDB's vector index 'score' for similarity_function='cosine' is a
# distance (lower = more similar), not a similarity — confirmed empirically
# against a live instance.
#
# Embeddings switched from gemini-embedding-001 to a local nomic-embed-text
# (see llm_clients._OllamaEmbedder) -- Gemini's free-tier embed quota
# (1000 requests/day) made a full-corpus ingest take days. This changed the
# distance distribution enough that the old auto-merge tier is no longer
# trustworthy: a 7-pair smoke test showed nomic separating synonyms from
# distinct entities more cleanly than Gemini did (no more Alice/Bob sitting
# closer than a true synonym pair, as Gemini did), but n=7 was never a real
# calibration, and a full sweep of the (pre-wipe) 239-node graph found
# same-type auto-merge candidates that were flat wrong, e.g. "workspace" ~
# "WorkspaceTrrDO" at 0.104 -- well inside the old 0.20 auto-merge band.
#
# So there is no more auto-merge tier: anything at or below NO_MATCH_DISTANCE
# goes to the LLM, full stop. The cost is real (most resolutions now cost a
# ~1.8s Haiku call instead of being free), but a wrong blind merge is much
# harder to undo than an extra LLM call, and Haiku's cheap enough that this
# is the right trade. NO_MATCH_DISTANCE=0.42 is carried over from the
# Gemini-era calibration and has NOT yet been re-fit to nomic on real
# resolve_confirm outcomes from a full ingest -- revisit once
# scripts/run_dedup_review.py has produced enough confirmed/rejected pairs
# under nomic to calibrate against real ground truth instead of a guess.
NO_MATCH_DISTANCE = 0.42
CANDIDATE_K = 5


def _singularize(word: str) -> str:
    """Conservative last-word singularization so plural/singular mentions of
    the same entity ("Cloudflare Worker" / "Cloudflare Workers") collide at
    the free exact-match tier instead of paying an embed + LLM confirm every
    time. Deliberately narrow: skips short words (avoids "gas"/"bus"-style
    3-4 letter words), words ending in "us"/"is"/"ss" (status, virus, basis,
    crisis, glass -- Latin/Greek singulars and doubled-s words a naive strip
    would mangle), and only touches the final word, since plurality in a
    multi-word entity name is carried by the last word. Verified against the
    live graph's 239 node names before shipping: introduces zero false
    same-type collisions, collapses one genuine duplicate (Durable
    Object/Objects). Known, accepted false-positive risk: a proper noun that
    is already singular but ends in a plain "s" (e.g. "Windows" the OS) gets
    stripped to "Window" and would collide with a real "Window" entity if one
    existed -- not worth guarding against given how rare that collision is
    versus how common the Thing/Things pattern is in extracted entity names."""
    if len(word) <= 4:
        return word
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
        return word[:-2]
    if word.endswith(('us', 'is', 'ss')):
        return word
    if word.endswith('s'):
        return word[:-1]
    return word


# Matched against a trailing/leading word only -- see normalize()'s steps 6-7
# below. Kept as module-level sets/regex rather than inline literals so the
# rule is visible without reading into the function body.
_DOC_EXT_RE = re.compile(
    r'\.(md|py|js|ts|txt|json|yaml|yml|csv|pdf|docx?|xlsx?|pptx?)$', re.IGNORECASE
)
_CORP_SUFFIXES = ('inc', 'llc', 'ltd', 'corp', 'co')


def normalize(text: str) -> str:
    """The matching key used by every resolver tier and by
    resolve_all_matching(). This is deliberately the AGGRESSIVE half of the
    normalize()/slugify() split (see slugify()'s docstring) -- it exists
    purely to collapse variant spellings onto the same key, so getting more
    aggressive over time is a pure win here: every rule below turns what used
    to be an embed + LLM-confirm round trip (~29s on the local model) into a
    free dict lookup. slugify() is NOT built from this function and does not
    change when this does, so none of these rules touch existing node ids.

    Each rule was measured individually against all 2167 live node names
    before shipping (2026-07-26): zero introduced new same-type collisions,
    alone or combined, and each one demonstrably fires on real extracted
    names (not dead code) -- see .local/IN_FLIGHT.md's Fix 4 section for the
    counts. "Same-type collision" is the risk that matters: two DIFFERENT
    real entities of the same type suddenly normalizing to the same string,
    which would make resolve_entity treat them as one. Re-measure the same
    way before adding another rule.
    """
    # 1. Diacritics: NFKD-decompose then drop the non-ASCII combining marks,
    #    so an accented spelling matches its unaccented one ("Café" == "Cafe").
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # 2. Possessive: "Jensen's inequality" == "Jensen inequality".
    text = re.sub(r"'s\b", '', text)
    # 3. Ampersand: "R&D" == "R and D" (word-boundary-safe since it just
    #    inserts spaces around "and", collapsed with everything else below).
    text = re.sub(r'&', ' and ', text)
    # 4. Trailing document extension: a concept and the literal filename it
    #    came from collapse ("instruction.md" == "instruction"). Only
    #    trailing, so a name that merely mentions ".md" mid-sentence is
    #    untouched.
    text = _DOC_EXT_RE.sub('', text.strip())

    text = text.lower()
    # 5. Underscores and hyphens become spaces, so a snake_case or kebab-case
    # spelling normalizes to the same string as the spaced display name.
    # Previously these were handled inconsistently and both wrongly: `\w`
    # *includes* underscore, so [^\w\s] never stripped it ("true_ventures" ->
    # "true_venture"), while hyphens were deleted outright ("zephyr-cloud-io"
    # -> "zephyrcloudio"). Neither could ever exact-match "True Ventures" /
    # "Zephyr Cloud IO". That mattered because 1148 of 2167 live nodes have an
    # underscore id + spaced name, and the query path's anchor comes from an
    # LLM guess that tends to be id-shaped -- so those all fell through to the
    # embed + LLM-confirm tiers instead of hitting the free exact-match tier.
    text = re.sub(r'[_\-]+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse runs of whitespace, so "a  b" and "a b" agree -- the substitution
    # above can introduce doubles ("a - b").
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split(' ') if text else []
    if words:
        words[-1] = _singularize(words[-1])

    # 6. Leading article: "The AI Platform" == "AI Platform". `len(words) > 1`
    #    guard so a name that's just "The" isn't stripped to nothing.
    if len(words) > 1 and words[0] == 'the':
        words = words[1:]
    # 7. Corporate suffix: "True Ventures Inc" == "True Ventures". Same
    #    non-empty guard -- a name that's literally "Inc" stays as-is.
    if len(words) > 1 and words[-1] in _CORP_SUFFIXES:
        words = words[:-1]
    # 8. Singularize every remaining word, not just the last -- rules 6/7 can
    #    change which word IS last (e.g. dropping a trailing "Inc" exposes a
    #    plural word that rule 5 never got to singularize). Idempotent on a
    #    word rule 5 already singularized.
    words = [_singularize(w) for w in words]
    return ' '.join(words)


def slugify(name: str) -> str:
    """Turns an entity name into a node id. Deliberately a SEPARATE, more
    conservative function from normalize() above, not normalize(name).replace(
    " ", "_") -- the two have opposite pressures. normalize() is a matching
    key: the more names it can collapse together, the more free tier-1 hits
    instead of a 29s embed+LLM-confirm round trip, so it should keep getting
    more aggressive over time (see Fix 4 in .local/IN_FLIGHT.md). An id is a
    stored primary key: every existing edge references one by its exact
    string, so churning the minting rule churns ids for every NEW node created
    from that point on, and (rarely, see ingester.py's collision handling) can
    change which node a name-collision lands on. Ids should change only
    deliberately, not as a side effect of tuning the matching key.

    Frozen at the normalize() logic as it existed before the "_/- -> space"
    fix (see that fix's comment above): hyphens are deleted outright
    ("zephyr-cloud-io" -> "zephyraiinternal"-style squashing), underscores are
    kept (`\\w` includes `_`, so [^\\w\\s] never strips it). This looks
    inconsistent in isolation, but it's the exact rule every id already on
    this graph was minted with -- matching it here means slugify() keeps
    producing ids in the same style newly-created nodes have always had,
    independent of how aggressive normalize() becomes for matching.
    Singularization is shared with normalize() (via _singularize) since it
    was already baked into every existing id before this split happened."""
    text = name.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    words = text.split(' ')
    if words:
        words[-1] = _singularize(words[-1])
    return '_'.join(words)


def embedding_text(name: str, context: str | None) -> str:
    """The string actually embedded for a node — bare name alone carries too
    little semantic signal to disambiguate short, similarly-named-but-different
    entities (see IMPROVEMENTS.md's "lack of context" finding: "Bob"/"vern",
    "PostgreSQL"/"crm" landing suspiciously close in embedding space). Callers
    with a source-text snippet available (entity extraction) should pass it as
    `context`; callers without one (consolidation targets, unmapped-log
    replay) pass `None` and fall back to name-only, matching legacy behavior."""
    return f"{name}: {context}" if context else name


def confirm_match(
    new_name: str, new_type: str | None, candidates: list[tuple[Node, float]], client: LLMClient
) -> str | None:
    """Ask the LLM to confirm whether `new_name` is the same real-world
    entity as exactly one of `candidates`. Shared by `resolve_entity`'s
    ambiguous-band tier and the offline dedup pass in `enrichment/dedup.py`
    so there's exactly one implementation of this prompt-filling logic.
    Returns the matching candidate's node id, or None if no confident match."""
    candidates_text = "\n".join(
        f'{i+1}. name="{c.name}", type={c.type}'
        for i, (c, _dist) in enumerate(candidates)
    )
    user_prompt = (
        ENTITY_MATCH_USER_PROMPT
        .replace("{new_name}", new_name)
        .replace("{new_type}", new_type or "unknown")
        .replace("{candidates}", candidates_text)
    )
    raw = client.generate_gemini(
        ENTITY_MATCH_SYSTEM_PROMPT, user_prompt,
        schema_type=EntityMatchResult, kind="resolve_confirm",
    )
    result = EntityMatchResult.model_validate_json(raw)

    if result.match_index is None:
        return None
    if not 1 <= result.match_index <= len(candidates):
        print(
            f"confirm_match: model returned out-of-range match_index="
            f"{result.match_index} for {len(candidates)} candidate(s) "
            f"(new_name={new_name!r}). Treating as no match."
        )
        return None
    return candidates[result.match_index - 1][0].id


def resolve_all_matching(node_name: str, kg: KnowledgeGarden) -> list[str]:
    """Every live node whose normalized name equals this one, across ALL
    types -- the read-side counterpart to resolve_entity's tier 1.

    Query-only. Every write path (ingest, consolidation, canonicalize_unmapped)
    must keep using resolve_entity, which returns a single, type-scoped id --
    a write has to pick exactly one node to attach an edge to, and type
    scoping is what prevents a "person" and a "tool" that happen to share a
    normalized name from being treated as the same entity.

    Query has the opposite problem: resolve_entity returning a single id means
    whichever same-named node the graph scan happens to hit first "wins," and
    every other node with that name is silently unreachable -- proven live on
    this graph: "chat-api" (service, 16 edges) and "chat_api" (concept, 2
    edges) both normalize to "chat api", and querying only ever surfaced the
    service. There's no wrong-merge risk here since nothing is written --
    returning every match and letting the caller union their edges only ever
    adds facts a single-anchor lookup would have missed.

    Does not fall through to embedding/LLM tiers: if a user's question
    resolves to zero exact matches, the existing embed+confirm path in
    resolve_entity (called separately by the query flow) is what handles
    fuzzy matching for a single best guess. This function's job is narrower --
    catch every node an exact match already identifies, not guess further."""
    normalized_name = normalize(node_name)
    return [
        candidate_id
        for candidate_id, candidate_name in kg.get_node_identities(None)
        if normalize(candidate_name) == normalized_name
    ]


def resolve_entity(
    node_name: str,
    node_type: str | None,
    kg: KnowledgeGarden,
    client: LLMClient,
    context: str | None = None,
):
    # 0. Alias fast path — cheap dict lookup, no embedding/LLM cost. Catches
    # structural identity aliasing (a name vs. a full name vs. an email
    # address) that the embedding-based tiers below cannot: those strings
    # just don't sit close together in embedding space, so the pair never
    # even reaches the LLM confirmation tier. Aliases are seeded by hand or
    # confirmed via the offline dedup pass (`enrichment/dedup.py`,
    # `scripts/run_dedup_review.py --apply`).
    run = current_run()

    normalized_name = normalize(node_name)
    aliases = load_aliases()
    if normalized_name in aliases:
        if run:
            run.event("resolve_tier", tier="alias", name=node_name, matched=True)
        return aliases[normalized_name]

    # 1. Exact-match fast path (cheap, avoids embedding calls for repeats).
    # Type-scoped when a type is known, so identical names of different
    # types (e.g. a "person" and a "tool" both called "Postgres") don't
    # cross-match, and the candidate pool stays small.
    # (id, name) tuples, not full Node objects: this tier only ever reads
    # .name to compare and .id to return. Fetching whole nodes here meant
    # deserializing every node's 768-float embedding and building a Node
    # object for each, on every single resolve_entity call -- ~2.5s on the
    # query path (node_type=None over 2167 nodes) and a few hundred ms on
    # each of ~5k calls per ingest. See KnowledgeGarden.get_node_identities().
    for candidate_id, candidate_name in kg.get_node_identities(node_type):
        if normalize(candidate_name) == normalized_name:
            if run:
                run.event("resolve_tier", tier="exact_match", name=node_name, matched=True)
            return candidate_id

    # 2. Embed the name (+ source-text context, when available) and search
    # for nearby candidates. See embedding_text()'s docstring for why bare
    # names alone are an insufficient signal here.
    embedding = client.embed(embedding_text(node_name, context))
    candidates = kg.find_similar_nodes(embedding, entity_type=node_type, k=CANDIDATE_K)
    if not candidates:
        if run:
            run.event("resolve_tier", tier="embedding_no_candidates", name=node_name, matched=False)
        return None

    top_distance = candidates[0][1]

    # 3. Distinct: nothing close enough to even ask about.
    if top_distance > NO_MATCH_DISTANCE:
        if run:
            run.event("resolve_tier", tier="embedding_no_match", name=node_name,
                       matched=False, top_distance=top_distance)
        return None

    # 4. Everything else goes to the LLM — no blind auto-merge tier. See the
    # module comment above for why.
    match_id = confirm_match(node_name, node_type, candidates, client)

    # Persist a confirmed match as an alias, so the next occurrence of this
    # exact spelling resolves at tier 0 (a dict lookup) instead of paying the
    # embed + LLM-confirm round trip again -- ~29s on the local model, and the
    # same variant recurs constantly across episodes. This is what makes the
    # resolver get cheaper as the graph matures rather than paying full price
    # forever; previously the confirmation was computed and thrown away, and
    # only the offline dedup pass ever wrote aliases.
    #
    # Accepted tradeoff: a wrong confirmation becomes sticky rather than being
    # re-decided next time. Judged worth it because the alternative is re-asking
    # the same model the same question and most likely getting the same answer,
    # just slower. Aliases are plain JSON in .local/entity_aliases.json --
    # hand-editable, and `scripts/run_dedup_review.py` is the corrective pass.
    if match_id is not None:
        save_alias(normalized_name, match_id)

    if run:
        run.event("resolve_tier", tier="llm_confirm", name=node_name,
                   matched=match_id is not None, top_distance=top_distance)
    return match_id
