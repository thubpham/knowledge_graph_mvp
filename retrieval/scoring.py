from datetime import datetime, timezone
from core.schema import Edge

# Structural relations assert a concrete, checkable relationship between two
# specific things ("Alice MEMBER_OF platform-team") and stay true until
# something changes it. DISCUSSED/REFERENCES-family relations mean "these
# co-occurred in the same episode" -- much weaker, and together with USES they
# are the majority of all semantic edges on this graph (measured 2026-07-26:
# REFERENCES 34%, USES 29%, DISCUSSED 13% of 2587 semantic edges), so without
# this split "recent" and "informative" were nearly the same bucket. This
# needed no schema change and no re-ingest: `relation` was already on every
# edge, just never used for scoring.
#
# extraction_schema.py's `relation` field is `str`, not a constrained Literal
# (unlike consolidation's CONSOLIDATION_RELATION), so extraction sometimes
# emits off-vocabulary relations despite the prompt's stated vocabulary
# (observed live: REFERENCED, REFERENCED_BY, MENTIONS, MENTIONS_IN, SENDS_TO,
# WORKED, WORKED_ON -- roughly 1% of edges). RELATION_WEIGHT.get() with a
# default handles that as "weak," matching its likely REFERENCES/DISCUSSED
# origin, rather than crashing or silently scoring it as durable.
RELATION_WEIGHT = {
    "MEMBER_OF": 1.0, "OWNS": 1.0, "DEPENDS_ON": 1.0, "USES": 1.0,
    "AUTHORED": 1.0, "ATTENDED": 1.0, "SCHEDULED": 1.0, "SENT_TO": 1.0,
    "DECIDED": 1.0, "RESOLVED_BY": 1.0,
    "DISCUSSED": 0.6, "REFERENCES": 0.6, "REPORTED": 0.6,
}
_DEFAULT_RELATION_WEIGHT = 0.6

# Authored documents (Notion, Docs) carry more deliberate, dense signal than
# a terse calendar invite or an inbox that's mostly logistics/marketing.
# Consolidation edges are LLM-synthesized from many episodes at once (the
# model already decided a fact was durable enough to promote -- see
# CONSOLIDATION_SYSTEM_PROMPT), so they're weighted above raw episode sources despite
# not being "authored" in the same sense. `None` (every edge created before
# 2026-07-26, when add_edge gained source_type) and any future source not
# listed here both fall through to the neutral 1.0 default via .get() --
# unattributed provenance is treated as average, not penalized, since there's
# no way to know what an untagged legacy edge actually came from.
SOURCE_WEIGHT = {
    "notion_page": 1.0,
    "gdocs_document": 1.0,
    "consolidation": 0.9,
    "gcal_event": 0.7,
    "gmail_message": 0.5,  # further modulated per-message by GMAIL_CATEGORY_WEIGHT
}
_DEFAULT_SOURCE_WEIGHT = 1.0

# Applied ONLY on top of the gmail_message base weight above, keyed by
# gmail_ingester._gmail_source_category()'s output. "personal" and "none" (a
# real human Re:/Fwd: thread Gmail didn't bucket into any CATEGORY_*) get the
# same 1.0 -- both are correspondence, not noise. CATEGORY_PROMOTIONS has no
# entry: the fetcher excludes it at the query level, so it's never ingested at
# all, not merely down-weighted. Measured on this inbox (2026-07-26, 200 of
# 443 last-30d messages): UPDATES 56% of volume and genuinely mixed (real
# conference reminders alongside "spend your $25 Uber Cash"), hence 0.6 rather
# than crushed further.
GMAIL_CATEGORY_WEIGHT = {
    "personal": 1.0,
    "none": 1.0,
    "forums": 0.8,
    "updates": 0.6,
}
_DEFAULT_GMAIL_CATEGORY_WEIGHT = 1.0
# Gmail's own learned-relevance signal, layered on top of the category weight
# rather than replacing it -- an IMPORTANT promotional-ish update and an
# IMPORTANT personal email should both get a boost, just from different
# baselines. Capped at 1.0 so it can raise a message but never exceed the
# ceiling every other source is scored against.
_IMPORTANT_SUFFIX = "_important"
_IMPORTANT_BOOST = 1.2


def _source_weight(edge: Edge) -> float:
    weight = SOURCE_WEIGHT.get(edge.source_type, _DEFAULT_SOURCE_WEIGHT)
    if edge.source_type == "gmail_message" and edge.source_category:
        category = edge.source_category
        important = category.endswith(_IMPORTANT_SUFFIX)
        if important:
            category = category[: -len(_IMPORTANT_SUFFIX)]
        category_weight = GMAIL_CATEGORY_WEIGHT.get(category, _DEFAULT_GMAIL_CATEGORY_WEIGHT)
        weight = weight * category_weight
        if important:
            weight = min(1.0, weight * _IMPORTANT_BOOST)
    return weight


def _naive_utc(dt: datetime) -> datetime:
    """Strip timezone info, converting to UTC first if the datetime is aware."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def score_edge(edge: Edge, now: datetime):
    now = _naive_utc(now)
    if edge.valid_until is not None and _naive_utc(edge.valid_until) < now:
        return 0.0
    age_days = (now - _naive_utc(edge.valid_from)).days
    recency = 1 / (1 + age_days / 365)
    relation_weight = RELATION_WEIGHT.get(edge.relation, _DEFAULT_RELATION_WEIGHT)
    return recency * relation_weight * _source_weight(edge)
