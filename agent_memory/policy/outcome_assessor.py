"""Decision-outcome pairing for policy training data (A4.3).

Save decisions — both saves *and* skips — are labelled by whether the turn
turned out to be needed:

  - **Replay relevance.** Every later query in the same namespace is replayed
    against the raw-turn BM25 index, which contains every logged turn whether
    or not it was curated. A turn that ranks in the top
    ``outcome_useful_rank`` for a query made after its decision was needed.
    Because the raw index treats curated and skipped turns identically, the
    label does not depend on the decision being judged (earlier versions only
    labelled saves, by "was it ever accessed", which both ignored skips and
    rewarded whatever the policy had already chosen to keep).
  - **Usage attribution.** If a response is recorded for a retrieval
    (``attribute_usage``), retrieved items whose content overlaps the response
    are marked used, and that marks their source turns (including turns
    consolidated into a merged memory) as useful.

Turns with no evidence of use are labelled not-useful once older than the
outcome window. Retrieval decisions are labelled helpful when attribution says
something was used, otherwise by the absence of a re-query on the same topic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from agent_memory.config import MEMORY_CONFIG
from agent_memory.core.decay import parse_time
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.text import content_terms, term_set

logger = logging.getLogger(__name__)


ReplayFn = Callable[[str, str, int], Awaitable[list[str]]]


async def bm25_replay(sqlite: SQLiteStore, query: str, namespace: str, k: int) -> list[str]:
    """Default replay: raw-turn BM25 ranking (raw log ids, best first)."""
    return [rid for rid, _ in await sqlite.search_raw_fts(query, limit=k, namespace=namespace)]


async def assess_save_outcomes(
    sqlite: SQLiteStore,
    lookback_days: int | None = None,
    now: datetime | None = None,
    config: dict | None = None,
    replay: ReplayFn | None = None,
) -> int:
    """Label unlabelled save/skip decisions. Returns the number labelled.

    ``replay(query, namespace, k)`` ranks raw turns for a past query; the
    memory manager passes its own retrieval engine restricted to the raw-turn
    layers (reranked when a reranker is configured). Without it, raw BM25.
    """
    cfg = config or MEMORY_CONFIG
    window = cfg["save_outcome_lookback_days"] if lookback_days is None else lookback_days
    rank_cut = cfg.get("outcome_useful_rank", 5)
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window)

    unlabelled = await sqlite.get_unlabelled_save_decisions()
    if not unlabelled:
        return 0

    by_ns: dict[str, list[dict]] = defaultdict(list)
    for d in unlabelled:
        by_ns[d.get("namespace") or "default"].append(d)

    updates: list[tuple[bool, str, str]] = []
    stamp = now.isoformat()
    for ns, decisions in by_ns.items():
        earliest = min(d["decided_at"] for d in decisions)
        hit_times: dict[str, list[datetime]] = defaultdict(list)
        queries = await sqlite.get_retrieval_queries(namespace=ns, since=earliest)
        # Bound the cost of a sleep cycle: replay the most recent queries.
        queries = queries[-cfg.get("outcome_replay_max_queries", 2000):]
        for q in queries:
            t = parse_time(q["decided_at"])
            ranked = (
                await replay(q["query"], ns, rank_cut) if replay is not None
                else await bm25_replay(sqlite, q["query"], ns, rank_cut)
            )
            for rid in ranked[:rank_cut]:
                hit_times[rid].append(t)
        used = await sqlite.get_used_raw_log_ids(ns)

        for d in decisions:
            decided = parse_time(d["decided_at"])
            useful = d["raw_log_id"] in used or any(
                t is not None and decided is not None and t > decided
                for t in hit_times.get(d["raw_log_id"], [])
            )
            if useful:
                updates.append((True, stamp, d["id"]))
            elif decided is not None and decided <= cutoff:
                updates.append((False, stamp, d["id"]))

    if updates:
        await sqlite.batch_update_save_outcomes(updates)
    logger.info("Save outcome assessment: labelled %d decisions", len(updates))
    return len(updates)


async def assess_retrieval_outcomes(
    sqlite: SQLiteStore,
    followup_turns: int | None = None,
    keyword_overlap_threshold: float | None = None,
    min_age_minutes: int = 60,
    config: dict | None = None,
) -> int:
    """Label retrievals as helpful or not. Returns the number assessed.

    Attribution (a response was recorded) wins; otherwise a retrieval that
    returned nothing is unhelpful, and one followed by a re-query on the same
    topic (within N turns, or N minutes when turns are not tracked) is
    unhelpful.
    """
    cfg = config or MEMORY_CONFIG
    followup_turns = followup_turns or cfg["retrieval_outcome_followup_turns"]
    overlap_threshold = keyword_overlap_threshold or cfg["retrieval_outcome_keyword_overlap"]
    minutes = cfg.get("retrieval_outcome_followup_minutes", 10)
    now = datetime.now(timezone.utc).isoformat()

    unassessed = await sqlite.get_unassessed_retrieval_decisions(min_age_minutes=min_age_minutes)
    updated = 0

    for row in unassessed:
        usage = await sqlite.get_usage_for_retrieval(row["id"])
        if usage:
            helpful = any(usage)
        elif not row.get("return_count"):
            helpful = False
        else:
            if row.get("turn") is not None:
                followups = await sqlite.get_retrieval_followups(
                    row["session_id"], row["turn"], window=followup_turns,
                )
            else:
                followups = await sqlite.get_retrieval_followups_by_time(
                    row["session_id"], row["decided_at"], minutes, exclude_id=row["id"],
                )
            original = set(content_terms(row["query"]))
            re_queried = False
            for fu_query in followups:
                fu = set(content_terms(fu_query))
                if original and fu:
                    overlap = len(original & fu) / max(len(original), 1)
                    if overlap > overlap_threshold:
                        re_queried = True
                        break
            helpful = not re_queried
        await sqlite.update_retrieval_outcome(row["id"], helpful, now)
        updated += 1

    logger.info("Retrieval outcome assessment: assessed %d decisions", updated)
    return updated


async def attribute_usage(
    sqlite: SQLiteStore,
    retrieval_id: str,
    response_text: str,
    config: dict | None = None,
    max_items: int = 10,
) -> list[dict]:
    """Mark which retrieved items a response actually used (lexical attribution).

    An item counts as used when at least ``usage_overlap_threshold`` of its
    content words appear in the response (and at least two, unless the item
    is tiny). Returns the rows recorded.
    """
    cfg = config or MEMORY_CONFIG
    threshold = cfg.get("usage_overlap_threshold", 0.25)
    hits = (await sqlite.get_retrieval_hits(retrieval_id))[:max_items]
    if not hits:
        return []
    response_terms = term_set(response_text)
    mem_ids = [h["memory_id"] for h in hits if h.get("memory_id")]
    raw_ids = [h["raw_log_id"] for h in hits if not h.get("memory_id") and h.get("raw_log_id")]
    mems = await sqlite.get_memories(mem_ids)
    raws = await sqlite.get_raw_turns(raw_ids)

    rows = []
    for h in hits:
        if h.get("memory_id"):
            mem = mems.get(h["memory_id"])
            content = mem.content if mem else ""
        else:
            content = (raws.get(h.get("raw_log_id")) or {}).get("content") or ""
        terms = term_set(content)
        shared = len(terms & response_terms)
        overlap = shared / len(terms) if terms else 0.0
        used = overlap >= threshold and (shared >= 2 or len(terms) <= 2)
        rows.append({
            "memory_id": h.get("memory_id"), "raw_log_id": h.get("raw_log_id"),
            "used": used, "overlap": round(overlap, 4), "source": "lexical",
        })
    await sqlite.log_usage(retrieval_id, rows, datetime.now(timezone.utc).isoformat())
    return rows


async def mark_used(
    sqlite: SQLiteStore, retrieval_id: str, memory_ids: list[str],
) -> list[dict]:
    """Explicit feedback: the caller says which returned items it used.

    ``memory_ids`` are ids as returned by retrieval (``raw:<id>`` for raw turns).
    """
    hits = await sqlite.get_retrieval_hits(retrieval_id)
    chosen = set(memory_ids)
    rows = []
    for h in hits:
        returned_id = h["memory_id"] or f"raw:{h['raw_log_id']}"
        rows.append({
            "memory_id": h.get("memory_id"), "raw_log_id": h.get("raw_log_id"),
            "used": returned_id in chosen, "overlap": None, "source": "explicit",
        })
    await sqlite.log_usage(retrieval_id, rows, datetime.now(timezone.utc).isoformat())
    return rows
