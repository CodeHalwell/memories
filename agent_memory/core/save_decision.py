"""Save decision: should a logged turn be curated into a memory?

Flow for each turn:

  1. Cheap rule checks that need no LLM — explicit "remember this" markers and
     the first turn of a session. (Previously these ran *after* the LLM call, so
     with no LLM available an explicit instruction was still dropped.)
  2. An LLM evaluation (keywords, entities, emotion, summary, should_save,
     confidence), when an LLM is available. Failure is not fatal.
  3. The policy controller decides from a feature vector that combines lexical
     features, novelty against existing memories, retrieval-gap overlap (A2.1)
     and the LLM's judgement. See ``agent_memory.policy``.

Every decision is logged with its features and probability so a learned model
can be trained on exactly what the policy saw. Skipping is never data loss:
the turn stays in the raw log and the raw-turn index.
"""

from __future__ import annotations

import logging

from agent_memory.config import MEMORY_CONFIG
from agent_memory.llm.client import llm_available, llm_complete_json
from agent_memory.models import Memory, RawLogEntry, SaveDecision
from agent_memory.policy.controller import PolicyController
from agent_memory.policy.decision_model import build_save_features
from agent_memory.storage.sqlite_store import SQLiteStore
from agent_memory.text import jaccard, lexical_keywords, term_set

logger = logging.getLogger(__name__)


def has_explicit_marker(content: str, config: dict | None = None) -> bool:
    cfg = config or MEMORY_CONFIG
    lower = (content or "").lower()
    return any(phrase in lower for phrase in cfg.get("explicit_save_markers", []))


def is_fast_path(arousal: float, surprise: float, content: str, config: dict | None = None) -> bool:
    """Rule-based save that bypasses the model (explicit marker or emotional spike)."""
    cfg = config or MEMORY_CONFIG
    if arousal > cfg["fast_path_arousal"] and surprise > cfg["fast_path_surprise"]:
        return True
    return has_explicit_marker(content, cfg)


async def get_retrieval_gaps(
    sqlite: SQLiteStore, session_id: str, namespace: str | None = None,
    config: dict | None = None,
) -> list[str]:
    """Terms from recent queries in this session that found nothing (A2.1)."""
    cfg = config or MEMORY_CONFIG
    return await sqlite.get_failed_retrieval_keywords(
        session_id, lookback=cfg["gap_lookback_turns"], namespace=namespace,
    )


def _compute_gap_overlap(content_keywords: list[str], gap_keywords: list[str]) -> float:
    """Compute overlap between content keywords and retrieval gap keywords."""
    if not content_keywords or not gap_keywords:
        return 0.0
    content_set = set(content_keywords)
    gap_set = set(gap_keywords)
    return len(content_set & gap_set) / max(len(content_set), 1)


async def _llm_evaluate(entry: RawLogEntry, cfg: dict) -> dict | None:
    prompt = f"""Evaluate whether this agent output should be saved as a memory:

Session: {entry.session_id}
Turn: {entry.turn}
Speaker: {entry.role}
Content:
<content>
{entry.content}
</content>

Respond with JSON only."""
    try:
        result = await llm_complete_json(prompt, system=cfg["prompts"]["save_decision"])
    except Exception:
        logger.warning("LLM save evaluation failed; deciding from lexical features")
        return None
    return result if isinstance(result, dict) else None


def _parse_keywords(result: dict | None, content: str, cfg: dict) -> list[tuple[str, float]]:
    keywords: list[tuple[str, float]] = []
    for kw in (result or {}).get("keywords", []) or []:
        try:
            keywords.append((str(kw["keyword"]).lower(), float(kw.get("weight", 1.0))))
        except (KeyError, TypeError, ValueError):
            continue
    if not keywords:
        # No LLM (or it returned none): extract lexically so keyword search and
        # compaction grouping still have something to work with.
        keywords = lexical_keywords(content, limit=min(cfg["max_keywords_per_memory"], 8))
    return keywords[:cfg["max_keywords_per_memory"]]


def _parse_entities(result: dict | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for e in (result or {}).get("entities", []) or []:
        if isinstance(e, dict) and e.get("name"):
            out.append((str(e["name"]).strip(), str(e.get("type", "") or "")))
        elif isinstance(e, str) and e.strip():
            out.append((e.strip(), ""))
    return out[:8]


def _f(result: dict | None, key: str, default: float) -> float:
    try:
        return float((result or {}).get(key, default))
    except (TypeError, ValueError):
        return default


async def make_save_decision(
    entry: RawLogEntry,
    is_first_turn: bool = False,
    sqlite: SQLiteStore | None = None,
    policy: PolicyController | None = None,
    config: dict | None = None,
) -> tuple[SaveDecision, Memory | None]:
    """Decide whether to curate a logged turn into a memory.

    Args:
        entry: The raw log entry to evaluate.
        is_first_turn: Whether this is the first turn of the session.
        sqlite: Store used for gap awareness (A2.1), novelty and save-rate
            features. Optional.
        policy: The policy controller (a default one is created if omitted).
        config: Configuration dict (defaults to ``MEMORY_CONFIG``).

    Returns:
        (SaveDecision, Memory or None).
    """
    cfg = config or MEMORY_CONFIG
    policy = policy or PolicyController(config=cfg)
    explicit = has_explicit_marker(entry.content, cfg)

    # LLM evaluation (skipped entirely when no LLM is configured/installed).
    result = await _llm_evaluate(entry, cfg) if llm_available(cfg) else None
    keywords = _parse_keywords(result, entry.content, cfg)
    content_kw_names = [kw for kw, _ in keywords]

    # Context features from the store.
    threshold = cfg["save_confidence_threshold"]
    gap_triggered = False
    gap_overlap = 0.0
    novelty = 1.0
    recent_rate = 0.0
    if sqlite is not None:
        try:
            gap_keywords = await get_retrieval_gaps(sqlite, entry.session_id, entry.namespace, cfg)
            gap_terms = set(gap_keywords)
            # Compare both extracted keywords and their component words, since
            # LLM keywords use underscores ("reinforcement_learning").
            kw_terms = {part for kw in content_kw_names for part in kw.split("_")}
            gap_overlap = max(
                _compute_gap_overlap(content_kw_names, gap_keywords),
                _compute_gap_overlap(sorted(kw_terms), sorted(gap_terms)),
            )
            if gap_overlap > cfg["gap_overlap_threshold"]:
                threshold *= cfg["gap_threshold_reduction"]
                gap_triggered = True
        except Exception:
            logger.debug("Gap detection failed, using default threshold")
        try:
            similar = await sqlite.most_similar_text(
                entry.content, namespace=entry.namespace, exclude_raw_id=entry.id,
            )
            if similar is not None:
                novelty = 1.0 - jaccard(term_set(entry.content), term_set(similar))
            recent_rate = await sqlite.recent_save_rate(entry.session_id, entry.namespace)
        except Exception:
            logger.debug("Novelty/save-rate features unavailable")

    features = build_save_features(
        entry.content, role=entry.role, turn=entry.turn, is_first_turn=is_first_turn,
        novelty=novelty, gap_overlap=gap_overlap, recent_save_rate=recent_rate,
        explicit_marker=explicit, llm=result,
    )
    verdict = policy.decide_save(features, llm_threshold=threshold)

    reason = verdict.reason
    if result and result.get("reason") and verdict.source == "heuristic:llm":
        reason = str(result["reason"])
    confidence = _f(result, "confidence", verdict.prob) if result else verdict.prob
    if verdict.decision == "fast_path":
        confidence = max(confidence, 0.9)

    dec = SaveDecision(
        raw_log_id=entry.id,
        session_id=entry.session_id,
        turn=entry.turn,
        decision=verdict.decision,
        reason=reason,
        confidence=confidence,
        gap_triggered=gap_triggered,
        threshold_used=verdict.threshold if verdict.decision != "fast_path" else threshold,
        namespace=entry.namespace,
        role=entry.role,
        policy_prob=verdict.prob,
        policy_source=verdict.source,
        features=features,
    )
    if not verdict.save:
        return dec, None

    mem = Memory(
        content=entry.content,
        summary=(result or {}).get("summary"),
        raw_log_id=entry.id,
        namespace=entry.namespace,
        session_id=entry.session_id,
        turn=entry.turn,
        valence=_f(result, "valence", 0.0),
        arousal=_f(result, "arousal", 0.0),
        surprise=_f(result, "surprise", 0.0),
        salience=_f(result, "salience", 0.7 if verdict.decision == "fast_path" else 0.5),
        fast_pathed=verdict.decision == "fast_path",
        keywords=keywords,
        entities=_parse_entities(result),
        role=entry.role,
        event_time=entry.timestamp,
    )
    return dec, mem
