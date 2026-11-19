"""Decay scoring for memory access patterns.

The score is the recency of a memory's last access (or of the remembered event,
if it has never been accessed), on a forgetting curve whose half-life grows with
use:

    effective_halflife = halflife * (1 + strength * ln(1 + access_count))
                                  * (1 + 0.5 * (arousal + surprise))
    decay = 2 ** (-days_since / effective_halflife)

so a memory is at 1.0 right after it is used and falls with time, and memories
that are retrieved often (the spacing effect) or that were emotionally charged
(A2.2) fall more slowly. Semantic (consolidated) memories have a floor (A2.2).

The score is always computed against the *current* time, so callers should use
:func:`decay_for_memory` at ranking time rather than trusting a stored value.
The stored ``decay_score`` column is a cache refreshed during the sleep cycle.

Earlier versions mixed recency with an additive frequency term evaluated at the
moment of access, which meant the stored score never fell with time and fell on
first use (from 1.0 to ~0.63). That made compaction unreachable and ranked used
memories below unused ones.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from agent_memory.config import MEMORY_CONFIG


def _as_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def parse_time(value: str | datetime | None) -> datetime | None:
    """Parse an ISO timestamp (``Z`` suffix allowed) to an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _as_aware(value)
    try:
        return _as_aware(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def compute_decay(
    last_accessed: datetime,
    access_count: int,
    arousal: float = 0.0,
    surprise: float = 0.0,
    is_semantic: bool = False,
    now: datetime | None = None,
    config: dict | None = None,
) -> float:
    """Return a decay score in [0.0, 1.0] (1.0 = just used).

    Args:
        last_accessed: When the memory was last retrieved (or created, if never).
        access_count: Total number of times the memory has been retrieved.
        arousal: Emotional arousal score (0.0 to 1.0).
        surprise: Emotional surprise score (0.0 to 1.0).
        is_semantic: True if this memory is a product of compaction.
        now: Evaluation time (defaults to the current UTC time).
        config: Configuration dict (defaults to ``MEMORY_CONFIG``).
    """
    cfg = config or MEMORY_CONFIG
    now = _as_aware(now) if now else datetime.now(timezone.utc)
    last_accessed = _as_aware(last_accessed)
    days_since = max((now - last_accessed).total_seconds() / 86400, 0.0)

    halflife = cfg.get("decay_halflife_days", 7) or 7
    strength = 1.0 + cfg.get("decay_frequency_strength", 0.5) * math.log1p(max(access_count, 0))
    # A2.2: emotional memories decay more slowly; arousal + surprise in [0, 2]
    # so the half-life is stretched by at most 2x.
    emotional = 1.0 + 0.5 * (min(max(arousal, 0.0), 1.0) + min(max(surprise, 0.0), 1.0))

    effective = halflife * strength * emotional
    score = 2.0 ** (-days_since / effective)

    if is_semantic:
        score = max(score, cfg.get("decay_semantic_floor", 0.3))
    return round(min(max(score, 0.0), 1.0), 4)


def decay_for_memory(mem, now: datetime | None = None, config: dict | None = None) -> float:
    """Current decay of a :class:`~agent_memory.models.Memory` (or row-like object).

    Recency is measured from the last access, falling back to when the event
    happened, then to when the memory was created.
    """
    anchor = (
        parse_time(getattr(mem, "last_accessed", None))
        or parse_time(getattr(mem, "event_time", None))
        or parse_time(getattr(mem, "created_at", None))
    )
    if anchor is None:
        return 1.0
    return compute_decay(
        anchor,
        getattr(mem, "access_count", 0) or 0,
        arousal=getattr(mem, "arousal", 0.0) or 0.0,
        surprise=getattr(mem, "surprise", 0.0) or 0.0,
        is_semantic=bool(getattr(mem, "is_semantic", False)),
        now=now,
        config=config,
    )
