"""Small, dependency-free text utilities shared by retrieval, labelling and the
decision model's features.

Everything here is deterministic and LLM-free so it works on every profile.
"""

from __future__ import annotations

import re
from collections import Counter

# Unicode-aware: letters/digits in any script (FTS5's unicode61 tokenizer
# indexes them too), with an optional apostrophe suffix ("ben's").
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do
does doesn't doing don't down during each few for from further get got had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him himself
his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's
me more most mustn't my myself no nor not now of off on once only or other ought our
ours ourselves out over own really same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't yes yet
you you'd you'll you're you've your yours yourself yourselves also like just ok okay
oh yeah hey hi hello thanks thank please sure great good well lot lots much many
one thing things something anything really actually maybe know think going go
""".split())


def tokens(text: str) -> list[str]:
    """Lowercased word tokens (apostrophes kept inside words)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def content_terms(text: str, min_len: int = 3) -> list[str]:
    """Tokens with stopwords and very short tokens removed, in order."""
    return [
        t for t in tokens(text)
        # Non-Latin words (e.g. CJK) are often one or two characters long.
        if t not in STOPWORDS and (len(t) >= min_len or t.isdigit() or not t.isascii())
    ]


def term_set(text: str) -> set[str]:
    return set(content_terms(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(needles: set[str], haystack: set[str]) -> float:
    """Fraction of ``needles`` present in ``haystack`` (1.0 for empty needles)."""
    if not needles:
        return 1.0
    return len(needles & haystack) / len(needles)


def fts_query(text: str, max_terms: int = 24) -> str | None:
    """Build a safe FTS5 MATCH expression (OR of quoted terms) from free text.

    Every term is double-quoted, so user text can never inject FTS operators.
    Returns None when the text has no usable terms.
    """
    seen: list[str] = []
    for term in content_terms(text, min_len=2):
        # Split on apostrophes: "ben's" would otherwise become the FTS phrase
        # "ben s", which never matches "Ben" (possessives are common in questions).
        for t in term.replace('"', "").split("'"):
            if (len(t) >= 2 or not t.isascii()) and t not in STOPWORDS and t not in seen:
                seen.append(t)
        if len(seen) >= max_terms:
            break
    if not seen:
        return None
    return " OR ".join(f'"{t}"' for t in seen)


def lexical_keywords(text: str, limit: int = 8) -> list[tuple[str, float]]:
    """Cheap keyword extraction for when no LLM is available.

    Ranks content terms by frequency, then length; weights are relative to the
    top term. Good enough for compaction grouping and keyword search.
    """
    counts = Counter(content_terms(text, min_len=4))
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[:limit]
    top = ranked[0][1]
    return [(term, round(0.5 + 0.5 * c / top, 3)) for term, c in ranked]


# ── Regex feature detectors (used by the decision model) ──

_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|tomorrow|"
    r"tonight|today|weekend|last (?:week|month|year)|next (?:week|month|year)|"
    r"(?:19|20)\d\d|\d{1,2}/\d{1,2})\b",
    re.IGNORECASE,
)
_FIRST_PERSON_FACT_RE = re.compile(
    r"\b(?:i am|i'm|i was|i have|i've|i had|i live|i work|i moved|i got|i went|"
    r"i started|i finished|i bought|i made|my|mine|we are|we're|we have|our)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:like|love|prefer|favou?rite|hate|dislike|enjoy|allergic|can't stand|"
    r"always|never|usually)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^\s*(?:what|when|where|who|why|how|which|do|does|did|can|could|would|is|are|will)\b",
    re.IGNORECASE,
)
_ACK_RE = re.compile(
    r"^\s*(?:hi|hey|hello|thanks|thank you|ok|okay|cool|great|nice|sure|yes|no|yeah|"
    r"bye|goodbye|lol|haha|wow|awesome|sounds good|got it)\b[\s!.?,]*",
    re.IGNORECASE,
)


NEGATIONS = frozenset("""
not no never nor none nobody nothing nowhere neither without cannot can't don't doesn't
didn't isn't aren't wasn't weren't won't wouldn't shouldn't couldn't hasn't haven't hadn't
""".split())


def negations(text: str) -> set[str]:
    """Polarity-bearing words in ``text`` (stopword filtering would drop them)."""
    return {t for t in tokens(text) if t in NEGATIONS}


def has_number(text: str) -> bool:
    return any(ch.isdigit() for ch in text or "")


def has_date_reference(text: str) -> bool:
    return bool(_DATE_RE.search(text or ""))


def has_first_person_fact(text: str) -> bool:
    return bool(_FIRST_PERSON_FACT_RE.search(text or ""))


def has_preference(text: str) -> bool:
    return bool(_PREFERENCE_RE.search(text or ""))


def is_question(text: str) -> bool:
    t = (text or "").strip()
    return t.endswith("?") or bool(_QUESTION_START_RE.match(t))


def is_acknowledgement(text: str) -> bool:
    """Short greeting/acknowledgement with little informational content."""
    t = (text or "").strip()
    return bool(_ACK_RE.match(t)) and len(content_terms(t)) <= 3
