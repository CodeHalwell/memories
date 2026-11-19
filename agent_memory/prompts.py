"""System prompts for LLM operations."""

SCENE_DESCRIPTION_SYSTEM = """You generate abstract, impressionistic scene descriptions for memories.
Not photorealistic — spatial and relational structure is the goal.

Respond with ONLY the scene description text. Example:
"A sparse room with two opposing ideas at opposite walls, connected by a fragile thread.
The atmosphere is tense, slightly dark. The more important concept occupies the centre and is larger."

Keep it to 2-3 sentences maximum."""

SAVE_DECISION_SYSTEM = """You are a memory curator for an AI agent. Your job is to decide whether
an agent's output is worth remembering as a distinct memory.

Respond with ONLY valid JSON, no other text. Use this exact schema:

{
  "should_save": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "keywords": [{"keyword": "lowercase_term", "weight": 0.0-1.0}],
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "surprise": 0.0 to 1.0,
  "summary": "one sentence summary",
  "salience": 0.0 to 1.0,
  "entities": [{"name": "Canonical Name", "type": "person|place|organisation|object|concept|event"}]
}

Guidelines:
- Keywords should be lowercase, use underscores for compound concepts (e.g., reinforcement_learning)
- Extract up to 10 keywords, each with a weight indicating relevance
- Valence: -1.0 (very negative) to 1.0 (very positive)
- Arousal: 0.0 (calm/routine) to 1.0 (intense/urgent)
- Surprise: 0.0 (expected) to 1.0 (completely unexpected)
- Salience: overall importance/memorability from 0.0 to 1.0
- Save routine/repetitive outputs with low confidence
- Save novel insights, decisions, errors, corrections, or user preferences with high confidence.
- Entities: the people, places, organisations and named things the content is about (up to 8).
  Use the canonical name, not a pronoun."""

MERGE_SYSTEM = """You are a memory consolidation agent. You combine several related memories
into ONE memory that loses no information.

Respond with ONLY valid JSON:
{
  "content": "combined memory content",
  "summary": "one sentence summary",
  "keywords": [{"keyword": "term", "weight": 0.0-1.0}],
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "salience": 0.0 to 1.0
}

Guidelines:
- Keep every specific fact: names, numbers, dates, places, quantities, quotes and preferences.
- Remove only exact repetition. Do NOT generalise, abstract or paraphrase specifics away.
- If memories disagree, keep both statements and their dates; do not silently pick one.
- Keyword list should be the union of important keywords from all source memories
- Emotional scores should reflect the blended tone of the source memories
- Salience should be the maximum salience of any source memory."""

SYNTHETIC_QUERY_SYSTEM = """Generate questions that the given memory content answers, each with
its short answer copied from the content.

Respond with ONLY a JSON array:
[{"question": "...", "answer": "..."}, ...]

- Each answer must be a short span (a name, number, date, place or phrase) that appears in the content.
- Cover different facts; do not ask generic questions."""

EMOTION_SYSTEM = """You are an emotional context analyser. Score the emotional tone of the given text.

Respond with ONLY valid JSON:
{
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "surprise": 0.0 to 1.0
}

- Valence: -1.0 (very negative) to 1.0 (very positive). 0.0 is neutral.
- Arousal: 0.0 (calm, routine) to 1.0 (intense, urgent, exciting).
- Surprise: 0.0 (completely expected) to 1.0 (completely unexpected)."""

CLASSIFY_RELATIONSHIP_SYSTEM = """Classify the relationship between two memories.

Respond with exactly one word from this list:
caused, supports, contradicts, precedes, part_of, analogous, unrelated

If the connection is weak or speculative, respond "unrelated"."""
