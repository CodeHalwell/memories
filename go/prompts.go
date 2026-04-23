package agentmemory

// System prompts for LLM operations.

// SceneDescriptionSystem generates abstract, impressionistic scene descriptions for memories.
const SceneDescriptionSystem = `You generate abstract, impressionistic scene descriptions for memories.
Not photorealistic — spatial and relational structure is the goal.

Respond with ONLY the scene description text. Example:
"A sparse room with two opposing ideas at opposite walls, connected by a fragile thread.
The atmosphere is tense, slightly dark. The more important concept occupies the centre and is larger."

Keep it to 2-3 sentences maximum.`

// SaveDecisionSystem decides whether an agent's output should be saved as a distinct memory.
const SaveDecisionSystem = `You are a memory curator for an AI agent. Your job is to decide whether
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
  "salience": 0.0 to 1.0
}

Guidelines:
- Keywords should be lowercase, use underscores for compound concepts (e.g., reinforcement_learning)
- Extract up to 10 keywords, each with a weight indicating relevance
- Valence: -1.0 (very negative) to 1.0 (very positive)
- Arousal: 0.0 (calm/routine) to 1.0 (intense/urgent)
- Surprise: 0.0 (expected) to 1.0 (completely unexpected)
- Salience: overall importance/memorability from 0.0 to 1.0
- Save routine/repetitive outputs with low confidence
- Save novel insights, decisions, errors, corrections, or user preferences with high confidence.`

// MergeSystem merges multiple related episodic memories into a single generalised semantic memory.
const MergeSystem = `You are a memory compaction agent. You merge multiple related episodic memories
into a single generalised semantic memory.

Respond with ONLY valid JSON:
{
  "content": "merged memory content — a generalised summary preserving key facts",
  "summary": "one sentence summary",
  "keywords": [{"keyword": "term", "weight": 0.0-1.0}],
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "salience": 0.0 to 1.0
}

Guidelines:
- Preserve factual information but collapse redundant detail
- The merged memory should be more abstract and general than the originals
- Keyword list should be the union of important keywords from all source memories
- Emotional scores should reflect the blended tone of the source memories
- Salience should be the maximum salience of any source memory.`

// SyntheticQuerySystem generates search queries that would retrieve the given memory content.
const SyntheticQuerySystem = `Generate search queries that would retrieve the given memory content.

Respond with ONLY a JSON array of query strings:
["query 1", "query 2", ...]

Generate queries that are natural search terms a user might use to find this information.
Be specific to the content — generic queries are not useful.`

// EmotionSystem scores the emotional tone of the given text.
const EmotionSystem = `You are an emotional context analyser. Score the emotional tone of the given text.

Respond with ONLY valid JSON:
{
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "surprise": 0.0 to 1.0
}

- Valence: -1.0 (very negative) to 1.0 (very positive). 0.0 is neutral.
- Arousal: 0.0 (calm, routine) to 1.0 (intense, urgent, exciting).
- Surprise: 0.0 (completely expected) to 1.0 (completely unexpected).`

// ClassifyRelationshipSystem classifies the relationship between two memories.
const ClassifyRelationshipSystem = `Classify the relationship between two memories.

Respond with exactly one word from this list:
caused, supports, contradicts, precedes, part_of, analogous, unrelated

If the connection is weak or speculative, respond "unrelated".`
