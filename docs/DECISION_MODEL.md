# The save-decision model

> Status: implemented and evaluated · 2026-11-19

This document describes how the memory system decides which turns to curate,
how it learns that decision from its own usage, and what the LoCoMo
experiments say about when learning helps.

## 1. What is being decided

Every turn is appended to the JSONL log and indexed in the **raw-turn index**
(BM25, plus a dense vector when embeddings are loaded). The decision is
therefore *not* "keep or lose" — it is whether to **curate** a turn into a
memory: summary, keywords, entities, emotional scores, graph links, eligibility
for consolidation, and a place in the curated-only layers.

That reframing matters for two reasons:

- **A wrong skip is recoverable.** Earlier versions dropped skipped turns from
  every retrievable store; a learned policy making that call would have been a
  data-loss risk. The research consensus (e.g. the raw-RAG study,
  arXiv:2610.30755, and Microsoft's human-inspired memory architecture,
  arXiv:2605.08538) is that preserving raw evidence is what matters most for
  recall.
- **Skips can be labelled.** Because skipped turns stay searchable, we can
  observe whether they were needed later — so training data is not limited to
  the turns the policy already chose to keep.

## 2. How a decision is made

`PolicyController.decide_save(features)` (`agent_memory/policy/controller.py`):

1. **Hard rules** — cannot be overridden by any model:
   - explicit instructions ("remember this", "don't forget", "note that", …);
   - the first turn of a session (`always_save_first_turn`, kept for
     compatibility; detected from SQLite, so it survives restarts);
   - high arousal *and* high surprise (when an LLM scored them).
2. **The active model**, in order of preference:
   - a **promoted learned model** (`LogisticDecisionModel`);
   - the **LLM heuristic** — the curator's `should_save` and `confidence`
     against `save_confidence_threshold` (lowered when the content overlaps a
     recent retrieval gap, A2.1);
   - the **lexical prior** — a hand-set logistic over cheap features, used
     when no LLM is available.

All rule checks happen *before* any LLM call. Each decision is logged in
`save_decisions` with its feature vector, probability, threshold and source
(`rule:explicit`, `heuristic:llm`, `learned:<version>`, …).

### Features (`decision_model.FEATURE_NAMES`)

All computed at decision time, so the model is trained on exactly what it
sees in deployment:

| Group | Features |
|---|---|
| Lexical | log length, content-word count, has number, has date reference, first-person fact, preference language, is question, is acknowledgement |
| Context | speaker is user, first turn, turn position, **novelty** (1 − Jaccard with the most similar curated memory), retrieval-gap overlap, recent save rate in the session, explicit marker |
| LLM (zero when absent) | available, should_save, confidence, salience, arousal, surprise |

## 3. How outcomes are labelled

`policy/outcome_assessor.py`, run in every sleep cycle:

- **Replay relevance.** Later queries in the same namespace are replayed
  against the raw-turn layers only (so curated and skipped turns are treated
  identically), through the manager's retrieval engine — reranked when a
  reranker is configured. A turn that ranks within `outcome_useful_rank` for a
  query made *after* its decision is labelled useful.
- **Usage attribution.** When the agent's reply is recorded
  (`record_response`, called automatically by `ChatConnector`, exposed as MCP
  `memory_feedback` and REST `POST /feedback`), retrieved items whose content
  words appear in the reply are marked *used*; their source turns — including
  turns consolidated into a merged memory — are labelled useful. Callers that
  know exactly what they used can call `mark_used` instead.
- Decisions with no evidence of use are labelled not-useful once older than
  `save_outcome_lookback_days`.

Retrieval decisions are labelled helpful when attribution says something was
used; otherwise by the absence of a re-query on the same topic (within N turns,
or N minutes when the caller does not pass turns).

## 4. Training and promotion

`policy/training.py`:

1. Labelled decisions with features, in time order.
2. The newest `policy_validation_fraction` (25%) is held out.
3. L2 logistic regression by Newton–Raphson, class-balanced (pure numpy; no
   new dependency). The threshold is chosen on the *training* split to
   maximise F-beta (beta = 2: a missed fact costs more than an extra memory)
   within the save-rate bounds.
4. The full learned policy (rules + model) and the policy that actually made
   those decisions are scored on the held-out rows.
5. **Promotion** writes `data/policy/save_model.json` only if the learned
   policy's F-beta beats the deployed one by `policy_promotion_margin` and its
   AUC is at least `policy_min_auc`. Otherwise the model is written as a
   candidate. Every run is recorded in `policy_training_runs`.

Training runs automatically in the sleep cycle once
`policy_min_save_examples` labelled decisions exist (and again when that
count grows by 20%), or on demand with `MemoryManager.train_policy()`.

## 5. Relation to the literature

- **Memory-R1** (arXiv:2508.19828), **Mem-α** (arXiv:2509.25911) and later
  RL work learn memory operations end to end with a QA reward. That needs a
  reader model and gold answers in the loop; it is the right direction when
  those exist. This implementation takes the cheap, offline-learnable slice:
  a supervised model over decision-time features with retrieval-derived
  labels, behind hard rules and a promotion gate.
- **MEMOIR** (arXiv:2611.07631) argues for "attribute usefulness after use,
  screen for preservation before compression". The usage attribution here is
  its lexical TRACE-FB variant; the merge validator is its preservation
  screen (see `core/compaction.py`).
- **Generative Agents** (Park et al., 2023) score importance at write time
  and never revise it. Here importance is a learned function of features and
  observed use, revisable every sleep cycle.

## 6. Evidence

_Pending: results from `python -m agent_memory.eval.locomo --experiment
policy` will be added here._
