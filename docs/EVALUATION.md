# Evaluation

> Status: first full run · 2026-11-19 · harness: `agent_memory/eval/locomo.py`

## Why this benchmark and metric

The research review found the repository had no measurements, and that the
bar for an agent memory system is now a **well-tuned raw-history baseline**:
dense retrieval plus a reranker over raw turns ranked first against twelve
agentic memory systems in six of seven settings (arXiv:2610.30755). Published
LoCoMo QA scores also move by up to 7 points with the judge alone and 9 with
the answer prompt (arXiv:2611.19199), so they are a poor way to compare
configurations.

LoCoMo (Maharana et al., 2024) annotates every question with the dialogue
turns that contain its evidence. That gives an **LLM-free retrieval metric**:

> **evidence recall@k** — the fraction of a question's evidence turns that
> appear among the top-k retrieved items, averaged over questions.

It needs no answer model and no judge, so it is cheap, deterministic and
reproducible offline, and it isolates the part of the system this repository
controls. It does not measure answer quality: a reader still has to use the
evidence, and LongMemEval shows reading strategy alone can move QA accuracy by
around 10 points.

## Protocol

- Data: `locomo10.json` (10 conversations, 5,882 turns). Scored on the 1,536
  non-adversarial questions that list evidence (4 of the 1,540 list none); the
  446 adversarial questions have no gold evidence and are excluded, as in
  prior work.
- Each conversation is fed turn by turn in timestamp order through
  `MemoryManager.process_turn`, as `"<speaker>: <text>"` with the session
  timestamp, into its own namespace and a fresh data directory. Image turns
  include their caption.
- No LLM is used anywhere (`llm_enabled=False`): curation decisions come from
  the lexical prior, merges and dream exploration do not run. Recency is
  switched off (the conversations are historical, so wall-clock recency is
  meaningless). Everything else is the default configuration.
- Embedder: `all-MiniLM-L6-v2`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Dev/test discipline: conversations `conv-26` and `conv-30` were used while
  developing (they exposed two scoring bugs, below). Results are reported on
  all ten; no weights were tuned on the benchmark.

Reproduce:

```bash
python -m agent_memory.eval.locomo --data locomo10.json --embedder minilm \
    --out results.json
python -m agent_memory.eval.locomo --data locomo10.json --experiment policy
```

## Configurations

| System | What it searches |
|---|---|
| `raw_bm25` | BM25 over every raw turn — the raw-history baseline |
| `raw_dense` | Dense (MiniLM) over every raw turn |
| `raw_hybrid` | Both, rank-fused |
| `curated_only` | Only curated memories (BM25 + dense + entities) — the reach of the architecture before this change, where a skipped turn was unrecoverable |
| `full` | Default: all layers rank-fused, plus graph expansion |
| `raw_hybrid_rerank` | `raw_hybrid` + cross-encoder over the fused top 50 |
| `full_rerank` | `full` + cross-encoder over the fused top 50 |

## Bugs the benchmark found

Two defects were invisible to unit tests and only showed up as numbers:

1. **Priors were overriding relevance.** Reciprocal-rank fusion (k = 60) has a
   flat top — adjacent ranks differ by about 1.6% after normalisation — and the
   additive importance term (0.05 × salience) was larger than that gap. Inside
   the system, dense recall@1 was 0.07 against 0.15 for the same model in a
   standalone numpy check. Recency/importance/mood are now tie-breaker sized
   (0.02 / 0.01 / 0.01) and the system reproduces the standalone numbers
   exactly. Graph expansion also ran even when a configuration excluded it;
   it is now an opt-in `graph` layer.
2. **Possessives never matched.** `"Ben's"` became the FTS5 phrase `"ben s"`,
   so questions such as *"What is the name of Ben's dog?"* returned nothing.
   Query terms are now split on apostrophes.

## Results

_Results pending: the full ten-conversation run is in progress and will be
added here._
