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
  developing (they exposed the first bug below). Results are reported on all
  ten; no weights were tuned on the benchmark.
- Aggregate results from this run are in `docs/results/` (numbers only; the
  dataset is not redistributed).

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

## Bugs found while building the benchmark

Two defects were invisible to the unit tests:

1. **Priors were overriding relevance.** Reciprocal-rank fusion (k = 60) has a
   flat top — adjacent ranks differ by about 1.6% after normalisation — and the
   additive importance term (0.05 × salience) was larger than that gap. Inside
   the system, dense recall@1 was 0.07 against 0.15 for the same model in a
   standalone numpy check. Recency/importance/mood are now tie-breaker sized
   (0.02 / 0.01 / 0.01) and the system reproduces the standalone numbers
   exactly. Graph expansion also ran even when a configuration excluded it;
   it is now an opt-in `graph` layer.
2. **Possessives never matched.** `"Ben's"` became the FTS5 phrase `"ben s"`,
   so questions such as *"What is the name of Ben's dog?"* returned nothing
   (caught by the harness's own fixture test; LoCoMo questions are full of
   possessives). Query terms are now split on apostrophes.

## Results

### Evidence recall@k — all 10 conversations, 1,536 questions

| System | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| `raw_bm25` (raw-history baseline) | 0.298 | 0.522 | 0.604 | 0.668 |
| `raw_dense` | 0.156 | 0.353 | 0.444 | 0.543 |
| `raw_hybrid` | 0.242 | 0.473 | 0.575 | 0.698 |
| `curated_only` | 0.268 | 0.507 | 0.597 | 0.706 |
| `full` | 0.247 | 0.508 | 0.607 | 0.722 |
| `raw_hybrid_rerank` (raw-RAG baseline as in arXiv:2610.30755) | 0.462 | 0.666 | 0.727 | 0.765 |
| **`full_rerank`** | **0.463** | **0.670** | **0.733** | **0.776** |

The lexical prior curated 71% of turns in every configuration.

### Recall@10 by question type

| System | single-hop (841) | multi-hop (282) | temporal (321) | open-domain (92) |
|---|---|---|---|---|
| `raw_bm25` | 0.684 | 0.342 | 0.709 | 0.309 |
| `raw_dense` | 0.514 | 0.246 | 0.490 | 0.250 |
| `raw_hybrid` | 0.664 | 0.349 | 0.628 | 0.271 |
| `curated_only` | 0.670 | 0.384 | 0.682 | 0.282 |
| `full` | 0.680 | 0.386 | 0.697 | 0.314 |
| `raw_hybrid_rerank` | 0.821 | 0.485 | 0.785 | 0.406 |
| `full_rerank` | 0.820 | 0.509 | 0.793 | 0.412 |

### What this shows

- **Reranking is the largest single gain.** A MiniLM cross-encoder over the
  fused top 50 lifts recall@1 from 0.298 (the best first stage) to 0.463 and
  recall@5 from 0.522 to 0.670. It is on by default (`reranker_model="auto"`)
  on the full local-model profile.
- **Fusion widens the pool; the reranker cashes it in.** Without reranking,
  fusing everything is *worse* than BM25 alone at rank 1 (0.247 vs 0.298):
  the small dense model is weak on this data (it matches questions to other
  questions) and pulls weaker candidates up. But fusion finds more evidence
  in the top 20 (0.722 vs 0.668), and once reranked that becomes the best
  result at every k.
- **The full system edges out the raw-RAG baseline, mostly on multi-hop.**
  `full_rerank` beats `raw_hybrid_rerank` by 0.001–0.011 overall and by 0.024
  recall@10 on multi-hop questions, where curated keywords and graph links
  connect evidence spread across sessions. The margin is small, consistent
  with the raw-RAG study: curation is not where most retrieval quality comes
  from. This run has no LLM curation (no summaries, LLM keywords or
  entities), so it understates what the curated layers can add — and it does
  not test it either.
- **Indexing every turn costs nothing and removes a failure mode.**
  `curated_only` — the reach of the earlier architecture, where a skipped
  turn was unrecoverable — is behind `full` at every k, and its gap would
  grow with a stricter save policy.
- **Dense retrieval needs a stronger model than MiniLM-L6 here.** On its own
  it trails BM25 at every k. The raw-RAG study used Qwen3-Embedding-4B and
  text-embedding-3-small; swapping the embedder is a one-line change
  (`--embedder <model>`), not tested here.

### Caveats

- Retrieval recall, not QA accuracy: a reader must still use the evidence.
- One benchmark of synthetic two-person conversations; LongMemEval, which has
  knowledge-update and abstention questions, is the natural next harness.
- `seconds` in the JSON output are not comparable across systems: some runs
  shared the CPU with the decision-model experiment.
- Small local models throughout (MiniLM embedder and cross-encoder, CPU).
- Measured at commit `5168f2b`. Review fixes merged afterwards keep raw-only
  configurations free of curation effects and make tokenisation Unicode-aware;
  `raw_bm25` re-measured afterwards gives 0.299 / 0.522 / 0.604 / 0.668
  (recall@1 +0.001, the rest unchanged). The other systems were not re-run.

The decision-model experiment is reported in
[DECISION_MODEL.md](DECISION_MODEL.md#6-evidence-locomo-10-conversations-5882-turns).
