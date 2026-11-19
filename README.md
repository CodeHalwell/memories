# Agent Memory System

A local-first memory system for AI agents. Every turn is logged and searchable;
a decision policy chooses which turns to *curate* into memories (keywords,
entities, emotion, graph links); retrieval fuses lexical, dense, entity and
graph signals; and a sleep cycle deduplicates, consolidates without losing
facts, labels past decisions by whether they were needed, and trains the
decision policy from those labels.

The design borrows from Complementary Learning Systems theory (fast episodic
storage, slow consolidation), but each mechanism is judged against a
raw-history baseline on a public benchmark — see
[docs/EVALUATION.md](docs/EVALUATION.md) and
[docs/DECISION_MODEL.md](docs/DECISION_MODEL.md).

## Architecture

| Layer | Technology | Role |
|-------|-----------|------|
| **Raw logs** | Append-only JSONL | Immutable history (per-namespace files) |
| **Raw-turn index** | SQLite FTS5 (+ optional vectors) | Every turn, curated or not, is retrievable |
| **Metadata** | SQLite (async) | Memories, keywords, entities, access and decision logs |
| **Graph** | Kuzu (embedded, optional) | Similarity links, entity mentions, lineage |
| **Vector** | Qdrant (embedded, optional) | Dense search over memories and raw turns; CLIP visual layer |

## Features

### Saving: log everything, curate selectively
- **Every turn is indexed.** A "skip" means *not curated*, never *lost*: the
  turn stays in the raw-turn index and can be retrieved, labelled and
  deleted like any other item.
- **Decision policy** (`agent_memory.policy`). Hard rules first (explicit
  "remember this" instructions, the first turn of a session, high arousal and
  surprise), checked before any LLM call. Then the active model: a promoted
  learned model, the LLM curator's confidence, or — with no LLM — a lexical
  prior. Every decision is logged with the features the policy saw.
- **Retrieval gap awareness (A2.1)** — queries that found nothing lower the
  save threshold for content on that topic.
- **Entities** extracted by the curator are indexed for search and linked in
  the graph; similar memories are linked at write time.

### Retrieval
- **Layers**, combined with reciprocal-rank fusion: BM25 over curated
  memories (`keyword`), BM25 over raw turns (`raw`), dense over memories
  (`semantic`) and raw turns (`raw_semantic`), entity matches (`entity`),
  graph neighbours of the top results (`graph`), and optionally ripgrep
  (`grep`) and CLIP (`visual`).
- **Reranking** — a cross-encoder rescoring the fused top 50; on by default
  on the full local-model profile. On LoCoMo this was the largest single gain.
- **Recency, importance and mood** are tie-breakers on top of relevance, so
  decay can reorder near-equals but never hides a relevant memory.
- **Time filters** (`since`/`until`) and dated context lines for the reader.
- **Feedback**: each retrieval returns a `retrieval_id`; `record_response`
  attributes which items a reply used.

### Decay (A2.2)
- Recency of last use (or of the event), on a forgetting curve whose
  half-life grows with each access (spacing effect) and with arousal and
  surprise. 1.0 when just used. Consolidated memories have a floor.

### Sleep cycle (`run_compaction`)
1. **Refresh** cached decay scores for the current time.
2. **Deduplicate** near-identical memories within a namespace (no LLM).
3. **Consolidate** low-value, keyword-overlapping memories within one
   namespace, only if the merge passes validation (A2.4): generated
   question/answer pairs must still be answerable from the merged text, and
   embedding degradation must stay within tolerance (A2.3 generation guard
   still applies). Sources move to the cold tier with lineage edges.
4. **Keyword reweighting** from graph connectivity (A2.5), idempotent.
5. **Dream exploration** (A3): nearest-neighbour bridges between sessions,
   classified by the LLM.
6. **Outcome assessment** (A4): label past save/skip decisions by replaying
   later queries against the raw-turn index, plus usage attribution.
7. **Policy training** (A5): when enough labels exist, train a logistic model
   and promote it only if it beats the deployed policy on held-out recent
   decisions.

### Privacy
- `delete_memory(id, purge_raw=True)` and `purge_namespace(ns)` remove data
  from every store; tenants never share log files.

### Visual/Spatial Layer
- **CLIP embeddings** of LLM-generated scene descriptions for salient
  memories (optional `visual` extra).

## Project Structure

```
agent_memory/
├── config.py                  # Central configuration (+ make_config overrides)
├── models.py                  # Data models (Memory, SaveDecision, ...)
├── text.py                    # Tokenisation, stopwords, FTS queries, lexical features
├── rerank.py                  # Reranker protocol + cross-encoder
├── core/
│   ├── memory_manager.py      # Main orchestrator
│   ├── save_decision.py       # Rules, LLM evaluation, features -> policy
│   ├── retrieval.py           # Layers, rank fusion, graph expansion, rerank
│   ├── compaction.py          # Dedupe, validated consolidation, tiering
│   ├── decay.py               # Forgetting curve with spacing/emotion
│   ├── keyword_reweight.py    # Graph-informed keyword boosting
│   └── dream_explorer.py      # Cross-session association discovery
├── storage/                   # JSONL, SQLite (+FTS5), Kuzu graph, Qdrant vectors
├── embeddings/                # Embedder protocols, hashing/callable/ST/CLIP
├── emotion/scorer.py          # Emotional dimension scoring
├── llm/client.py              # LiteLLM provider-agnostic inference
├── policy/
│   ├── controller.py          # Hard rules + active decision model
│   ├── decision_model.py      # Features, heuristic and logistic models
│   ├── outcome_assessor.py    # Replay labels, usage attribution
│   ├── training.py            # Training, metrics, promotion gate
│   └── export.py              # Training data export
├── eval/locomo.py             # LoCoMo retrieval + decision-model benchmark
└── integrations/              # MCP, REST, LangChain, LlamaIndex, AutoGen, chat
```

> The Rust, Go and C# ports under `rust/`, `go/` and `csharp/` predate
> namespacing, the raw-turn index and the decision model; the Python package
> is the reference implementation.

## Installation

Requires Python 3.11+. Dependencies are tiered so you install only what a given
deployment needs — the core install is lightweight (SQLite + JSONL + numpy) and
every heavy backend is an opt-in extra. Heavy imports are deferred at runtime,
so importing the library never requires an extra you haven't installed.

```bash
pip install -e ".[full]"     # everything (prior default behaviour)
pip install -e ".[lite]"     # full features without torch/CLIP (no visual layer)
pip install -e ".[dev]"      # full + mcp + test tooling
```

### Dependency extras

| Extra | Pulls in | Enables |
|-------|----------|---------|
| *(core)* | `aiosqlite`, `numpy` | raw logs + metadata + scoring |
| `llm` | `litellm` | LLM-driven save/merge/emotion scoring |
| `text` | `sentence-transformers` | local text embeddings (semantic search) |
| `visual` | `open-clip-torch` (torch) | CLIP visual/spatial layer |
| `graph` | `kuzu` | embedded graph relationships |
| `vectors` | `qdrant-client` | embedded vector store |
| `mcp` | `mcp` | Model Context Protocol server |
| `lite` | llm + text + graph + vectors | full features, no torch/CLIP |
| `full` | + visual | everything |

### Lite / edge profile

For edge, serverless, or air-gapped targets you can run without the embedding
stack. Initialize with `load_embeddings=False`: storage and retrieval still
work, with the vector layers skipped and retrieval using BM25 over memories and
raw turns, plus entity and graph layers (the graph is skipped if `kuzu` is not
installed). With no LLM installed, save decisions use the lexical prior.

```python
manager = MemoryManager()
await manager.initialize(load_embeddings=False)   # no torch / no sentence-transformers
```

### Pluggable embedders (torch-free semantic search)

Embedders are injected through small `Protocol`s (`TextEmbedderProtocol` /
`VisualEmbedderProtocol`), so you can keep the semantic vector layer **without
torch** by supplying a lightweight embedder. Bundled implementations:

- `HashingTextEmbedder` — offline, deterministic lexical embeddings (no deps).
- `CallableTextEmbedder` — wrap any `str -> sequence[float]` function (ONNX, a
  remote embeddings API, …).
- `NullVisualEmbedder` — disable the visual layer for text-only deployments.

```python
from agent_memory import MemoryManager, HashingTextEmbedder, NullVisualEmbedder

manager = MemoryManager(
    text_embedder=HashingTextEmbedder(dimension=256),
    visual_embedder=NullVisualEmbedder(),
)
await manager.initialize()        # full semantic (Qdrant) path, no torch/CLIP
```

## Multi-tenancy

Every operation accepts an optional `namespace` (default `"default"`) that
isolates data per tenant — a user, an agent, or any logical partition. Memories,
retrieval, and single-memory lookups are scoped to their namespace, so one store
can safely back many users:

```python
await manager.process_turn(session_id="s1", turn=1, content="...", namespace="user-42")
results = await manager.retrieve("query", namespace="user-42")   # never sees other tenants
```

Isolation is enforced at the SQLite query layer (with `get_memory` as a
catch-all safety net) and the vector payload filter. Databases created before
the namespace column are migrated in place on open, with existing rows defaulting
to `"default"`.

## Integrations

### MCP server

Expose memory to any MCP-capable client (Claude Desktop/Code, the Agent SDK, or
other agent frameworks) with no per-framework code:

```bash
pip install -e ".[mcp]"      # or ".[lite]" / ".[full]" plus ".[mcp]"
python -m agent_memory.integrations.mcp_server
```

Tools: `memory_save`, `memory_retrieve`, `memory_get`, `memory_feedback`,
`memory_delete`, `memory_compact`.
Configure via `AGENT_MEMORY_DATA_DIR` and `AGENT_MEMORY_PROFILE` (`full`/`lite`).

### REST service

An HTTP surface for hosted deployments and chatbot connectors:

```bash
pip install -e ".[server]"
uvicorn agent_memory.integrations.rest_server:app
```

Endpoints: `POST /memories`, `POST /retrieve`, `GET /memories/{id}`,
`DELETE /memories/{id}`, `POST /feedback`, `POST /compact`, `GET /health` — all
namespace-aware, same env config as above.

### LangChain

A `BaseRetriever` (plus message-recording helpers) so a LangChain / LangGraph
agent can use the system as long-term memory:

```bash
pip install -e ".[langchain]"
```

```python
from agent_memory.service import MemoryService
from agent_memory.integrations.langchain import AgentMemoryRetriever, arecord_message

service = MemoryService(load_embeddings=False)
await service.initialize()

retriever = AgentMemoryRetriever(service=service, namespace="user-42")
docs = await retriever.ainvoke("what does the user like?")      # -> list[Document]
await arecord_message(service, ai_message, session_id="s1", turn=3, namespace="user-42")
```

### LlamaIndex

A `BaseRetriever` (plus a message-recording helper) for LlamaIndex query engines
and agents:

```bash
pip install -e ".[llamaindex]"
```

```python
from agent_memory.service import MemoryService
from agent_memory.integrations.llamaindex import AgentMemoryRetriever

service = MemoryService(load_embeddings=False)
await service.initialize()

retriever = AgentMemoryRetriever(service, namespace="user-42")
nodes = await retriever.aretrieve("what does the user like?")   # -> list[NodeWithScore]
```

### AutoGen

An implementation of AutoGen's `Memory` interface — `update_context` injects
relevant memories into the model context before each model call:

```bash
pip install -e ".[autogen]"
```

```python
from agent_memory.integrations.autogen import AgentMemory

memory = AgentMemory(service, namespace="user-42")
# pass to an AutoGen agent: AssistantAgent(..., memory=[memory])
```

> Both retrievers are **async-native** — use `ainvoke` / `aretrieve`. The
> synchronous APIs raise, because the underlying store is bound to the event
> loop it was initialized on.

### Chatbots

`ChatConnector` is a framework-agnostic helper (no platform dependency) for
wiring a chatbot to per-user memory: it maps a platform user to an isolated
namespace, recalls context before a reply, and records turns after.

```python
from agent_memory.service import MemoryService
from agent_memory.integrations import ChatConnector

service = MemoryService()
await service.initialize()
connector = ChatConnector(service, namespace_prefix="discord")

# In your platform's message handler:
context = await connector.context_block(user_id, message_text)   # "" if nothing relevant
reply = await my_llm(system=context, user=message_text)
await connector.record_assistant_message(user_id, reply)
```

Each user is isolated in their own namespace; the same pattern backs a Discord,
Slack, Telegram, or web binding. Context lines carry the event date, and
`record_assistant_message` attributes the reply to the preceding recall, which
is the outcome signal the decision policy learns from.

For programmatic/out-of-process use, `agent_memory.service.MemoryService`
provides the same verbs as transport-agnostic, dict-in/dict-out `async` methods
(shared by the MCP server and future REST/connector adapters).

## Usage

```python
from agent_memory import MemoryManager

manager = MemoryManager()
await manager.initialize()

# Log a turn; the policy decides whether to curate it (it is searchable either way)
await manager.process_turn(
    session_id="session-001",
    turn=1,
    content="I moved to Leeds in March and I'm learning the cello.",
    role="user",
    namespace="user-42",
    timestamp="2026-03-14T18:05:00+00:00",   # optional; defaults to now
)

# Retrieve (curated memories and raw turns, rank-fused and reranked)
results = await manager.retrieve("where does the user live?", namespace="user-42")
for m in results:
    print(m.kind, m.event_time, m.content)

# Tell the system what the answer used (the decision policy learns from this)
await manager.record_response(results[0].retrieval_id, "You live in Leeds.")

# Sleep cycle: dedupe, consolidate, reweight, dream, label outcomes, train policy
await manager.run_compaction()

await manager.close()
```

## Configuration

Defaults live in `agent_memory/config.py` (`MEMORY_CONFIG`). Pass overrides
per manager with `MemoryManager(config={...})`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_enabled` | `True` | Set `False` to run with no LLM calls at all |
| `retrieval_layers` | keyword, raw, semantic, raw_semantic, entity, graph | Layers fused per query |
| `retrieval_top_k` | 10 | Results returned |
| `reranker_model` | `"auto"` | Cross-encoder on the full profile; `None` disables |
| `recency_weight` / `importance_weight` | 0.02 / 0.01 | Tie-breakers added to fused relevance |
| `save_confidence_threshold` | 0.5 | LLM-heuristic save threshold |
| `always_save_first_turn` | `True` | Hard rule (kept for compatibility) |
| `decay_halflife_days` | 7 | Base half-life; access and emotion lengthen it |
| `keyword_overlap_merge_threshold` | 0.6 | Keyword overlap to consider merging |
| `merge_min_answer_preservation` | 1.0 | Fraction of generated answers a merge must keep |
| `save_outcome_lookback_days` | 14 | Unused decisions are labelled not-useful after this |
| `outcome_useful_rank` | 1 | Replay rank that counts as "needed" |
| `policy_min_save_examples` | 1000 | Labelled decisions needed before training |
| `policy_auto_train` | `"auto"` | Train (and maybe promote) in the sleep cycle when a reranker labels outcomes |

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite exercises the real save → retrieve → compact flow on the lite and
edge profiles (no LLM, no torch), with an LLM stub where curation or merging
needs one.

### Benchmark

```bash
# download locomo10.json from https://github.com/snap-research/locomo
python -m agent_memory.eval.locomo --data locomo10.json --embedder minilm
python -m agent_memory.eval.locomo --data locomo10.json --experiment policy
```

LLM-free evidence recall@k for each configuration, and a cross-validated
evaluation of the decision model. Results and discussion:
[docs/EVALUATION.md](docs/EVALUATION.md).

## License

See repository for license details.
