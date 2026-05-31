# Agent Memory System

A local-first, cognitively-inspired memory system for AI agents. Built on Complementary Learning Systems (CLS) theory, it provides persistent, searchable, and self-organizing memory with emotional modulation, graph-based relationships, and sleep-inspired consolidation.

## Architecture

The system uses four complementary storage layers:

| Layer | Technology | Role |
|-------|-----------|------|
| **Raw Logs** | Append-only JSONL | Immutable conversation history with byte-offset indexing |
| **Metadata** | SQLite (async) | Memory records, access tracking, compaction history, policy decisions |
| **Graph** | Kuzu (embedded) | Relationships between memories and entities |
| **Vector** | Qdrant (embedded) | Semantic search via text embeddings, visual search via CLIP |

## Features

### Memory Saving
- **LLM-driven decisions** — Each conversation turn is evaluated by an LLM to decide whether it should be saved as a memory, with structured confidence scoring
- **Fast-path bypass** — High-arousal or high-surprise events are saved immediately without LLM evaluation
- **Retrieval gap awareness (A2.1)** — When the system detects a gap in its knowledge (recent retrieval misses), the save threshold is lowered to capture more information in that area

### Retrieval
- **Three-layer search** — Grep (ripgrep subprocess), keyword (SQLite FTS), and semantic (Qdrant vector similarity) layers run in parallel, results are merged
- **Graph traversal** — Related memories are discovered via graph walks up to configurable depth
- **Mood-congruent weighting** — Emotionally similar memories are boosted based on the current emotional context
- **Decay scoring** — Memory relevance decays over time using an exponential model with recency and frequency components

### Emotional Modulation (A2.2)
- **Arousal and surprise slow decay** — High-arousal or surprising memories decay more slowly, reflecting the psychological finding that emotional events are better remembered
- **Semantic floor** — Semantic (high-generation) memories never decay below a floor of 0.3, preserving distilled knowledge

### Compaction
- **Keyword-overlap grouping** — Memories sharing sufficient keyword overlap are candidates for merging
- **Valence exclusion** — Emotionally opposed memories (large valence delta) are never merged together
- **Generation gap guard (A2.3)** — Only memories within 1 generation of each other can be merged, preventing loss of abstraction hierarchy
- **Merge validation via generative replay (A2.4)** — Before committing a merge, synthetic queries test whether the merged memory retrieves as well as the originals; merges that degrade retrieval beyond tolerance are rejected
- **Tiering** — Memories flow through hot → warm → cold tiers based on age, access patterns, and generation
- **Lineage tracking** — Full `EVOLVED_FROM` graph edges preserve compaction history

### Keyword Reweighting (A2.5)
- Keywords shared by graph-connected memories are boosted in weight, improving retrieval for strongly related topics

### Dream Exploration (A3)
- **Random walk strategy** — Semi-random graph traversals discover latent connections between memories from different sessions
- **Cluster bridge strategy** — Identifies memories at the boundary of semantic clusters and tests cross-cluster connections
- **Edge commitment** — Discovered relationships are committed to the graph with full logging of exploration runs

### Policy Logging (A4)
- **Decision-outcome pairing** — Save and retrieval decisions are logged with outcomes to enable future policy learning
- **Outcome assessment** — Periodic assessment marks whether saved memories were actually useful (accessed) and whether retrievals were helpful (no immediate re-query on same topic)
- **Training data export** — JSONL export of labeled decision data for future RL-based policy training

### Policy Controller Stub (A5)
- Heuristic-based controller providing `should_save`, `retrieval_config`, and `compaction_priority` methods
- Hard constraints enforced: never delete raw logs, never compact fast-path gen-0 memories
- Designed as a drop-in point for a future learned policy (v2)

### Visual/Spatial Layer
- **CLIP embeddings** — Images are embedded using OpenCLIP for cross-modal retrieval
- **Dual-code search** — Text queries can retrieve visual memories and vice versa

## Project Structure

```
agent_memory/
├── config.py                  # Central configuration
├── models.py                  # Data models (Memory, SaveDecision, etc.)
├── core/
│   ├── memory_manager.py      # Main orchestrator
│   ├── save_decision.py       # LLM-driven save logic with gap awareness
│   ├── retrieval.py           # Three-layer retrieval with mood weighting
│   ├── compaction.py          # Merge, tier, validate with generative replay
│   ├── decay.py               # Exponential decay with emotional modulation
│   ├── keyword_reweight.py    # Graph-informed keyword boosting
│   └── dream_explorer.py      # Exploratory graph walks
├── storage/
│   ├── jsonl_log.py           # Immutable JSONL log with byte-offset index
│   ├── sqlite_store.py        # Async SQLite for metadata and policy data
│   ├── graph_store.py         # Kuzu graph for relationships
│   └── vector_store.py        # Qdrant vectors for semantic search
├── embeddings/
│   ├── text_embedder.py       # Sentence-transformer embeddings
│   └── visual_embedder.py     # CLIP embeddings for images
├── emotion/
│   └── scorer.py              # Emotional dimension scoring
├── llm/
│   └── client.py              # LiteLLM provider-agnostic inference
└── policy/
    ├── controller.py           # Heuristic policy controller (v1)
    ├── outcome_assessor.py     # Decision-outcome assessment
    └── export.py               # Training data export
```

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
work, with the semantic/visual layers skipped and retrieval falling back to the
grep + keyword (incl. content substring) + graph layers.

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

Tools: `memory_save`, `memory_retrieve`, `memory_get`, `memory_compact`.
Configure via `AGENT_MEMORY_DATA_DIR` and `AGENT_MEMORY_PROFILE` (`full`/`lite`).

### REST service

An HTTP surface for hosted deployments and chatbot connectors:

```bash
pip install -e ".[server]"
uvicorn agent_memory.integrations.rest_server:app
```

Endpoints: `POST /memories`, `POST /retrieve`, `GET /memories/{id}`,
`POST /compact`, `GET /health` — all namespace-aware, same env config as above.

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

> Both retrievers are **async-native** — use `ainvoke` / `aretrieve`. The
> synchronous APIs raise, because the underlying store is bound to the event
> loop it was initialized on.

For programmatic/out-of-process use, `agent_memory.service.MemoryService`
provides the same verbs as transport-agnostic, dict-in/dict-out `async` methods
(shared by the MCP server and future REST/connector adapters).

## Usage

```python
from agent_memory.core.memory_manager import MemoryManager

manager = MemoryManager()
await manager.initialize()

# Process a conversation turn
await manager.process_turn(
    session_id="session-001",
    turn_number=1,
    content="The user asked about Python async patterns...",
    emotional_context={"valence": 0.3, "arousal": 0.2},
)

# Retrieve relevant memories
results = await manager.retrieve(
    query="async patterns in Python",
    emotional_context={"valence": 0.0, "arousal": 0.0},
)

# Run compaction (includes keyword reweighting, dream exploration, outcome assessment)
await manager.run_compaction()

# Clean up
await manager.close()
```

## Configuration

All settings are in `agent_memory/config.py` via the `MEMORY_CONFIG` dictionary. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `save_confidence_threshold` | 0.5 | Minimum LLM confidence to save a memory |
| `fast_path_arousal` | 0.85 | Arousal threshold for fast-path save bypass |
| `gap_threshold_reduction` | 0.7 | Multiplier applied to save threshold when a retrieval gap is detected |
| `decay_halflife_days` | 7 | Half-life for memory decay scoring |
| `keyword_overlap_merge_threshold` | 0.6 | Minimum keyword overlap to consider merging |
| `max_generation_gap_for_merge` | 1 | Maximum generation gap allowed for merging |
| `merge_degradation_tolerance` | 0.15 | Maximum retrieval quality degradation allowed for a merge |
| `dream_walk_count` | 50 | Number of random walks per dream exploration run |
| `dream_similarity_threshold` | 0.7 | Minimum similarity for a discovered edge |
| `hot_tier_threshold` | 500 | Hot-tier count that triggers compaction |

## Testing

```bash
pytest
```

93 tests covering all subsystems: storage layers, save decisions, retrieval, compaction, decay, keyword reweighting, dream exploration, policy logging, and policy controller.

## License

See repository for license details.
