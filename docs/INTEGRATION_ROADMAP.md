# Integration Roadmap

> **Status:** Planning draft · **Date:** 2026-05-30 · **Owner:** core team
>
> This document reviews the current state of the Agent Memory System and lays out
> a staged plan for embedding it into the systems people actually run agents on:
> agent frameworks, edge/embedded devices, chatbots, and hosted services.

---

## Progress log

- **2026-05-30 — Phase 0 (foundations) + first slice of Phase 1 landed:**
  - ✅ **§2.1 Public API** — `agent_memory/__init__.py` now exports
    `MemoryManager`, the data models, `MEMORY_CONFIG`, and `__version__`.
  - ✅ **§2.2 Dependency tiering** — heavy imports (kuzu, qdrant, torch/CLIP,
    sentence-transformers, litellm) are deferred to first use; `pyproject`
    extras split into `llm`/`text`/`visual`/`graph`/`vectors`/`mcp` plus
    `lite`/`full` bundles. Core install is now `aiosqlite` + `numpy`.
  - ✅ **§4.1 Lite profile (partial)** — `MemoryManager.initialize(load_embeddings=False)`
    runs storage + retrieval without the embedding stack; retrieval falls back
    to grep + keyword (incl. a new SQLite content-substring search) + graph.
  - ✅ **§3.1 MCP server (first cut)** — `agent_memory/integrations/mcp_server.py`
    exposes `memory_save/retrieve/get/compact` over MCP, built on a new
    transport-agnostic `MemoryService` facade (shared with future REST work).
  - ✅ **§2.4 Namespacing** — a `namespace` dimension (default `"default"`)
    threaded through the models, SQLite (with an in-place migration for legacy
    DBs), vector payloads, graph nodes, retrieval, the service facade, and the
    MCP tools. Isolation is enforced at the SQLite query layer with `get_memory`
    as a catch-all safety net, plus the vector payload filter on the full
    profile. Multi-user chatbots and the hosted service can now share one store.
  - ✅ **§5.1 REST service** — `agent_memory/integrations/rest_server.py`, a
    FastAPI app over the same `MemoryService` facade (extra: `server`). Endpoints
    for save/retrieve/get/compact/health, all namespace-aware. This is the HTTP
    surface for hosted deployments and the chatbot connectors in §5.2.
  - Tests: 117 passing (guard tests for the `mcp`/`server` extras skip when the
    extra is installed).

- **2026-05-31 — Phase 2 (framework adapters):**
  - ✅ **§3.2 LangChain adapter** — `agent_memory/integrations/langchain.py`
    provides `AgentMemoryRetriever` (a `BaseRetriever` returning `Document`s from
    the multi-layer, namespace-scoped retrieval) plus `arecord_message` to
    persist LangChain messages as memories (extra: `langchain`). Built on the
    shared `MemoryService`; namespace isolation verified through the adapter.
  - ✅ **§3.2 LlamaIndex adapter** — `agent_memory/integrations/llamaindex.py`
    provides a LlamaIndex `BaseRetriever` returning scored `NodeWithScore`s
    (descending rank score), plus `arecord_message` for `ChatMessage`s (extra:
    `llamaindex`). Same `MemoryService` backing; namespace isolation verified.
  - **Not yet:** provider Protocols (§2.3), more framework adapters (CrewAI/
    AutoGen, §3.2), edge/Rust build (§4.2+), platform connectors (§5.2), store
    scale-out (§5.3).
  - Tests: 125 passing.

---

## 1. Where the project stands today

The system is a **mature, feature-complete memory library** — but it is *only* a
library. It is consumed by importing `MemoryManager` and calling its async
methods in-process.

### What exists

| Area | State | Notes |
|------|-------|-------|
| Core memory pipeline | **Solid** | save-decision → store → retrieve → compact → dream → policy log |
| Storage layers | **Solid** | JSONL (raw), SQLite (metadata), Kuzu (graph), Qdrant (vectors) |
| Cognitive features | **Solid** | emotional decay, generative-replay merge validation, dream exploration, gap-aware saves |
| Tests | **Good** | 93 tests across all subsystems |
| Multi-language ports | **Structural** | Rust, Go, C# skeletons mirror the Python module layout |
| Documentation | **Library-level** | README covers config + in-process usage |

### What is missing for integration

These are the gaps that block the systems named in this roadmap. Every later
section depends on closing them.

1. **No public package surface.** `agent_memory/__init__.py` exports nothing.
   Consumers reach into `agent_memory.core.memory_manager`. There is no stable,
   versioned API contract.
2. **No serving layer.** No REST, gRPC, WebSocket, or MCP endpoint. The system
   cannot be used out-of-process or across a network.
3. **No framework adapters.** Nothing for LangChain, LlamaIndex, CrewAI,
   AutoGen, Pydantic-AI, or the Claude Agent SDK.
4. **Heavy, hard-coupled dependencies.** `torch`, `sentence-transformers`, and
   `open-clip-torch` are imported directly (e.g. `embeddings/visual_embedder.py`).
   This is a multi-hundred-MB install — a non-starter for edge and serverless.
5. **No tenancy model.** Memories are keyed only by `session_id`. There is no
   `user_id` / `agent_id` / `namespace` dimension, so a single store cannot
   safely serve multiple users or agents.
6. **Embedded-only stores.** Kuzu and Qdrant run embedded against a local data
   directory. There is no shared/remote backend option for horizontal scaling.

> **Guiding principle:** the core stays a clean in-process library. Everything
> below is built as *adapters around* that library, not rewrites of it.

---

## 2. Foundational work (prerequisite for everything else)

These items are not optional features — they unblock the three integration
tracks. Do them first.

### 2.1 Stable public API (`__init__.py`)

Export a curated surface so adapters and external code depend on a contract, not
internal module paths:

```python
from agent_memory import MemoryManager, Memory, SaveDecision, MEMORY_CONFIG
```

Add a thin **facade** if needed so the four-method core (`process_turn`,
`retrieve`, `run_compaction`, `get_memory`) is the only thing adapters touch.

### 2.2 Dependency tiering via optional extras

Split `pyproject.toml` so the heavy ML stack is opt-in:

```toml
[project.optional-dependencies]
core    = []                                   # SQLite + JSONL only, no ML
text    = ["sentence-transformers"]            # local text embeddings
visual  = ["open-clip-torch", "torch"]         # CLIP visual layer
remote-embed = ["litellm"]                     # embeddings via API, no torch
graph   = ["kuzu"]
vectors = ["qdrant-client"]
server  = ["fastapi", "uvicorn"]
```

This is the single highest-leverage change. It is what makes edge and serverless
deployment viable, and it lets a chatbot run with `core+remote-embed` and zero
torch.

### 2.3 Pluggable provider interfaces

Define `Protocol`s (interfaces) for the swappable pieces, with the current
implementations as the defaults:

- `TextEmbedder` — local sentence-transformers **or** remote API embeddings.
- `VisualEmbedder` — CLIP **or** no-op (visual layer disabled).
- `VectorStore` — embedded Qdrant **or** remote Qdrant/pgvector.
- `GraphStore` — embedded Kuzu **or** no-op/SQLite-backed adjacency for edge.
- `LLMClient` — already abstracted via LiteLLM; formalize the seam.

The Rust, Go, and C# ports already trend toward interfaces
(`csharp/Llm/ILlmClient.cs`, `rust/src/llm/mod.rs`) — mirror that in Python.

### 2.4 Tenancy / namespacing

Add an optional `namespace` (or `user_id` + `agent_id`) dimension threaded
through `Memory`, the SQLite schema, the graph node properties, and the Qdrant
payload filters. Default to a single `"default"` namespace so existing behavior
is unchanged. **Without this, no shared service or multi-user chatbot is safe.**

---

## 3. Track A — Agent frameworks

**Goal:** make the system a drop-in long-term memory backend for the popular
agent stacks. The integration shape is nearly identical across frameworks:
*after* each turn call `process_turn`; *before* the next LLM call, `retrieve`
and inject results into context.

### 3.1 Model Context Protocol (MCP) server — **highest priority**

An MCP server is the single most leveraged integration: it works with Claude
Desktop/Code, the Agent SDK, and any MCP-capable client *without per-framework
code*. Expose tools:

- `memory_save(content, session_id, namespace?, emotional_context?)`
- `memory_retrieve(query, namespace?, top_k?)`
- `memory_get(memory_id)`
- `memory_compact()` (admin/maintenance)

Ship as `agent-memory-mcp`, a stdio server wrapping `MemoryManager`. This alone
covers a large share of the "agent frameworks + chatbots" surface.

### 3.2 Native adapters

| Framework | Integration point |
|-----------|-------------------|
| **LangChain / LangGraph** | Implement `BaseChatMessageHistory` + a `BaseRetriever`; optionally a LangGraph checkpointer-style memory node |
| **LlamaIndex** | Custom `BaseMemory` / `VectorStoreIndex`-compatible retriever |
| **CrewAI** | Plug into its `memory` slot (short/long/entity memory) |
| **AutoGen** | `Memory` protocol implementation for agents |
| **Pydantic-AI** | Dependency-injected memory tool |
| **Claude Agent SDK** | Tool definitions backed by the MCP server in 3.1 |

Each adapter is a thin translation layer — framework message objects ↔
`RawLogEntry` / `Memory`. Keep them in a separate `agent_memory_integrations`
package so the core has zero framework dependencies.

### 3.3 Acceptance

A reference example per framework showing: agent runs, memories persist across
sessions, and retrieval visibly improves a multi-session conversation.

---

## 4. Track B — Edge & embedded computing

**Goal:** run on resource-constrained or offline targets — IoT gateways,
on-device assistants, robotics, air-gapped environments.

### 4.1 The "lite" profile

Compose the foundational work into a shippable minimal build:

- **No torch.** Use `remote-embed` (API embeddings) *or* a quantized local
  model (e.g. ONNX / GGUF MiniLM via `fastembed`/`onnxruntime`).
- **Visual layer off** by default.
- **Graph optional.** Provide a SQLite-backed adjacency fallback so Kuzu is not
  required on tiny targets.
- **SQLite + JSONL only** as the irreducible core — both are already
  edge-friendly and dependency-light.

### 4.2 Lean down via the Rust port

Rust is the natural edge target: no GC, small static binary, runs on ARM and
WASM. Prioritize completing `rust/` to a working **core+text** profile:

- SQLite via `rusqlite`, vectors via an embedded HNSW crate or `sqlite-vec`.
- Local embeddings via `candle` or ONNX runtime, or remote via HTTP.
- Build targets: `aarch64` (Raspberry Pi / Jetson) and `wasm32` (browser /
  edge functions).

### 4.3 Sync model

Edge nodes should run fully offline and **sync opportunistically** to a central
store when connected. The append-only JSONL log + byte-offset index is already a
good foundation for one-way log shipping; design a reconciliation pass for
memories/graph/vectors (last-write-wins per namespace to start).

### 4.4 Acceptance

A `core+text` build running on a Raspberry-Pi-class device (and/or WASM) doing
save + retrieve with a <100 MB footprint and no GPU.

---

## 5. Track C — Chatbots & hosted service

**Goal:** give chatbots persistent, personalized memory across conversations,
and provide a scalable multi-tenant backend.

### 5.1 REST / WebSocket service

Wrap `MemoryManager` in a FastAPI service (`server` extra) exposing the same
verbs as the MCP server plus health/admin routes. Run compaction/dream on a
background scheduler rather than inline.

### 5.2 Platform connectors

Thin bots for **Discord, Slack, Telegram, and a web widget**. Each maps
`platform user → namespace`, calls `retrieve` before responding and
`process_turn` after. The heavy lifting lives in the service; connectors stay
small.

### 5.3 Scaling the stores

For a hosted multi-tenant deployment the embedded stores become the bottleneck:

- **Vectors:** remote Qdrant cluster or pgvector.
- **Metadata:** Postgres option alongside SQLite (the async `SQLiteStore`
  interface is a good template for a `PostgresStore`).
- **Graph:** managed graph DB, or defer graph features per-tenant.
- **Tenancy:** enforced via the namespace dimension from §2.4 on every query.

The **Go port** is the natural fit for this network service tier (strong
concurrency, easy single-binary deploys); promote `go/` toward a working server
build in parallel.

### 5.4 Acceptance

A multi-tenant service handling concurrent users with namespace isolation,
fronted by at least one working chat connector, with background maintenance.

---

## 6. Suggested sequencing

```
Phase 0  Foundations (§2)            ──►  unblocks everything
         · public API  · dep extras  · provider Protocols  · namespacing

Phase 1  MCP server (§3.1)           ──►  broadest reach, fastest payoff
         + lite profile (§4.1)

Phase 2  Framework adapters (§3.2)   ──►  LangChain/LlamaIndex first
         + REST service (§5.1)

Phase 3  Edge build (§4.2–4.3)       ──►  Rust core+text, sync
         + chat connectors (§5.2)

Phase 4  Scale-out stores (§5.3)     ──►  multi-tenant hosted offering
         + Go service tier
```

**Rationale:** Phase 0 is mandatory plumbing. The MCP server (Phase 1) is the
highest reach-per-effort item and demonstrates the architecture end-to-end
before investing in many bespoke adapters. Edge and scale-out are heavier and
come once the seams are proven.

## 7. Cross-cutting concerns

- **Security & privacy:** memory is sensitive PII. Need per-namespace
  encryption-at-rest options, a delete/forget API (GDPR), and audit logging.
  The existing prompt-injection hardening must extend to any save/retrieve
  exposed over a network.
- **Observability:** structured metrics on save rate, retrieval hit quality
  (the policy logging in `policy/` is a strong starting point), and compaction
  outcomes.
- **Versioning & migrations:** once stores are shared, schema migrations become
  mandatory. Add migration tooling alongside the public API.
- **Port parity:** define which profiles each language targets — Python = full
  reference + frameworks; Rust = edge; Go = service tier; C# = .NET ecosystem —
  rather than chasing 100% parity everywhere.

---

*This is a living plan. Each track should graduate into its own design doc with
concrete interfaces before implementation begins.*
