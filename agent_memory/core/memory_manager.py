"""Memory Manager — orchestrates save, retrieve, compact and learn.

This is the primary interface for the agent runtime. It coordinates the
storage backends, embedding models, the decision policy and the sleep cycle.

Write path (``process_turn``):
  every turn is appended to the JSONL log and indexed in the raw-turn index
  (BM25, plus a dense vector when embeddings are loaded); the policy then
  decides whether to *curate* it into a memory (keywords, entities, emotion,
  graph links). Skipping never loses the turn.

Read path (``retrieve``):
  rank-fused retrieval over curated memories and raw turns (see
  ``core/retrieval.py``); the ranked list is logged for outcome labelling.

Sleep cycle (``run_compaction``):
  dedupe/consolidate (A2.3/A2.4), keyword reweighting (A2.5), dream
  exploration (A3), outcome assessment (A4), and — when enough labelled data
  has accumulated — training and gated promotion of the learned save model (A5).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agent_memory.config import DATA_DIR, make_config
from agent_memory.core.compaction import CompactionEngine
from agent_memory.core.decay import compute_decay
from agent_memory.core.dream_explorer import commit_discoveries, exploratory_walk
from agent_memory.core.keyword_reweight import reweight_keywords_from_graph
from agent_memory.core.retrieval import RetrievalEngine
from agent_memory.core.save_decision import make_save_decision
from agent_memory.embeddings.text_embedder import TextEmbedder
from agent_memory.embeddings.visual_embedder import VisualEmbedder
from agent_memory.llm.client import llm_available
from agent_memory.models import CompactionResult, Memory, RawLogEntry
from agent_memory.policy.controller import PolicyController
from agent_memory.policy.outcome_assessor import (
    assess_retrieval_outcomes,
    assess_save_outcomes,
    attribute_usage,
    mark_used,
)
from agent_memory.rerank import reranker_from_config
from agent_memory.storage.graph_store import GraphStore
from agent_memory.storage.jsonl_log import JSONLLogger
from agent_memory.storage.sqlite_store import SQLiteStore, entity_key
from agent_memory.storage.vector_store import VectorStore

if TYPE_CHECKING:
    from agent_memory.embeddings.base import (
        TextEmbedderProtocol,
        VisualEmbedderProtocol,
    )
    from agent_memory.rerank import RerankerProtocol

logger = logging.getLogger(__name__)


class MemoryManager:
    """Top-level orchestrator for the agent memory system."""

    def __init__(
        self,
        data_dir: Path | None = None,
        text_embedder: "TextEmbedderProtocol | None" = None,
        visual_embedder: "VisualEmbedderProtocol | None" = None,
        config: dict | None = None,
        reranker: "RerankerProtocol | None" = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        # A private copy: overrides here never leak into other managers.
        self.config = make_config(config)

        # Storage backends
        self.jsonl = JSONLLogger(self.data_dir / "logs" / "sessions")
        self.sqlite = SQLiteStore(self.data_dir / "memory.db")
        self.graph = GraphStore(self.data_dir / "graph")
        self.vector = VectorStore(self.data_dir / "vectors")
        self.policy_dir = self.data_dir / "policy"

        # Embedding providers (injectable). Defaults are the model-backed
        # wrappers (lazy-loaded). Inject a torch-free embedder (e.g.
        # HashingTextEmbedder + NullVisualEmbedder) to run semantic retrieval on
        # the edge profile without the heavy `text`/`visual` extras.
        self.text_embedder = text_embedder if text_embedder is not None else TextEmbedder()
        self.visual_embedder = visual_embedder if visual_embedder is not None else VisualEmbedder()
        self._visual_enabled = not getattr(self.visual_embedder, "is_null", False)
        self._reranker_arg = reranker
        self.reranker = reranker

        self.policy = PolicyController(config=self.config, model_dir=self.policy_dir)

        # Sub-engines (initialized after storage is ready)
        self._retrieval: RetrievalEngine | None = None
        self._compaction: CompactionEngine | None = None
        self._embeddings_loaded: bool = False
        self._graph_available: bool = False

    async def initialize(self, load_embeddings: bool = True) -> None:
        """Initialize storage backends and the retrieval/compaction engines.

        Args:
            load_embeddings: When True (default), load the text/visual embedding
                models and initialize the vector store — required for the vector
                layers. When False, run on the **lite profile**: retrieval uses
                the BM25 (memory + raw-turn), entity and graph layers only. The
                graph layer is optional too: without the ``graph`` extra it is
                skipped with a warning.
        """
        await self.sqlite.initialize()
        try:
            self.graph.initialize()
            self._graph_available = True
        except ImportError:
            logger.warning("Graph extra not installed; graph layer disabled")
            self._graph_available = False
        if load_embeddings:
            self.vector.initialize(
                text_dim=self.text_embedder.dimension,
                visual_dim=self.visual_embedder.dimension,
            )
        self._embeddings_loaded = load_embeddings
        if self._reranker_arg is None:
            self.reranker = reranker_from_config(self.config, load_embeddings, self.text_embedder)
        await self._backfill_raw_index()

        self._retrieval = RetrievalEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder,
            # Drop the visual embedder when embeddings are off (lite profile) or
            # when visual is disabled (NullVisualEmbedder), so the visual layer
            # is skipped cleanly instead of failing per-query.
            visual_embedder=(
                self.visual_embedder if (load_embeddings and self._visual_enabled) else None
            ),
            embeddings_available=load_embeddings,
            config=self.config,
            log_dir_for=self.jsonl.namespace_dir,
            reranker=self.reranker,
            graph_available=self._graph_available,
            llm_enabled=llm_available(self.config),
        )
        self._compaction = CompactionEngine(
            sqlite=self.sqlite,
            graph=self.graph,
            vector=self.vector,
            text_embedder=self.text_embedder if load_embeddings else None,
            visual_embedder=self.visual_embedder,
            config=self.config,
            graph_available=self._graph_available,
            vectors_available=load_embeddings,
        )

    async def initialize_lite(self) -> None:
        """Initialize without loading embedding models (for testing or lightweight use)."""
        await self.initialize(load_embeddings=False)

    async def close(self) -> None:
        """Close all storage backends."""
        await self.sqlite.close()
        self.graph.close()
        self.vector.close()

    async def _backfill_raw_index(self) -> None:
        """Index turns logged before the raw-turn index existed."""
        if not self.sqlite.fts_enabled or await self.sqlite.raw_index_size():
            return
        refs = await self.sqlite.iter_raw_refs()
        if not refs:
            return
        added = 0
        for ref in refs:
            try:
                entry = self.jsonl.read_entry(ref["file_path"], ref["byte_offset"] or 0)
            except Exception:
                continue
            if entry.id == ref["id"] and entry.content:
                await self.sqlite.add_raw_fts(ref["rowid"], entry.content)
                added += 1
        await self.sqlite.db.commit()
        logger.info("Backfilled %d turns into the raw-turn index", added)

    # ── Core operations ──

    async def process_turn(
        self, session_id: str, turn: int, content: str,
        role: str = "assistant", token_count: int = 0,
        model: str = "", provider: str = "", namespace: str = "default",
        timestamp: str | None = None,
    ) -> Memory | None:
        """Log a turn and decide whether to curate it into a memory.

        This is the main entry point called at the end of each turn.
        ``namespace`` isolates this turn's data to a tenant (e.g. a user or
        agent id). ``timestamp`` (ISO 8601) is when the turn happened; it
        defaults to now and is what time filters and recency use.

        Returns the Memory if one was created, else None. The turn is
        retrievable from the raw-turn index either way.
        """
        # 1. Create and persist raw log entry
        entry = RawLogEntry(
            namespace=namespace,
            session_id=session_id,
            turn=turn,
            content=content,
            role=role,
            token_count=token_count,
            model=model,
            provider=provider,
        )
        if timestamp:
            entry.timestamp = timestamp
        file_path, byte_offset = self.jsonl.append(entry)

        # 2. Index the raw turn (metadata + full text) so it is retrievable
        #    whether or not it is curated.
        await self.sqlite.index_raw_log(
            entry_id=entry.id,
            session_id=session_id,
            turn=turn,
            timestamp=entry.timestamp,
            file_path=file_path,
            byte_offset=byte_offset,
            namespace=namespace,
            content=content,
            role=role,
        )
        raw_vector = None
        if self._embeddings_loaded and self.config.get("index_raw_vectors", True) and content.strip():
            try:
                raw_vector = self.text_embedder.embed(content)
                self.vector.upsert_raw_vector(
                    entry.id, raw_vector, namespace=namespace,
                    session_id=session_id, timestamp=entry.timestamp,
                )
            except Exception:
                logger.exception("Failed to embed raw turn %s", entry.id)

        # 3. Decide (restart-safe first-turn detection via the raw index)
        is_first = not await self.sqlite.session_has_prior_turns(session_id, namespace, entry.id)
        decision, memory = await make_save_decision(
            entry, is_first_turn=is_first, sqlite=self.sqlite,
            policy=self.policy, config=self.config,
        )

        # 4. Log the decision (with features, for the decision model)
        await self.sqlite.log_save_decision(decision)
        if memory is None:
            return None

        # 5. Save the memory
        memory.namespace = namespace
        await self.sqlite.save_memory(memory)

        # 6. Graph node, entity mentions
        if self._graph_available:
            try:
                self.graph.add_memory_node(
                    memory_id=memory.id,
                    summary=memory.summary or "",
                    tier=memory.tier,
                    salience=memory.salience,
                    valence=memory.valence,
                    compaction_gen=memory.compaction_gen,
                    created_at=memory.created_at,
                    namespace=namespace,
                )
                for name, etype in memory.entities:
                    key = entity_key(name)
                    if not key:
                        continue
                    eid = f"{namespace}::{key}"
                    self.graph.add_entity_node(eid, name=name, entity_type=etype or "")
                    self.graph.add_mentions(memory.id, eid)
                memory.graph_node_id = memory.id
                await self.sqlite.update_memory_graph_ref(memory.id, memory.id)
            except Exception:
                logger.exception("Graph update failed for memory %s", memory.id)

        # 7. Text embedding + write-time links to similar memories
        if not self._embeddings_loaded:
            return memory
        try:
            text_vector = (
                raw_vector if raw_vector is not None and memory.content == content
                else self.text_embedder.embed(memory.content)
            )
            point_id = self.vector.upsert_text_vector(
                memory_id=memory.id,
                vector=text_vector,
                tier=memory.tier,
                valence=memory.valence,
                arousal=memory.arousal,
                session_id=session_id,
                created_at=memory.created_at,
                namespace=namespace,
            )
            memory.vector_id = point_id
            await self.sqlite.update_memory_vector_ref(memory.id, point_id)
            if self._graph_available:
                self._link_similar(memory, point_id)
        except Exception:
            logger.exception("Failed to create text embedding for memory %s", memory.id)

        # 8. Visual layer — generate scene description and CLIP embedding for
        # salient memories (skipped when visual is disabled / text-only profile)
        if (self._visual_enabled and llm_available(self.config)
                and memory.salience > self.config["visual_salience_threshold"]):
            await self._generate_visual_layer(memory)

        return memory

    def _link_similar(self, memory: Memory, point_id: str) -> None:
        """A-MEM-style write-time linking: connect to the nearest existing memories."""
        threshold = self.config.get("link_similarity_threshold", 0.8)
        now = datetime.now(timezone.utc).isoformat()
        for nb in self.vector.nearest_memories(
            point_id, limit=self.config.get("link_max_neighbours", 3), namespace=memory.namespace,
        ):
            if nb["score"] >= threshold:
                try:
                    self.graph.add_relates_to(
                        memory.id, nb["memory_id"], weight=nb["score"],
                        relationship_type="similar", created_at=now,
                    )
                except Exception:
                    logger.debug("Link %s -> %s failed", memory.id, nb["memory_id"])

    async def retrieve(
        self, query: str, session_id: str | None = None,
        top_k: int | None = None, namespace: str = "default",
        turn: int | None = None, since: str | None = None, until: str | None = None,
    ) -> list[Memory]:
        """Retrieve ranked memories and raw turns for ``query``.

        Results are isolated to ``namespace``. Uncurated turns come back with
        ``kind == "raw"`` and an id of ``raw:<raw_log_id>``. Every result
        carries the ``retrieval_id`` to pass to :meth:`record_response` or
        :meth:`mark_used`. The ranked list is logged for policy learning (A4).
        """
        if self._retrieval is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")

        top_k = top_k or self.config["retrieval_top_k"]
        trace = await self._retrieval.retrieve_detailed(
            query=query, session_id=session_id, top_k=top_k, namespace=namespace,
            since=since, until=until,
        )
        memories = trace.memories
        retrieval_id = str(uuid.uuid4())
        for m in memories:
            m.retrieval_id = retrieval_id

        if self.config.get("policy_logging_enabled"):
            try:
                await self.sqlite.log_retrieval_decision(
                    decision_id=retrieval_id,
                    session_id=session_id or "",
                    turn=turn,
                    query=query,
                    decided_at=datetime.now(timezone.utc).isoformat(),
                    layers_queried=trace.layers_used,
                    graph_depth=self.config["graph_traversal_depth"],
                    mood_weight=self.config["mood_congruent_weight"],
                    top_k=top_k,
                    memory_ids=[m.id for m in memories],
                    return_count=len(memories),
                    namespace=namespace,
                    top_score=trace.top_score,
                    lexical_hit=trace.lexical_hit,
                    miss=trace.miss,
                    hits=[
                        {
                            "rank": i + 1,
                            "memory_id": m.id if m.kind == "memory" else None,
                            "raw_log_id": m.raw_log_id,
                            "score": m.score,
                            "layers": m.matched_layers,
                        }
                        for i, m in enumerate(memories)
                    ],
                )
            except Exception:
                logger.exception("Failed to log retrieval decision")

        return memories

    async def replay_raw(self, query: str, namespace: str, k: int) -> list[str]:
        """Rank raw turns for a past query without side effects (for labelling).

        Uses only the raw-turn layers, so curated and uncurated turns are
        treated identically, and the reranker when one is configured.
        """
        if self._retrieval is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")
        layers = ["raw"] + (["raw_semantic"] if self._embeddings_loaded else [])
        trace = await self._retrieval.retrieve_detailed(
            query, top_k=k, namespace=namespace, enable_mood_congruent=False,
            enable_visual=False, record_access=False, layers=layers,
        )
        return [m.raw_log_id for m in trace.memories]

    async def record_response(self, retrieval_id: str, response_text: str) -> list[dict]:
        """Attribute which retrieved items a response used (outcome signal)."""
        return await attribute_usage(self.sqlite, retrieval_id, response_text, config=self.config)

    async def mark_used(self, retrieval_id: str, memory_ids: list[str]) -> list[dict]:
        """Explicit feedback: the caller used these returned items."""
        return await mark_used(self.sqlite, retrieval_id, memory_ids)

    async def run_compaction(self, trigger: str = "scheduled") -> CompactionResult:
        """Run a sleep cycle.

        Phases:
          1. Decay refresh, deduplication and validated consolidation
          2. Keyword reweighting from graph structure (A2.5)
          3. Exploratory association discovery (A3, scheduled/manual only)
          4. Outcome assessment (A4)
          5. Save-policy training and gated promotion (A5), when data allows
        """
        if self._compaction is None:
            raise RuntimeError("MemoryManager not initialized — call initialize() first")

        result = await self._compaction.run(trigger=trigger)
        notes = [result.notes or ""]

        if self._graph_available:
            try:
                result.keywords_updated = await reweight_keywords_from_graph(self.sqlite, self.graph)
            except Exception:
                logger.exception("Keyword reweighting failed")

        if (trigger in ("scheduled", "manual") and self.config.get("dream_enabled")
                and self._graph_available and self._embeddings_loaded
                and llm_available(self.config)):
            try:
                discoveries = await exploratory_walk(
                    sqlite=self.sqlite,
                    graph=self.graph,
                    vector=self.vector,
                    text_embedder=self.text_embedder,
                )
                if discoveries:
                    result.edges_discovered = await commit_discoveries(
                        discoveries, self.graph, self.sqlite,
                    )
            except Exception:
                logger.exception("Dream exploration failed")

        if self.config.get("policy_logging_enabled"):
            try:
                await assess_save_outcomes(self.sqlite, config=self.config, replay=self.replay_raw)
                await assess_retrieval_outcomes(self.sqlite, config=self.config)
            except Exception:
                logger.exception("Outcome assessment failed")

        auto_train = self.config.get("policy_auto_train")
        if auto_train == "auto":
            auto_train = self.reranker is not None
        if auto_train:
            try:
                from agent_memory.policy.training import should_retrain

                if await should_retrain(self.sqlite, self.config):
                    report = await self.train_policy()
                    notes.append(f"policy: {report['reason']}")
            except Exception:
                logger.exception("Policy training failed")

        result.notes = "; ".join(n for n in notes if n)
        return result

    async def train_policy(self, force: bool = False, min_examples: int | None = None) -> dict:
        """Train the save-decision model from labelled decisions.

        The model is only activated if it beats the deployed policy on held-out
        (most recent) decisions; see ``agent_memory.policy.training``.
        """
        from agent_memory.policy.training import train_save_policy

        report = await train_save_policy(
            self.sqlite, self.policy_dir, config=self.config,
            force=force, min_examples=min_examples,
        )
        if report.promoted:
            self.policy.reload()
        return report.to_dict()

    async def get_memory(
        self, memory_id: str, namespace: str | None = None,
    ) -> Memory | None:
        """Fetch a single memory (or ``raw:<id>`` turn) and log the access.

        If ``namespace`` is given, the lookup is restricted to that namespace
        (returns None for an item belonging to a different tenant).
        """
        if memory_id.startswith("raw:"):
            raw_id = memory_id[4:]
            row = (await self.sqlite.get_raw_turns([raw_id])).get(raw_id)
            if row is None or (namespace is not None and row["namespace"] != namespace):
                return None
            return Memory(
                id=memory_id, created_at=row["timestamp"], updated_at=row["timestamp"],
                content=row.get("content") or "", raw_log_id=raw_id,
                namespace=row["namespace"], session_id=row["session_id"], turn=row["turn"],
                salience=0.0, tier="raw", role=row.get("role") or "assistant",
                event_time=row["timestamp"], kind="raw",
            )

        mem = await self.sqlite.get_memory(memory_id, namespace=namespace)
        if mem is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        mem.access_count += 1
        mem.last_accessed = now
        mem.decay_score = compute_decay(
            datetime.fromisoformat(now), mem.access_count,
            arousal=mem.arousal, surprise=mem.surprise,
            is_semantic=mem.is_semantic, config=self.config,
        )
        await self.sqlite.record_accesses([(memory_id, "primary", None, None)], now)
        return mem

    async def delete_memory(
        self, memory_id: str, namespace: str | None = None, purge_raw: bool = False,
    ) -> bool:
        """Delete a curated memory from every store.

        With ``purge_raw``, the source turn is also removed from the raw-turn
        index (otherwise it stays retrievable as raw evidence). The JSONL log
        line itself is never rewritten; use :meth:`purge_namespace` to remove a
        tenant's logs entirely.
        """
        if memory_id.startswith("raw:"):
            raw_id = memory_id[4:]
            row = (await self.sqlite.get_raw_turns([raw_id])).get(raw_id)
            if row is None or (namespace is not None and row["namespace"] != namespace):
                return False
            await self._purge_raw(raw_id)
            return True
        mem = await self.sqlite.delete_memory(memory_id, namespace=namespace)
        if mem is None:
            return False
        if self._graph_available:
            try:
                self.graph.delete_memory_node(memory_id)
            except Exception:
                logger.debug("Graph delete failed for %s", memory_id)
        if self._embeddings_loaded:
            try:
                self.vector.delete_memory(memory_id)
            except Exception:
                logger.debug("Vector delete failed for %s", memory_id)
        if purge_raw:
            await self._purge_raw(mem.raw_log_id)
        return True

    async def _purge_raw(self, raw_id: str) -> None:
        await self.sqlite.purge_raw(raw_id)
        if self._embeddings_loaded:
            try:
                self.vector.delete_raw(raw_id)
            except Exception:
                logger.debug("Raw vector delete failed for %s", raw_id)

    async def purge_namespace(self, namespace: str, delete_logs: bool = True) -> dict:
        """Delete everything stored for a tenant (memories, indexes, decision
        logs, graph, vectors and — by default — its JSONL log files)."""
        info = await self.sqlite.purge_namespace(namespace)
        if self._graph_available:
            try:
                self.graph.delete_namespace(namespace)
            except Exception:
                logger.exception("Graph purge failed for namespace %s", namespace)
        if self._embeddings_loaded:
            try:
                self.vector.delete_namespace(namespace)
            except Exception:
                logger.exception("Vector purge failed for namespace %s", namespace)
        removed = 0
        if delete_logs and namespace != "default":
            removed = self.jsonl.delete_files(info["files"])
        return {
            "memories_deleted": len(info["memory_ids"]),
            "raw_turns_deleted": len(info["raw_log_ids"]),
            "log_files_deleted": removed,
        }

    # ── Visual layer ──

    async def _generate_visual_layer(self, memory: Memory) -> None:
        """Generate a scene description and CLIP embedding for a memory."""
        try:
            from agent_memory.llm.client import llm_complete
            scene = await llm_complete(
                f"Generate an abstract scene description for this memory:\n\n<memory_content>\n{memory.content}\n</memory_content>",
                system=self.config["prompts"]["scene_description"],
            )
            memory.scene_description = scene.strip()

            spatial_bytes = self.visual_embedder.embed_to_bytes(memory.scene_description)
            memory.spatial_embedding = spatial_bytes

            visual_vector = self.visual_embedder.embed(memory.scene_description)
            self.vector.upsert_visual_vector(
                memory_id=memory.id,
                vector=visual_vector,
                session_id=memory.session_id,
                created_at=memory.created_at,
                namespace=memory.namespace,
            )

            await self.sqlite.update_memory_visual(
                memory.id, memory.scene_description, spatial_bytes,
            )
        except Exception:
            logger.exception("Visual layer generation failed for memory %s", memory.id)
